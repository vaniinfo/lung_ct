"""CT-CLIP zero-shot scores for the 18 CT-RATE abnormalities on local CT volumes (ct_clip env, GPU). Local only.

Model: CT-CLIP (Hamamci et al.), checkpoint CT-CLIP_v2.pt from the gated CT-RATE repository, code from
github.com/ibrahimethemhamamci/CT-CLIP (commit recorded in the run manifest). Licence CC BY-NC-SA 4.0, research use.

Published procedure, followed here (scripts/zero_shot.py and scripts/data_inference_nii.py of that repository):
  * HU clipped to [-1000, 1000]; trilinear resampling to 0.75 x 0.75 mm in plane and 1.5 mm between slices;
    divided by 1000; centre crop / pad (value -1) to 480 x 480 x 240; array layout as stored in CT-RATE
  * per abnormality two prompts, "<name> is present." and "<name> is not present."; softmax over the two logits;
    the score is the "present" share
Our implementation differences:
  * our local volumes are already in HU (the original files need RescaleSlope / Intercept from the metadata), so slope 1
    and intercept 0 are used; spacing is read from the NIfTI header (written from the same metadata)
  * array layout: checked on valid_301_a_2 that the local file has exactly the layout of the original CT-RATE file
    (models/ct_clip/orientation_check_valid_301.json)
  * the checkpoint is loaded with strict=False and every missing / unexpected key is recorded (newer transformers version)
  * one scan at a time, no accelerate wrapper

Variants (--variant):
  base       CT-CLIP_v2.pt       zero-shot: two prompts per abnormality, softmax over the two logits
  vocabfine  CT_VocabFine_v2.pt  same architecture and same prompt scoring, fine-tuned with the 18 label prompts (scripts/zero_shot.py)
  classfine  CT_LiPro_v2.pt      CT-CLIP image encoder + linear layer (512 -> 18), sigmoid per abnormality; empty text input
                                 (scripts/ct_lipro_inference.py); class order = label columns of the CT-RATE labels file
The scores of the three variants are on different scales and must not be compared as numbers across variants.

A score is an uncalibrated similarity share, not a probability of disease and not a diagnosis. No threshold is applied.

Usage: environments/ct_clip/Scripts/python.exe scripts/run_ctclip.py --cases valid_301_a_2 valid_541_a_2 valid_832_a_2 valid_988_a_2
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "models" / "ct_clip" / "src" / "CT-CLIP"
CKPTS = {"base": "CT-CLIP_v2.pt", "vocabfine": "CT_VocabFine_v2.pt", "classfine": "CT_LiPro_v2.pt"}
HF_CACHE = ROOT / "models" / "ct_clip" / "hf"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.pop("HF_TOKEN", None)            # the text encoder is public; no token is used or needed here
sys.path.insert(0, str(CODE / "CT_CLIP"))
sys.path.insert(0, str(CODE / "transformer_maskgit"))

PATHOLOGIES = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion', 'Coronary artery wall calcification',
               'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela',
               'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
               'Interlobular septal thickening']       # order of scripts/zero_shot.py
TEXT_MODEL = "microsoft/BiomedVLP-CXR-BERT-specialized"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def to_tensor(path: Path):
    """Same steps as nii_img_to_tensor in the repository's data_inference_nii.py, with slope 1 / intercept 0 (file is in HU)."""
    import nibabel as nib
    import numpy as np
    import torch
    import torch.nn.functional as F
    img = nib.load(str(path))
    z = [float(v) for v in img.header.get_zooms()[:3]]
    if abs(z[0] - z[1]) > 1e-3:
        raise ValueError(f"in-plane spacing is not square: {z}")
    data = np.clip(img.get_fdata(), -1000, 1000).transpose(2, 0, 1)
    t = torch.tensor(data).unsqueeze(0).unsqueeze(0)
    cur, tgt = (z[2], z[0], z[0]), (1.5, 0.75, 0.75)
    new = [int(t.shape[2 + i] * cur[i] / tgt[i]) for i in range(3)]
    t = F.interpolate(t, size=new, mode="trilinear", align_corners=False)[0][0]
    t = (t.permute(1, 2, 0) / 1000).float()
    dh, dw, dd = 480, 480, 240
    h, w, d = t.shape
    hs, ws, ds = max((h - dh) // 2, 0), max((w - dw) // 2, 0), max((d - dd) // 2, 0)
    t = t[hs:min(hs + dh, h), ws:min(ws + dw, w), ds:min(ds + dd, d)]
    ph, pw, pd_ = (dh - t.size(0)) // 2, (dw - t.size(1)) // 2, (dd - t.size(2)) // 2
    t = F.pad(t, (pd_, dd - t.size(2) - pd_, pw, dw - t.size(1) - pw, ph, dh - t.size(0) - ph), value=-1)
    info = {"header_spacing_mm": z, "resampled_shape_xyz": [new[1], new[2], new[0]], "source_shape": list(img.shape),
            "cropped_voxels_xyz": [max(h - dh, 0), max(w - dw, 0), max(d - dd, 0)]}
    return t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0), info      # (1, 1, 240, 480, 480)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--volumes", default="data/raw/ct_rate/volumes")
    ap.add_argument("--variant", choices=list(CKPTS), default="base")
    args = ap.parse_args()
    import importlib.metadata as md
    import torch
    from transformers import BertModel, BertTokenizer
    from transformer_maskgit import CTViT
    from ct_clip import CTCLIP

    CKPT = ROOT / "models" / "ct_clip" / CKPTS[args.variant]
    run_dir = ROOT / "runs" / f"E21_ctclip_{args.variant}_{len(args.cases)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True)
    tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL, do_lower_case=True, cache_dir=str(HF_CACHE))
    text_encoder = BertModel.from_pretrained(TEXT_MODEL, cache_dir=str(HF_CACHE))
    text_encoder.resize_token_embeddings(len(tokenizer))          # as in the repository scripts
    image_encoder = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20, temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
                          dim_head=32, heads=8)
    clip = CTCLIP(image_encoder=image_encoder, text_encoder=text_encoder, dim_image=294912, dim_text=768, dim_latent=512, extra_latent_projection=False,
                  use_mlm=False, downsample_image_embeds=False, use_all_token_embeds=False)
    state = torch.load(str(CKPT), map_location="cpu")
    head = None
    if args.variant == "classfine":
        class ImageLatentsClassifier(torch.nn.Module):          # same modules and names as scripts/ct_lipro_inference.py
            def __init__(self, trained_model, latent_dim, num_classes, dropout_prob=0.3):
                super().__init__()
                self.trained_model = trained_model
                self.dropout = torch.nn.Dropout(dropout_prob)
                self.relu = torch.nn.ReLU()
                self.classifier = torch.nn.Linear(latent_dim, num_classes)

            def forward(self, *a, **k):
                k["return_latents"] = True
                _, image_latents, _ = self.trained_model(*a, **k)
                return self.classifier(self.dropout(self.relu(image_latents)))
        head = ImageLatentsClassifier(clip, 512, len(PATHOLOGIES))
        res = head.load_state_dict(state, strict=False)
    else:
        res = clip.load_state_dict(state, strict=False)
    del state
    device = torch.device("cuda")
    (head or clip).to(device).eval()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
    manifest = {"run_id": run_dir.name, "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
                "packages": {p: md.version(p) for p in ("torch", "transformers", "numpy", "nibabel", "einops", "vector-quantize-pytorch")},
                "model": {"checkpoint": str(CKPT.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256": sha256(CKPT), "code": "github.com/ibrahimethemhamamci/CT-CLIP",
                          "code_commit": commit, "text_encoder": TEXT_MODEL, "missing_keys": list(res.missing_keys), "unexpected_keys": list(res.unexpected_keys)},
                "variant": args.variant,
                "prompts": None if args.variant == "classfine" else ["<name> is present.", "<name> is not present."],
                "score": ("sigmoid of the linear head output per abnormality; uncalibrated here; no threshold applied" if args.variant == "classfine"
                          else "softmax over the two logits, 'present' share; uncalibrated; no threshold applied"),
                "preprocessing": "HU clip [-1000, 1000]; trilinear to 0.75 x 0.75 x 1.5 mm; / 1000; centre crop / pad (-1) to 480 x 480 x 240",
                "cases": []}
    rows = []
    for cid in args.cases:
        rec = {"case_id": cid, "status": "ok"}
        try:
            p = ROOT / args.volumes / f"{cid}.nii.gz"
            rec["ct_sha256"] = sha256(p)
            t0 = time.time()
            x, info = to_tensor(p)
            rec["preprocess"] = info
            torch.cuda.reset_peak_memory_stats()
            scores = {}
            with torch.no_grad():
                xg = x.to(device)
                if head is not None:
                    tok = tokenizer("", return_tensors="pt", padding="max_length", truncation=True, max_length=200).to(device)
                    out = torch.sigmoid(head(tok, xg, device=device).float().reshape(-1))
                    scores = {name: float(out[i]) for i, name in enumerate(PATHOLOGIES)}
                for name in ([] if head is not None else PATHOLOGIES):
                    tok = tokenizer([f"{name} is present.", f"{name} is not present."], return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                    out = clip(tok, xg, device=device)
                    scores[name] = float(torch.softmax(out.float().reshape(-1), dim=0)[0])
            rec.update(scores=scores, seconds=round(time.time() - t0, 1), peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20))
            rows.append({"case_id": cid, **{k: round(v, 4) for k, v in scores.items()}})
            print(cid, "ok", rec["seconds"], "s", "Hiatal hernia", round(scores["Hiatal hernia"], 4), flush=True)
        except Exception as e:  # noqa: BLE001  recorded, never dropped
            rec.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
            print(cid, "FAILED", rec["error"], flush=True)
        manifest["cases"].append(rec)
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    with open(run_dir / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id"] + PATHOLOGIES)
        w.writeheader()
        w.writerows(rows)
    print("->", run_dir)
    return 0 if all(c["status"] == "ok" for c in manifest["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
