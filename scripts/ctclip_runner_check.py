"""Bounded implementation check of scripts/run_ctclip.py (ct_clip env, GPU). One scan; changes no result.

Checks, each recorded with its evidence in <out>/runner_check.json:
  1. checkpoint identity   local SHA-256 and size reproduce the Git-LFS pointer blob id listed by the repository
                           (sha1 of "blob <n>\\0" + pointer text); every tensor in the loaded model equals the checkpoint tensor
  2. condition ordering    our list == list in the authors' zero_shot.py == list in ct_vocabfine_train.py == label columns of the
                           CT-RATE labels file (the order the ClassFine head was trained on, by the authors' code)
  3. outputs before normalisation   the two logits per condition (zero-shot, VocabFine) and the 18 logits (ClassFine)
  4. normalisation dimensions       shape of the model output, the axis the softmax runs over, and that each pair sums to 1
  5. authors' inference path        (a) the authors' own preprocessing function (data_inference_nii.nii_img_to_tensor) on the ORIGINAL
                                    CT-RATE file with the CT-RATE metadata row, against our preprocessing of the local HU file;
                                    (b) the authors' apply_softmax on the same logits; (c) the authors' strict checkpoint loader
     What is NOT reproduced: the authors' CTClipInference class (accelerate wrapper, data loader, report files).
Usage: environments/ct_clip/Scripts/python.exe scripts/ctclip_runner_check.py --case valid_301_a_2 --original <original CT-RATE nii.gz> --out runs/E21_ctclip_pilot_20261124
"""
import argparse
import ast
import csv
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_ctclip as rc  # noqa: E402  (sets sys.path for the CT-CLIP code and the HF cache; pops HF_TOKEN)

LFS_BLOB_IDS = {"base": "3479be82358c6e1accbbacc0d892d0dba58ef186", "classfine": "a0a146196692016f7bedbdf3931f03146683f23a",
                "vocabfine": "8e0934e3c81b4840929e815df001cc14379bf3e6"}      # "oid" of each file in the repository tree listing, read 2026-11-24


def list_in_source(path: Path, var: str) -> list:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == var for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{var} not found in {path.name}")


def function_from_source(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    import torch
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)     # the authors' function text, unchanged
    return ns[name]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default="valid_301_a_2")
    ap.add_argument("--original", required=True, help="original CT-RATE file of the case (dataset/valid/...), named <case>.nii.gz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from transformers import BertModel, BertTokenizer
    from transformer_maskgit import CTViT
    from ct_clip import CTCLIP
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    R = {"case": args.case, "checks": {}}
    scripts = rc.CODE / "scripts"

    # ---- 2. condition ordering
    labels_csv = glob.glob(str(ROOT / "data/raw/ct_rate/hf_cache/**/valid_predicted_labels.csv"), recursive=True)[0]
    with open(labels_csv, newline="", encoding="utf-8") as f:
        cols = next(csv.reader(f))[1:]
    zs, vf = list_in_source(scripts / "zero_shot.py", "pathologies"), list_in_source(scripts / "ct_vocabfine_train.py", "pathologies_all")
    R["checks"]["condition_ordering"] = {
        "ours_equals_authors_zero_shot_list": rc.PATHOLOGIES == zs, "ours_equals_authors_vocabfine_training_list": rc.PATHOLOGIES == vf,
        "ours_equals_label_file_columns": rc.PATHOLOGIES == cols, "hiatal_hernia_index": rc.PATHOLOGIES.index("Hiatal hernia"),
        "note": "zero-shot and VocabFine score a condition by its NAME in the prompt, so their order cannot be wrong. The ClassFine head has 18 unnamed outputs; "
                "their order is the label-file column order by the authors' code (one_hot_labels = columns[1:]). This is established from code, not from data: "
                "ClassFine outputs are too similar across conditions on our scans for the data to confirm the order."}

    # ---- 5a. authors' preprocessing on the original file vs ours on the local HU file
    sys.path.insert(0, str(scripts))
    import data_inference_nii as authors                                    # the authors' module, unchanged
    meta = pd.read_csv(glob.glob(str(ROOT / "data/raw/ct_rate/hf_cache/**/validation_metadata.csv"), recursive=True)[0])
    orig = Path(args.original)
    assert orig.name == f"{args.case}.nii.gz"
    xa = authors.CTReportDatasetinfer.nii_img_to_tensor(None, orig.as_posix(), meta)       # (1, 240, 480, 480)
    xo, info = rc.to_tensor(ROOT / "data/raw/ct_rate/volumes" / f"{args.case}.nii.gz")      # (1, 1, 240, 480, 480)
    d = (xa.float() - xo[0].float()).abs()
    row = meta[meta["VolumeName"] == orig.name].iloc[0]
    R["checks"]["authors_preprocessing_vs_ours"] = {
        "authors_function": "CTReportDatasetinfer.nii_img_to_tensor (scripts/data_inference_nii.py), original CT-RATE file + CT-RATE metadata row",
        "metadata_row": {k: str(row[k]) for k in ("RescaleSlope", "RescaleIntercept", "XYSpacing", "ZSpacing")},
        "authors_tensor_shape": list(xa.shape), "our_tensor_shape": list(xo.shape), "our_preprocess_info": info,
        "max_abs_difference": float(d.max()), "mean_abs_difference": float(d.mean()), "share_of_voxels_differing_by_more_than_0.001": float((d > 1e-3).float().mean()),
        "unit": "normalised intensity (HU / 1000)",
        "note": "the local file stores HU as rounded int16 with scanner padding set to -1024; after the clip at -1000 HU this should leave no difference"}
    x_auth = xa.unsqueeze(0)            # what the authors' DataLoader (batch size 1) hands to the model

    authors_softmax = function_from_source(scripts / "zero_shot.py", "apply_softmax")
    tokenizer = BertTokenizer.from_pretrained(rc.TEXT_MODEL, do_lower_case=True, cache_dir=str(rc.HF_CACHE))
    device = torch.device("cuda")
    prov = json.loads((ROOT / "models/ct_clip/PROVENANCE.json").read_text(encoding="utf-8"))["files"]
    earlier = {}
    for v, pat in (("base", "runs/E21_ctclip_zero_shot_4cases_*"), ("vocabfine", "runs/E21_ctclip_vocabfine_4cases_*"), ("classfine", "runs/E21_ctclip_classfine_4cases_*")):
        m = json.loads((Path(sorted(glob.glob(str(ROOT / pat)))[-1]) / "run_manifest.json").read_text(encoding="utf-8"))
        earlier[v] = next(c["scores"] for c in m["cases"] if c["case_id"] == args.case)

    R["checks"]["checkpoint_identity_and_loading"], R["checks"]["outputs_before_normalisation"], R["checks"]["normalisation"], R["checks"]["authors_path_scores"] = {}, {}, {}, {}
    for v, fname in rc.CKPTS.items():
        ck = ROOT / "models/ct_clip" / fname
        sha, size = rc.sha256(ck), ck.stat().st_size
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n".encode()
        blob = hashlib.sha1(b"blob " + str(len(pointer)).encode() + b"\0" + pointer).hexdigest()
        text_encoder = BertModel.from_pretrained(rc.TEXT_MODEL, cache_dir=str(rc.HF_CACHE))
        text_encoder.resize_token_embeddings(len(tokenizer))
        image_encoder = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20, temporal_patch_size=10, spatial_depth=4, temporal_depth=4, dim_head=32, heads=8)
        clip = CTCLIP(image_encoder=image_encoder, text_encoder=text_encoder, dim_image=294912, dim_text=768, dim_latent=512, extra_latent_projection=False,
                      use_mlm=False, downsample_image_embeds=False, use_all_token_embeds=False)
        state = torch.load(str(ck), map_location="cpu")
        strict_error = None
        head = None
        if v == "classfine":
            class ImageLatentsClassifier(torch.nn.Module):
                def __init__(self, trained_model, latent_dim, num_classes, dropout_prob=0.3):
                    super().__init__()
                    self.trained_model, self.dropout, self.relu = trained_model, torch.nn.Dropout(dropout_prob), torch.nn.ReLU()
                    self.classifier = torch.nn.Linear(latent_dim, num_classes)

                def forward(self, *a, **k):
                    k["return_latents"] = True
                    _, lat, _ = self.trained_model(*a, **k)
                    return self.classifier(self.dropout(self.relu(lat)))
            model = ImageLatentsClassifier(clip, 512, 18)
        else:
            model = clip
            try:
                clip.load(str(ck))                                             # the authors' loader: torch.load + strict load_state_dict
            except Exception as e:  # noqa: BLE001
                strict_error = f"{type(e).__name__}: {str(e)[:300]}"
        res = model.load_state_dict(state, strict=False)
        own = model.state_dict()
        shared = [k for k in state if k in own]
        equal = sum(bool(torch.equal(own[k].cpu(), state[k].cpu())) for k in shared)
        unexpected = {k: {"shape": list(state[k].shape), "is_arange": bool(torch.equal(state[k].flatten().long(), torch.arange(state[k].numel())))} for k in res.unexpected_keys}
        R["checks"]["checkpoint_identity_and_loading"][v] = {
            "file": fname, "bytes": size, "sha256": sha, "sha256_recorded_at_download": prov[v]["sha256"], "sha256_unchanged_since_download": sha == prov[v]["sha256"],
            "git_lfs_pointer_blob_id_computed": blob, "git_lfs_pointer_blob_id_in_repository_listing": LFS_BLOB_IDS[v], "matches_repository": blob == LFS_BLOB_IDS[v],
            "checkpoint_tensors": len(state), "model_tensors": len(own), "tensors_in_both": len(shared), "tensors_identical_after_loading": equal,
            "missing_keys": list(res.missing_keys), "unexpected_keys": unexpected,
            "authors_strict_loader": "not applicable (classifier wrapper)" if v == "classfine" else (strict_error or "loaded without error")}
        del state
        model.to(device).eval()
        ours, auth, raw, shapes = {}, {}, {}, set()
        with torch.no_grad():
            if v == "classfine":
                tok = tokenizer("", return_tensors="pt", padding="max_length", truncation=True, max_length=200).to(device)
                o1 = model(tok, xo.to(device), device=device)
                o2 = model(tok, x_auth.to(device), device=device)
                shapes.add(tuple(o1.shape))
                raw = {n: float(o1.float().reshape(-1)[i]) for i, n in enumerate(rc.PATHOLOGIES)}
                ours = {n: float(torch.sigmoid(o1.float().reshape(-1))[i]) for i, n in enumerate(rc.PATHOLOGIES)}
                auth = {n: float((1 / (1 + torch.exp(-o2.float().reshape(-1))))[i]) for i, n in enumerate(rc.PATHOLOGIES)}     # the authors' sigmoid formula
            else:
                for n in rc.PATHOLOGIES:
                    tok = tokenizer([f"{n} is present.", f"{n} is not present."], return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                    o1 = model(tok, xo.to(device), device=device)
                    o2 = model(tok, x_auth.to(device), device=device)
                    shapes.add(tuple(o1.shape))
                    raw[n] = {"logit_present": float(o1.float().reshape(-1)[0]), "logit_not_present": float(o1.float().reshape(-1)[1])}
                    ours[n] = float(torch.softmax(o1.float().reshape(-1), dim=0)[0])
                    a = authors_softmax(o2)                                       # the authors' function on the authors' tensor
                    auth[n] = float(a.detach().cpu().numpy()[0])
                    raw[n]["authors_softmax_pair_sum"] = float(a.sum())
        R["checks"]["outputs_before_normalisation"][v] = raw
        R["checks"]["normalisation"][v] = (
            {"model_output_shape": sorted(shapes), "rule": "sigmoid applied to each of the 18 logits separately; the 18 scores are not normalised against each other"} if v == "classfine" else
            {"model_output_shape": sorted(shapes), "rule": "softmax over the two logits of ONE condition (present, not present); axis 0 of a 2-element output; "
             "conditions are never normalised against each other", "all_pairs_sum_to_1": all(abs(r["authors_softmax_pair_sum"] - 1) < 1e-5 for r in raw.values())})
        R["checks"]["authors_path_scores"][v] = {
            "max_abs_difference_ours_vs_authors_path": max(abs(ours[n] - auth[n]) for n in rc.PATHOLOGIES),
            "max_abs_difference_vs_earlier_runner_output": max(abs(ours[n] - earlier[v][n]) for n in rc.PATHOLOGIES),
            "hiatal_hernia": {"ours": ours["Hiatal hernia"], "authors_path": auth["Hiatal hernia"], "earlier_run": earlier[v]["Hiatal hernia"]}}
        del model, clip
        torch.cuda.empty_cache()
    R["not_reproduced"] = ["authors' CTClipInference class and data loader (accelerate wrapper, report and label files)",
                           "authors' software versions (their code pins none; transformers 4.30.1 is mentioned in a comment, 4.44.2 is used here)"]
    (out / "runner_check.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    c = R["checks"]
    print(json.dumps({"ordering": {k: v for k, v in c["condition_ordering"].items() if k != "note"},
                      "preprocessing": {k: c["authors_preprocessing_vs_ours"][k] for k in ("max_abs_difference", "mean_abs_difference", "share_of_voxels_differing_by_more_than_0.001")},
                      "loading": {v: {k: x[k] for k in ("matches_repository", "tensors_in_both", "tensors_identical_after_loading", "missing_keys", "unexpected_keys", "authors_strict_loader")}
                                  for v, x in c["checkpoint_identity_and_loading"].items()},
                      "normalisation": c["normalisation"], "scores": c["authors_path_scores"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
