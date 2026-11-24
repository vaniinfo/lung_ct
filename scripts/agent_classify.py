"""Grounded agent, pipeline step 2 (colipri env): zero-shot finding scores with COLIPRI.

For each HU-valid case, the CT is encoded once and each finding is scored with its own
prompt pair ("No X present" / "X present"); the probability is the softmax over that pair,
exactly what colipri's ZeroShotImageClassificationPipeline returns for a two-prompt call.
Preprocessing is the package default (reorient to RAS, 2 mm, clamp -1000..1000 HU, 192^3).
Scores are uncalibrated. No reference masks or labels are read.

Writes runs/P3_colipri_<n>cases_<time>/<case>/classifier.json and run_manifest.json.
Usage (Git Bash):
  environments/colipri/Scripts/python.exe scripts/agent_classify.py \
      --qc-run runs/P1_qc_20cases_20261107_223515 --cases coronacases_001 coronacases_006
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "colipri" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qc-run", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--n", type=int)
    g.add_argument("--cases", nargs="+")
    ap.add_argument("--config", default="configs/colipri_ct_v1.json")
    args = ap.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if not cfg["model"]["revision"] or not cfg["tokenizer"].get("reviewed"):
        print("REFUSING: model revision not pinned or tokenizer remote code not reviewed (see config)")
        return 2

    qc = json.loads((ROOT / args.qc_run / "run_manifest.json").read_text(encoding="utf-8"))
    qcs = {c["case_id"]: c for c in qc["cases"]}
    order = args.cases or sorted(qcs)
    eligible = [c for c in order if c in qcs and qcs[c].get("status") != "failed" and qcs[c]["ct"].get("hu_valid")]
    if args.n:
        eligible = eligible[:args.n]
    skipped = [c for c in order if c not in eligible]

    import importlib.metadata as md

    import torch
    from huggingface_hub import hf_hub_download

    import colipri
    from colipri import get_model, get_processor

    weights = Path(hf_hub_download(cfg["model"]["repo_id"], cfg["model"]["filename"],
                                   revision=cfg["model"]["revision"]))
    # colipri's default loader (accelerate init_empty_weights + load_checkpoint_and_dispatch) leaves
    # every parameter on the "meta" device with accelerate 1.15 / torch 2.6. Build the model with
    # real tensors instead and load the checkpoint with the package's own plain loader.
    model = get_model(pretrained=False)
    # colipri 0.1.3 bug: Model(temperature: float = 1) registers softmax_temperature as an INTEGER
    # tensor, so copying the checkpoint value (0.07) into it truncates to 0 and every score becomes
    # NaN. Make it a float buffer before loading, then verify every tensor against the file.
    model.softmax_temperature = model.softmax_temperature.float()
    from safetensors.torch import load_file
    ckpt = load_file(str(weights))
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {list(missing)[:10]}, unexpected {list(unexpected)[:10]}")
    state = model.state_dict()
    bad = [k for k, v in ckpt.items() if not torch.equal(state[k].cpu(), v.to(state[k].dtype))]
    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if bad or meta:
        raise RuntimeError(f"weights not loaded: {len(bad)} differ from file (e.g. {bad[:5]}), {len(meta)} on meta")
    load_info = {"loader": "load_state_dict(strict) with float softmax_temperature",
                 "tensors_verified": len(ckpt), "softmax_temperature": float(model.softmax_temperature)}
    model = model.cuda().eval()
    processor = get_processor()
    prompts = [s for f in cfg["findings"] for s in (f["negative"], f["positive"])]
    tok_ids, tok_mask = processor.process_text(prompts)
    with torch.no_grad():
        text_emb = model.encode_text(tok_ids.to(model.device), tok_mask.to(model.device))

    run = ROOT / "runs" / f"P3_colipri_{len(eligible)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True)
    manifest = {"run_id": run.name, "config": cfg, "skipped": skipped, "cases": [],
                "provenance": {"started_utc": datetime.now(timezone.utc).isoformat(),
                               "qc_run": args.qc_run, "weights_sha256": sha256_file(weights),
                               "colipri_version": colipri.__version__ if hasattr(colipri, "__version__") else md.version("colipri"),
                               "packages": {p: md.version(p) for p in ("torch", "transformers", "torchio", "colipri")},
                               "gpu": torch.cuda.get_device_name(0), "weights_load": load_info}}
    for cid in eligible:
        ct = ROOT / qcs[cid]["ct"]["file"].replace("\\", "/")
        rec = {"case_id": cid, "ct_file": str(ct.relative_to(ROOT)), "status": "ok"}
        print(f"[{cid}] ...", end=" ", flush=True)
        try:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            imgs = processor.to_images_batch(processor.process_images(str(ct))).to(model.device)
            diag = {"image_shape": list(imgs.shape), "image_finite": bool(torch.isfinite(imgs).all()),
                    "image_min": float(imgs.nan_to_num().min()), "image_max": float(imgs.nan_to_num().max()),
                    "text_finite": bool(torch.isfinite(text_emb).all()),
                    "temperature": float(model.softmax_temperature)}
            findings, precision = None, None
            for prec in ("bf16", "fp32"):  # never write a non-finite score
                with torch.no_grad():
                    if prec == "bf16":
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            img_emb = model.encode_image(imgs, project=True, pool=True, normalize=True)
                    else:
                        img_emb = model.encode_image(imgs.float(), project=True, pool=True, normalize=True)
                    img_emb = img_emb.float()
                    diag[f"image_embedding_finite_{prec}"] = bool(torch.isfinite(img_emb).all())
                    if not diag[f"image_embedding_finite_{prec}"]:
                        continue
                    out = []
                    for i, f in enumerate(cfg["findings"]):
                        p = model.classify(img_emb, text_emb.float()[2 * i:2 * i + 2])[0]
                        out.append({"key": f["key"], "label": f["label"], "p_present": round(float(p[1]), 4),
                                    "ct_rate_label": f["ct_rate_label"]})
                    if all(math.isfinite(x["p_present"]) for x in out):
                        findings, precision = out, prec
                        break
            rec["diagnostics"] = diag
            if findings is None:
                raise RuntimeError(f"non-finite scores in bf16 and fp32; diagnostics {diag}")
            rec["precision"] = precision
            rec.update(seconds=round(time.time() - t0, 1),
                       peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 2**20))
            d = run / cid
            d.mkdir()
            (d / "classifier.json").write_text(json.dumps(
                {"case_id": cid, "model": f"{cfg['model']['repo_id']}@{cfg['model']['revision'][:8]}",
                 "config_version": cfg["config_version"], "reliability": cfg["reliability"],
                 "high_threshold": cfg["high_threshold"], "low_threshold": cfg["low_threshold"],
                 "findings": findings}, indent=2), encoding="utf-8")
            top = sorted(findings, key=lambda x: -x["p_present"])[:3]
            print("ok", precision, rec["seconds"], "s;", ", ".join(f"{t['label']} {t['p_present']:.2f}" for t in top))
        except Exception as e:  # reported, never dropped
            rec.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            print("FAILED", rec["error"])
        manifest["cases"].append(rec)
    manifest["provenance"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bad = sum(c["status"] != "ok" for c in manifest["cases"])
    print(f"done: {len(eligible) - bad} ok, {bad} failed -> {run}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
