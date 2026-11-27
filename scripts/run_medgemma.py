"""Phase 3 (vlm env): local MedGemma 1.5 4B inference on HU CT volumes.

Builds the official-notebook CT representation (configs/medgemma_ct_v1.json), runs the
model with greedy decoding and a bounded output, and writes per case:
  model_input/slice_XX.jpg   exact images given to the processor (pre-resize)
  model_input.json           slice positions, world coordinates, hashes, prompt hash
  vlm_raw.txt                complete raw response (written before parsing)
  vlm_result.json            parse status, errors/warnings, runtime, tokens, GPU memory
  vlm_parsed.json            only if the response is schema-valid
Run-level: run_manifest.json, batch_results.csv, failures.csv.

The adapter reads only the CT path and HU flag from the Phase 1 QC manifest. It never
reads reference masks or labels, and the prompt contains no case identifiers.

Usage (Git Bash):
    environments/vlm/Scripts/python.exe scripts/run_medgemma.py \
        --qc-run runs/P1_qc_20cases_20261107_223515 --n 1 [--dry-run]
--dry-run builds and saves the representation and prompt without loading the model.
"""
import argparse
import csv
import hashlib
import io
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
sys.path.insert(0, str(ROOT / "src"))
MODEL_CACHE = ROOT / "models" / "medgemma" / "hub"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import nibabel as nib  # noqa: E402

from lung_ct.vlm.ct_representation import build_representation  # noqa: E402
from lung_ct.vlm.prompting import build_messages, parse_response, prompt_sha256  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def load_model(mcfg: dict):
    import torch
    import transformers
    from huggingface_hub import snapshot_download

    if mcfg.get("quantization"):
        raise NotImplementedError("quantization is not part of medgemma_ct_v1")
    local = Path(snapshot_download(mcfg["model_id"], revision=mcfg["revision"],
                                   cache_dir=str(MODEL_CACHE)))
    kw = dict(dtype=getattr(torch, mcfg["dtype"]), device_map=mcfg["device_map"],
              offload_buffers=mcfg["offload_buffers"])
    processor = transformers.AutoProcessor.from_pretrained(
        local, use_fast=mcfg["use_fast_processor"], **kw)
    model = transformers.AutoModelForImageTextToText.from_pretrained(local, **kw)
    model.eval()
    device_map = {k: str(v) for k, v in (getattr(model, "hf_device_map", None) or {}).items()}
    if not device_map:  # single-device placement leaves hf_device_map unset
        device_map = {"": str(model.device)}
    weights = {p.name: sha256_file(p) for p in sorted(local.glob("*.safetensors"))}
    info = {"local_snapshot": str(local), "weights_sha256": weights,
            "hf_device_map": device_map,
            "offloaded_modules": [k for k, v in device_map.items() if v in ("cpu", "disk")],
            "model_dtype": str(model.dtype),
            "image_processor_class": type(processor.image_processor).__name__,
            "gpu_memory_after_load_mb": round(torch.cuda.memory_allocated() / 2**20)}
    return processor, model, info


def infer_case(cid: str, ct_path: Path, cfg: dict, case_dir: Path, processor, model) -> dict:
    from PIL import Image

    rec = {"case_id": cid, "ct_file": str(ct_path.relative_to(ROOT)),
           "ct_sha256": sha256_file(ct_path), "status": "ok", "issues": []}
    rep = build_representation(nib.load(str(ct_path)), cfg["representation"])
    in_dir = case_dir / "model_input"
    in_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if cfg["representation"]["image_format"] == "jpeg" else "png"
    for s, data in zip(rep.slices, rep.images):
        (in_dir / f"slice_{s['sample_number']:02d}.{ext}").write_bytes(data)
    # The model receives the decoded encoded bytes, as in the notebook (data-URL images).
    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in rep.images]
    messages = build_messages(images, [s["label"] for s in rep.slices], cfg["prompt"])
    model_input = {"case_id": cid, "representation": rep.info, "slices": rep.slices,
                   "prompt_version": cfg["prompt"]["version"],
                   "prompt_sha256": prompt_sha256(cfg["prompt"])}
    rec["representation"] = rep.info

    if processor is None:  # dry run
        (case_dir / "model_input.json").write_text(json.dumps(model_input, indent=2))
        rec["status"] = "dry_run"
        return rec

    import torch
    gcfg = cfg["generation"]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, continue_final_message=False,
        return_tensors="pt", tokenize=True, return_dict=True)
    ids = inputs["input_ids"]
    image_token_id = getattr(model.config, "image_token_id", None) or \
        getattr(model.config, "image_token_index", None)
    model_input["processor_output"] = {
        "input_tokens": int(ids.shape[1]),
        "image_tokens": int((ids == image_token_id).sum()) if image_token_id is not None else None,
        "pixel_values_shape": list(inputs["pixel_values"].shape) if "pixel_values" in inputs else None,
        "pixel_values_dtype_on_device": "bfloat16",
    }
    (case_dir / "model_input.json").write_text(json.dumps(model_input, indent=2))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        inputs = inputs.to(model.device, dtype=torch.bfloat16)
        out = model.generate(**inputs, do_sample=gcfg["do_sample"],
                             max_new_tokens=gcfg["max_new_tokens"])
    elapsed = time.time() - t0
    new = out[0, ids.shape[1]:]
    text = processor.decode(new, skip_special_tokens=True)
    (case_dir / "vlm_raw.txt").write_text(text, encoding="utf-8")  # before any parsing

    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    truncated = len(new) >= gcfg["max_new_tokens"] and int(new[-1]) not in eos
    parsed = parse_response(text, truncated, cfg["prompt"].get("schema"))
    rec.update({
        "parse_status": parsed["status"],
        "extraction_method": parsed["extraction_method"],
        "errors": parsed["errors"], "warnings": parsed["warnings"],
        "truncated": truncated,
        "input_tokens": model_input["processor_output"]["input_tokens"],
        "image_tokens": model_input["processor_output"]["image_tokens"],
        "output_tokens": int(len(new)),
        "generate_seconds": round(elapsed, 1),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_gpu_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
    })
    (case_dir / "vlm_result.json").write_text(json.dumps(
        {**rec, **{k: v for k, v in parsed.items() if k == "unvalidated_object"}}, indent=2))
    if parsed["status"] == "valid":
        (case_dir / "vlm_parsed.json").write_text(json.dumps(parsed["parsed"], indent=2))
    else:
        rec["status"] = "parse_failure"
        rec["issues"].append(f"{parsed['status']}: {'; '.join(parsed['errors'])}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qc-run", required=True, type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--n", type=int)
    g.add_argument("--cases", nargs="+")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "medgemma_ct_v1.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    qc = json.loads((ROOT / args.qc_run / "run_manifest.json").read_text())
    qc_cases = {c["case_id"]: c for c in qc["cases"]}
    order = args.cases if args.cases else sorted(qc_cases)
    eligible, skipped = [], []
    for cid in order:
        c = qc_cases.get(cid)
        if c is None or c["status"] == "failed":
            skipped.append({"case_id": cid, "reason": "absent or failed in Phase 1 QC"})
        elif not c["ct"]["hu_valid"]:  # windows are HU-defined; no override
            skipped.append({"case_id": cid, "reason": f"non-HU: {c['ct']['hu_reason']}"})
        else:
            eligible.append(cid)
    if args.n:
        eligible = eligible[:args.n]

    import importlib.metadata as md
    tag = "P3_vlm_dry" if args.dry_run else "P3_vlm"
    run_id = f"{tag}_{len(eligible)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    pkgs = ("torch", "torchvision", "transformers", "accelerate", "huggingface_hub", "tokenizers",
            "pillow", "nibabel", "numpy")
    manifest = {
        "run_id": run_id,
        "provenance": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "qc_run": str(args.qc_run),
            "config_file": str(cfg_path.relative_to(ROOT)),
            "config_sha256": sha256_file(cfg_path),
            "prompt_sha256": prompt_sha256(cfg["prompt"]),
            "python": platform.python_version(),
            "packages": {p: md.version(p) for p in pkgs},
            "dry_run": args.dry_run,
        },
        "config": cfg,
        "skipped": skipped,
        "cases": [],
    }
    print(f"{run_id}: {len(eligible)} cases, {len(skipped)} skipped")

    processor = model = None
    if not args.dry_run:
        import torch
        manifest["provenance"]["gpu"] = torch.cuda.get_device_name(0)
        manifest["provenance"]["cuda_runtime"] = torch.version.cuda
        try:
            processor, model, minfo = load_model(cfg["model"])
        except Exception as e:  # e.g. gated repo access not granted
            manifest["model_load_error"] = f"{type(e).__name__}: {e}"
            manifest["provenance"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"MODEL LOAD FAILED: {manifest['model_load_error']}")
            return 2
        manifest["model_info"] = minfo
        if minfo["offloaded_modules"]:
            print(f"WARNING offloaded modules: {minfo['offloaded_modules']}")

    for cid in eligible:
        print(f"[{cid}] ...", end=" ", flush=True)
        case_dir = run_dir / cid
        case_dir.mkdir()
        try:
            rec = infer_case(cid, ROOT / qc_cases[cid]["ct"]["file"], cfg, case_dir,
                             processor, model)
        except Exception as e:
            rec = {"case_id": cid, "status": "failed", "issues": [f"{type(e).__name__}: {e}"],
                   "traceback": traceback.format_exc()}
            (case_dir / "vlm_result.json").write_text(json.dumps(rec, indent=2))
        manifest["cases"].append(rec)
        print(rec["status"], {k: rec[k] for k in ("parse_status", "input_tokens",
                                                   "output_tokens", "generate_seconds",
                                                   "peak_gpu_memory_mb") if k in rec},
              "; ".join(rec["issues"]))

    manifest["provenance"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    cols = ["case_id", "status", "parse_status", "truncated", "n_images", "input_tokens",
            "image_tokens", "output_tokens", "generate_seconds", "peak_gpu_memory_mb"]
    with open(run_dir / "batch_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in manifest["cases"]:
            w.writerow({**r, "n_images": r.get("representation", {}).get("n_images")})
    with open(run_dir / "failures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "stage", "reason"])
        for s in skipped:
            w.writerow([s["case_id"], "eligibility", s["reason"]])
        for r in manifest["cases"]:
            if r["status"] in ("failed", "parse_failure"):
                w.writerow([r["case_id"], r["status"], "; ".join(r["issues"])])

    bad = sum(r["status"] in ("failed", "parse_failure") for r in manifest["cases"])
    print(f"done: {len(manifest['cases']) - bad} ok, {bad} failed/unparsed -> {run_dir}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
