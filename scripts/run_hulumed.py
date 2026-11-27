"""Phase 3 comparison (hulumed env): local Hulu-Med inference on HU CT volumes.

Third model for the same task as scripts/run_medgemma.py. Slices are built by OUR code
(lung_ct.vlm.ct_representation: explicit reorientation to radiological display,
superior->inferior, HU windowing) - never by Hulu-Med's own NIfTI loader, which min-max
scales each slice and does not reorient. Representation, prompt and decoding come from
configs/hulumed_ct_v1.json. Prompt text and answer schema are identical to MedGemma's
(same prompt_sha256); following Hulu-Med's documented 3D usage, the slices are passed as
one ordered "video" block between the instruction and the query (no per-slice labels).

Hulu-Med loads with trust_remote_code: the model repository's own Python code runs.
The revision is pinned in the config after the downloaded code has been reviewed.

Output layout matches run_medgemma.py, so scripts/score_vlm_runs.py scores both:
  model_input/slice_XX.jpg   exact images given to the processor (pre-resize)
  model_input.json           slice positions, world coordinates, hashes, prompt hash
  vlm_raw.txt                complete raw response (written before parsing)
  vlm_result.json            parse status, errors/warnings, runtime, tokens, GPU memory
  vlm_parsed.json            only if the response is schema-valid
Run-level: run_manifest.json, batch_results.csv, failures.csv.

The adapter reads only the CT path and HU flag from the Phase 1 QC manifest. It never
reads reference masks or labels, and the prompt contains no case identifiers.

Usage (Git Bash):
    environments/hulumed/Scripts/python.exe scripts/run_hulumed.py \
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
MODEL_CACHE = ROOT / "models" / "hulumed" / "hub"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import nibabel as nib  # noqa: E402

from lung_ct.vlm.ct_representation import build_representation  # noqa: E402
from lung_ct.vlm.prompting import parse_response, prompt_sha256  # noqa: E402


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


def patch_vision_sdpa(model) -> int:
    """Run the vision encoder's SDPA attention segment by segment.

    The reviewed remote VisionSdpaAttention builds one dense [N, N] block-diagonal mask over
    ALL patches of ALL slices (N ~ 57k for 85 slices) and runs attention over it: O(N^2)
    memory (~197 GiB). Attention with a block-diagonal mask is exactly attention computed
    separately inside each block, so this replacement gives the same result (it is what the
    flash-attention path does with cu_seqlens) without the dense mask. Returns the number of
    patched modules (0 means the remote code changed and the patch did not apply).
    """
    import sys
    import types

    import torch
    import torch.nn.functional as F

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
        rope = sys.modules[type(self).__module__].apply_rotary_pos_emb_vision
        n = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(n, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(n, self.num_heads, self.head_dim)
        q = rope(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = rope(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        out = torch.empty_like(q)
        bounds = cu_seqlens.tolist()
        for a, b in zip(bounds[:-1], bounds[1:]):
            qs, ks, vs = (t[a:b].transpose(0, 1) for t in (q, k, v))
            out[a:b] = F.scaled_dot_product_attention(qs, ks, vs, dropout_p=0.0).transpose(0, 1)
        return self.out_proj(out.reshape(n, -1))

    count = 0
    for m in model.modules():
        if type(m).__name__ == "VisionSdpaAttention":
            m.forward = types.MethodType(forward, m)
            count += 1
    return count


def load_model(mcfg: dict):
    import torch
    import transformers
    from huggingface_hub import snapshot_download

    if mcfg.get("quantization"):
        raise NotImplementedError("quantization is not part of hulumed_ct_v1")
    if not mcfg.get("revision"):
        raise RuntimeError("model.revision must be pinned (reviewed remote code) before loading")
    local = Path(snapshot_download(mcfg["model_id"], revision=mcfg["revision"],
                                   cache_dir=str(MODEL_CACHE)))
    model = transformers.AutoModelForCausalLM.from_pretrained(
        local, trust_remote_code=True, torch_dtype=getattr(torch, mcfg["dtype"]),
        device_map=mcfg["device_map"], attn_implementation=mcfg["attn_implementation"])
    processor = transformers.AutoProcessor.from_pretrained(local, trust_remote_code=True)
    model.eval()
    n_patched = patch_vision_sdpa(model) if mcfg["attn_implementation"] == "sdpa" else 0
    if mcfg["attn_implementation"] == "sdpa" and n_patched == 0:
        raise RuntimeError("vision SDPA patch did not apply (remote code changed?)")
    device_map = {k: str(v) for k, v in (getattr(model, "hf_device_map", None) or {}).items()}
    if not device_map:
        device_map = {"": str(model.device)}
    weights = {p.name: sha256_file(p) for p in sorted(local.glob("*.safetensors"))}
    code = {p.name: sha256_file(p) for p in sorted(local.glob("*.py"))}
    info = {"local_snapshot": str(local), "weights_sha256": weights,
            "remote_code_sha256": code,
            "hf_device_map": device_map,
            "offloaded_modules": [k for k, v in device_map.items() if v in ("cpu", "disk")],
            "model_dtype": str(model.dtype),
            "processor_class": type(processor).__name__,
            "vision_sdpa_patched_modules": n_patched,
            "vision_sdpa_patch": "per-segment SDPA, exact equivalent of the block-diagonal mask",
            "resolved_revision": local.name,
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
    pc = cfg["prompt"]
    conversation = [{"role": "user", "content": [
        {"type": "text", "text": pc["instruction"]},
        {"type": "video", "num_frames": len(images)},
        {"type": "text", "text": pc["query"]}]}]
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
    hcfg = cfg["model"]
    # Reviewed remote processing_hulumed.py (rev in config): conversation= path; images are
    # (modal, frames) pairs as its own _gather_multimodal_data builds them; a "video" block
    # makes the processor pick merge_size 2 and return modals=["video"] for generate().
    inputs = processor(conversation=conversation, images=[("video", images)],
                       add_system_prompt=hcfg["add_system_prompt"], add_generation_prompt=True,
                       return_tensors="pt")
    ids = inputs["input_ids"]
    model_input["processor_output"] = {
        "input_tokens": int(ids.shape[1]),
        "image_tokens": None,
        "pixel_values_shape": list(inputs["pixel_values"].shape) if "pixel_values" in inputs else None,
        "merge_sizes": inputs["merge_sizes"].tolist() if "merge_sizes" in inputs else None,
        "modals": inputs.get("modals"),
        "add_system_prompt": hcfg["add_system_prompt"],
        "message_layout": "instruction text, one video block (num_frames = n slices), query text",
    }
    (case_dir / "model_input.json").write_text(json.dumps(model_input, indent=2))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=gcfg["do_sample"],
                             max_new_tokens=gcfg["max_new_tokens"], use_cache=True,
                             pad_token_id=processor.tokenizer.eos_token_id)
    elapsed = time.time() - t0
    # Hulu-Med generates from embeddings and returns only new tokens; strip a prompt echo
    # if a version returns it.
    seq = out[0]
    n_in = ids.shape[1]
    if seq.shape[0] > n_in and torch.equal(seq[:n_in].cpu(), ids[0].cpu()):
        seq = seq[n_in:]
    new = seq
    text = processor.tokenizer.decode(new, skip_special_tokens=True).strip()
    (case_dir / "vlm_raw.txt").write_text(text, encoding="utf-8")  # before any parsing

    eos = model.generation_config.eos_token_id or processor.tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    truncated = len(new) >= gcfg["max_new_tokens"] and len(new) > 0 and int(new[-1]) not in eos
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
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "hulumed_ct_v1.json")
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
    tag = "P3_hulumed_dry" if args.dry_run else "P3_hulumed"
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
