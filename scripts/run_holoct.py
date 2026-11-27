"""Comparison model (holoct env): local HoloCT-4B on our CT-RATE cases.

HoloCT (Alibaba DAMO, MIT) is a 3D CT vision-language model with a mask decoder. Here it is an
outside comparison for the lung_ct checklist, not part of the agent: for every case it answers one
Yes/No question per CT-RATE label (configs/holoct_ct_questions_v1.json) and, optionally, lists
abnormal findings with masks. Everything runs locally and offline; HoloCT's own loader is used
(it reorients from the NIfTI affine and clips HU to -1000..1000, no min-max scaling).

The script reads only case ids, set names and CT paths. It never reads labels or report text, and
the prompts contain no case identifiers.

Output: runs/P3_holoct_<label>_<time>/
  <case_id>/answers.json        question, raw answer, parsed yes / no / unparsed, seconds
  <case_id>/findings*/          response text (+ mask_XXX.nii.gz with --grounded)
  run_manifest.json, batch_results.csv, failures.csv

Usage (Git Bash):
    environments/holoct/Scripts/python.exe scripts/run_holoct.py --set test20c [--n 1] [--findings] [--grounded]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLO_SRC = ROOT / "models" / "holoct" / "src"
os.environ["HF_HUB_OFFLINE"] = "1"          # local weights only; no network during inference
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_yes_no(text: str) -> str:
    m = re.match(r"\W*(yes|no|true|false)\b", text.strip(), flags=re.I)
    if not m:
        return "unparsed"
    return "yes" if m.group(1).lower() in ("yes", "true") else "no"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", dest="sets", nargs="+", default=[], help="set name(s) in data/raw/ct_rate/cases.json")
    ap.add_argument("--cases", nargs="+", default=[], help="explicit case ids, e.g. valid_301_a_2")
    ap.add_argument("--n", type=int, default=0, help="only the first N cases (0 = all)")
    ap.add_argument("--config", default="configs/holoct_ct_questions_v1.json")
    ap.add_argument("--model", default="models/holoct/weights/HoloCT-4B")
    ap.add_argument("--findings", action="store_true", help="also ask for a free-text list of findings")
    ap.add_argument("--grounded", action="store_true", help="also ask for findings with masks (slower)")
    ap.add_argument("--label", default="", help="added to the output folder name")
    ap.add_argument("--resume", type=Path, help="existing runs/P3_holoct_* folder: skip cases that already have answers.json")
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    all_cases = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    ids = [c for c, v in all_cases.items() if v.get("set") in args.sets] + [c for c in args.cases]
    ids = list(dict.fromkeys(ids))
    if args.n:
        ids = ids[: args.n]
    if not ids:
        raise SystemExit("no cases selected (use --set or --cases)")
    sets = {c: all_cases.get(c, {}).get("set") for c in ids}     # set name only; labels are not read
    del all_cases

    tag = args.label or "_".join(args.sets) or "cases"
    if args.resume:
        out = args.resume if args.resume.is_absolute() else ROOT / args.resume
        if not (out / "run_manifest.json").is_file():
            raise SystemExit(f"--resume: not a HoloCT run folder: {out}")
    else:
        out = ROOT / "runs" / f"P3_holoct_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out.mkdir(parents=True)
    print(f"{len(ids)} cases -> {out}", flush=True)

    sys.path.insert(0, str(HOLO_SRC))
    import gc
    import shutil
    import subprocess
    import torch
    import transformers
    from inference.dataset.demo_dataset import DemoDataset
    from inference.demo_utils import generate_case, load_model_and_processor, save_case

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    device, dtype = torch.device("cuda:0"), torch.bfloat16
    model_dir = ROOT / args.model
    t0 = time.time()
    model, processor = load_model_and_processor(model_dir, device, dtype, "sdpa")
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.0f} s, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    def ask(ds, prompt, generation, keep_mask_features):
        """One prompt. Encoder features are kept only when masks may be decoded (saves several GB)."""
        model.model.cache_for_seg_infer = keep_mask_features
        try:
            return generate_case(model, processor, ds, prompt, generation, dec["seed"], 0.5, device, dtype)
        finally:
            model.model.cached_encode_feats = None
            gc.collect()
            torch.cuda.empty_cache()

    dec = cfg["decoding"]
    gen_q = {"max_new_tokens": dec["max_new_tokens"], "do_sample": dec["do_sample"]}
    gen_f = {"max_new_tokens": 1024, "do_sample": True, "temperature": 0.15, "top_p": 0.85, "repetition_penalty": 1.15}
    try:
        code_commit = subprocess.run(["git", "-C", str(HOLO_SRC), "rev-parse", "HEAD"], capture_output=True,
                                     text=True, check=True).stdout.strip()
    except Exception:
        code_commit = "unknown"
    manifest = {
        "stage": "P3_holoct", "started_utc": datetime.now(timezone.utc).isoformat(), "cases": ids, "sets": sets,
        "config": args.config, "config_sha256": sha256_file(cfg_path), "model_dir": args.model,
        "model_repo": cfg["model"]["repo"], "holoct_code_commit": code_commit,
        "weights_sha256": {p.name: sha256_file(p) for p in sorted(model_dir.glob("*.safetensors"))},
        "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0), "offline": True, "findings": args.findings, "grounded": args.grounded,
        "note": "Comparison model, not part of the lung_ct agent. Research use only; not a diagnosis.",
    }
    if args.resume:
        old = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        for k in ("config_sha256", "weights_sha256", "holoct_code_commit"):
            if old.get(k) != manifest[k]:
                raise SystemExit(f"--resume: {k} differs from the earlier part of this run")
        manifest["started_utc"] = old["started_utc"]
        manifest["resumed_utc"] = old.get("resumed_utc", []) + [datetime.now(timezone.utc).isoformat()]
        manifest["cases"] = list(dict.fromkeys(old["cases"] + ids))
        manifest["sets"] = {**old.get("sets", {}), **sets}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows, failures = [], []
    for i, cid in enumerate(ids, 1):
        ct = ROOT / "data/raw/ct_rate/volumes" / f"{cid}.nii.gz"
        cdir = out / cid
        t_case = time.time()
        if (cdir / "answers.json").is_file() and all((cdir / m / "result.json").is_file() for m in
                                                     ["findings"] * args.findings + ["findings_grounded"] * args.grounded):
            print(f"[{i}/{len(ids)}] {cid}: already done, skipped", flush=True)
            continue
        try:
            if not ct.is_file():
                raise FileNotFoundError(f"CT not found: {ct}")
            torch.cuda.reset_peak_memory_stats()
            ds = DemoDataset(str(ct))
            prep_s = time.time() - t_case
            if cdir.exists():           # half-finished case from an interrupted run
                shutil.rmtree(cdir)
            cdir.mkdir()
            answers = []
            for q in cfg["questions"]:
                prompt = cfg["question_template"].format(q=q["q"])
                tq = time.time()
                r = ask(ds, prompt, gen_q, keep_mask_features=False)
                answers.append({"ct_rate_label": q["ct_rate_label"], "prompt": prompt, "raw": r["text"],
                                "answer": parse_yes_no(r["text"]), "seconds": round(time.time() - tq, 2)})
            rec = {"case_id": cid, "set": sets[cid], "ct_sha256": sha256_file(ct), "ct_shape": list(ds.original_shape),
                   "config_version": cfg["config_version"], "answers": answers}
            (cdir / "answers.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            modes = ([("findings", cfg["findings_prompt"])] if args.findings else []) + \
                    ([("findings_grounded", cfg["findings_grounded_prompt"])] if args.grounded else [])
            for mode, prompt in modes:
                r = ask(ds, prompt, gen_f, keep_mask_features=True)
                save_case(r, cdir / mode, ds)
            yes = [a["ct_rate_label"] for a in answers if a["answer"] == "yes"]
            unp = sum(a["answer"] == "unparsed" for a in answers)
            rows.append({"case_id": cid, "set": sets[cid], "status": "ok", "yes": len(yes), "unparsed": unp,
                         "prep_s": round(prep_s, 1), "total_s": round(time.time() - t_case, 1),
                         "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)})
            print(f"[{i}/{len(ids)}] {cid}: {len(yes)} yes, {unp} unparsed, {rows[-1]['total_s']} s, "
                  f"peak {rows[-1]['gpu_peak_gib']} GiB | {'; '.join(yes) or 'none'}", flush=True)
        except Exception as e:  # a failed case is reported, never dropped silently
            failures.append({"case_id": cid, "error": f"{type(e).__name__}: {e}"})
            rows.append({"case_id": cid, "set": sets[cid], "status": "FAILED", "yes": "", "unparsed": "",
                         "prep_s": "", "total_s": round(time.time() - t_case, 1), "gpu_peak_gib": ""})
            (out / f"{cid}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{i}/{len(ids)}] {cid}: FAILED {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()

    fields = ["case_id", "set", "status", "yes", "unparsed", "prep_s", "total_s", "gpu_peak_gib"]
    for name, cols, data in (("batch_results.csv", fields, rows), ("failures.csv", ["case_id", "error"], failures)):
        new = not (out / name).is_file()
        with open(out / name, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerows(data)
    done_n = len(list(out.glob("*/answers.json")))
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), model_load_s=round(load_s, 1),
                    ok=done_n, failed_this_pass=len(failures))
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: {done_n} cases answered in total, {len(failures)} failed in this pass -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
