"""Phase 3 comparison (nvreason env): local NV-Reason-CT inference on HU CT volumes.

Third AI reader next to scripts/run_medgemma.py and scripts/run_hulumed.py. It is "shown only":
nothing here changes a checklist decision, a measurement or a threshold.

Different from the other two readers in one important way: NV-Reason-CT reads the NIfTI file
itself as ONE 3D block (its own processor: LPS from the affine, 2 mm, a 192^3 crop = 384 mm,
-1000..1000 HU). Our slice builder is not used. So for each case this script also saves what
the model was given (crop pictures and how much of the scan fell outside the cube).

The model is asked in its documented way (configs/nvreason_ct_v1.json):
  1. one structured chest CT report (reasoning on);
  2. one "Is ... present in this CT? Answer only Yes or No." question per checklist condition;
  3. for a Yes on a condition that has a side: one right / left / both question (our wording).
The checklist-shaped answer (vlm_parsed.json, schema checklist_v1) is ASSEMBLED here from the
yes/no answers. Yes -> present, No -> absent; any other answer fails the case, nothing is guessed.

The model loads with trust_remote_code: the repository's own Python code runs. The revision is
pinned in the config after that code was read; an unpinned revision is refused. Weights and code
are read from models/nvreason/hub only; no CT data leaves this computer.

Per case:
  model_input/crop_axial.png, crop_coronal.png   the cube the model received (lung window)
  model_input.json     source geometry, crop statistics, prompts, token counts
  report_raw.txt       complete report response (reasoning included), written before any parsing
  report.txt           the report without the reasoning block
  questions.json       every question, the raw answer, how it was read, and score_yes: the share of "Yes" between
                       "Yes" and "No" for the first answer token (0 to 1, uncalibrated; not a probability of disease)
  vlm_raw.txt          all raw answers, one per line (written before parsing)
  vlm_result.json      status, errors/warnings, runtime, tokens, GPU memory
  vlm_parsed.json      only if all yes/no answers could be read
Run-level: run_manifest.json, batch_results.csv, failures.csv.

Usage (Git Bash):
    environments/nvreason/Scripts/python.exe scripts/run_nvreason.py \
        --qc-run runs/<P1_qc_verified_...> --n 1 [--dry-run] [--no-report]
--dry-run builds and saves the model's input cube without loading the model weights.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
MODEL_CACHE = ROOT / "models" / "nvreason" / "hub"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # weights are already on disk; nothing is fetched at run time

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from lung_ct.vlm.checklist_schema import READER_CONDITIONS  # noqa: E402
from lung_ct.vlm.prompting import parse_response, prompt_sha256  # noqa: E402

CUBE_MM = 384.0


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


def snapshot_dir(mcfg: dict) -> Path:
    if not mcfg.get("revision"):
        raise RuntimeError("model.revision must be pinned (reviewed remote code) before loading")
    local = MODEL_CACHE / ("models--" + mcfg["model_id"].replace("/", "--")) / "snapshots" / mcfg["revision"]
    if not (local / "config.json").exists():               # read from disk only; setup_nvreason.bat does the download
        raise FileNotFoundError(f"{local} not found - run setup_nvreason.bat first")
    return local


def load_processor(local: Path):
    import transformers
    processor = transformers.AutoProcessor.from_pretrained(local, trust_remote_code=True)
    loader = processor.image_processor_3d
    original, cache = loader.load_image, {}

    def cached(filename, normalize_mode=None, anatomy_region=None):
        """One scan is asked ~25 questions: load and resample it once (same call, same result)."""
        key = (str(filename), str(normalize_mode), str(anatomy_region))
        if key not in cache:
            cache.clear()                                   # keep one scan in memory
            cache[key] = original(filename, normalize_mode=normalize_mode, anatomy_region=anatomy_region)
        return cache[key]

    loader.load_image = cached
    return processor


def load_model(mcfg: dict, local: Path):
    import torch
    import transformers

    if mcfg.get("quantization"):
        raise NotImplementedError("quantization is not part of nvreason_ct_v1")
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        local, trust_remote_code=True, dtype=getattr(torch, mcfg["dtype"]),
        attn_implementation=mcfg["attn_implementation"]).eval().to(mcfg["device"])
    weights = {p.name: sha256_file(p) for p in sorted(local.glob("*.safetensors"))}
    code = {p.name: sha256_file(p) for p in sorted(local.glob("*.py"))}
    info = {"local_snapshot": str(local), "weights_sha256": weights, "remote_code_sha256": code,
            "device": str(model.device), "model_dtype": str(model.dtype),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "resolved_revision": local.name,
            "gpu_memory_after_load_mb": round(torch.cuda.memory_allocated() / 2**20)}
    return model, info


def source_geometry(ct_path: Path) -> dict:
    """What the file says, and how much of it cannot fit into the model's 384 mm cube."""
    img = nib.load(str(ct_path))
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    codes = "".join(nib.aff2axcodes(img.affine))
    extent = {}
    for ax, n, z in zip(codes, img.shape[:3], zooms):
        name = {"L": "left_right", "R": "left_right", "A": "front_back", "P": "front_back",
                "S": "head_foot", "I": "head_foot"}[ax]
        extent[name] = round(n * z, 1)
    return {"shape": [int(x) for x in img.shape[:3]], "spacing_mm": [round(z, 4) for z in zooms],
            "orientation": codes, "extent_mm": extent,
            "outside_cube_mm": {k: round(max(0.0, v - CUBE_MM), 1) for k, v in extent.items()},
            "cube_mm": CUBE_MM}


def save_cube(cube, in_dir: Path) -> dict:
    """cube: tensor [1, z, y, x] in HU/1000, x -> patient left, y -> posterior, z -> superior."""
    from PIL import Image
    hu = cube[0].float().cpu().numpy() * 1000.0
    lo, hi = -1350.0, 150.0                                  # lung window W1500 / L-600

    def png(a, name):
        g = np.clip((a - lo) / (hi - lo), 0, 1)
        Image.fromarray((g * 255).astype(np.uint8)).resize((576, 576), Image.NEAREST).save(in_dir / name)

    nz, ny, nx = hu.shape
    png(hu[nz // 2], "crop_axial.png")                       # rows anterior -> posterior, columns patient right -> left
    png(hu[::-1, ny // 2, :], "crop_coronal.png")            # rows superior -> inferior, columns patient right -> left
    return {"cube_shape_zyx": [int(nz), int(ny), int(nx)], "cube_spacing_mm": 2.0,
            "hu_min": float(hu.min()), "hu_max": float(hu.max()), "hu_mean": round(float(hu.mean()), 1),
            "fraction_at_minus_1000": round(float((hu <= -999.5).mean()), 4),
            "fraction_lung_range_-950_-500": round(float(((hu > -950) & (hu < -500)).mean()), 4),
            "pictures": {"crop_axial.png": "middle axial slice of the cube: anterior at the top, patient right on the image left",
                         "crop_coronal.png": "middle coronal slice of the cube: head at the top, patient right on the image left"},
            "window_hu": [lo, hi]}


_EMPTY_ANSWER = re.compile(r"<answer>\s*</answer>")     # the report ends with an empty answer tag; the raw file keeps it


def split_thinking(text: str) -> tuple[str, str]:
    if "</think>" in text:
        think, _, rest = text.partition("</think>")
        return think.replace("<think>", "").strip(), _EMPTY_ANSWER.sub("", rest).strip()
    return "", _EMPTY_ANSWER.sub("", text).strip()


def read_yes_no(text: str) -> str | None:
    w = re.sub(r"[^a-z]", " ", split_thinking(text)[1].lower()).split()
    return {"yes": "present", "no": "absent"}.get(w[0]) if w else None


def read_side(text: str) -> str | None:
    w = re.sub(r"[^a-z]", " ", split_thinking(text)[1].lower()).split()
    if not w:
        return None
    return {"right": "right", "left": "left", "both": "both", "bilateral": "both"}.get(w[0])


class Asker:
    def __init__(self, processor, model, ct_path: Path, region: str, do_sample: bool):
        self.p, self.m, self.ct, self.region, self.do_sample = processor, model, str(ct_path), region, do_sample
        self.calls = []
        ids = [processor.tokenizer.encode(w, add_special_tokens=False) for w in ("Yes", "No")]
        if any(len(i) != 1 for i in ids):
            raise RuntimeError(f"'Yes' / 'No' are not single tokens for this tokenizer: {ids}")
        self.yes_id, self.no_id = ids[0][0], ids[1][0]

    def ask(self, prompt_text: str, thinking: bool, max_new_tokens: int, score_yes_no: bool = False) -> dict:
        import torch
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        prompt = self.p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=thinking)
        inputs = self.p(text=prompt, images3d=[self.ct], anatomy_region=self.region, return_tensors="pt").to(self.m.device)
        n_in = int(inputs.input_ids.shape[1])
        t0 = time.time()
        with torch.inference_mode():
            gen = self.m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=self.do_sample, use_cache=True,
                                  output_scores=score_yes_no, return_dict_in_generate=True)
        out = gen.sequences
        new = out[0, n_in:]
        text = self.p.batch_decode(new[None], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        eos = self.m.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        rec = {"prompt": prompt_text, "enable_thinking": thinking, "raw": text, "input_tokens": n_in,
               "image_tokens": int((inputs.input_ids == self.m.config.image_token_id).sum()),
               "output_tokens": int(len(new)), "seconds": round(time.time() - t0, 2),
               "truncated": bool(len(new) >= max_new_tokens and int(new[-1]) not in eos)}
        if score_yes_no:      # how the model split its first answer token between "Yes" and "No"; uncalibrated, not a probability of disease
            pr = torch.softmax(gen.scores[0][0].float(), 0)
            y, n = float(pr[self.yes_id]), float(pr[self.no_id])
            rec["score"] = {"p_yes_of_yes_no": round(y / (y + n), 4) if y + n > 0 else None, "p_yes": round(y, 4), "p_no": round(n, 4),
                            "p_other_first_token": round(max(0.0, 1 - y - n), 4)}
        self.calls.append(rec)
        return rec


def infer_case(cid: str, ct_path: Path, cfg: dict, case_dir: Path, processor, model, with_report: bool) -> dict:
    rec = {"case_id": cid, "ct_file": str(ct_path.relative_to(ROOT)), "ct_sha256": sha256_file(ct_path),
           "status": "ok", "issues": []}
    in_dir = case_dir / "model_input"
    in_dir.mkdir(parents=True, exist_ok=True)
    rcfg = cfg["representation"]
    geom = source_geometry(ct_path)
    cube = processor.image_processor_3d.load_image(str(ct_path), normalize_mode=rcfg["normalize_mode"],
                                                   anatomy_region=rcfg["anatomy_region"])
    model_input = {"case_id": cid, "representation": rcfg["name"], "source": geom, "cube": save_cube(cube, in_dir),
                   "prompt_version": cfg["prompt"]["version"],
                   "prompts_sha256": prompt_sha256({"report": cfg["report"], "questions": cfg["questions"]})}
    cut = {k: v for k, v in geom["outside_cube_mm"].items() if v > 0}
    if cut:
        rec["issues"].append("scan larger than the model's 384 mm cube: " + ", ".join(f"{k} {v} mm not seen" for k, v in cut.items()))
    rec["outside_cube_mm"] = geom["outside_cube_mm"]
    (case_dir / "model_input.json").write_text(json.dumps(model_input, indent=2))
    if model is None:
        rec["status"] = "dry_run"
        return rec

    import torch
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    asker = Asker(processor, model, ct_path, rcfg["anatomy_region"], cfg["generation"]["do_sample"])

    if with_report:
        r = asker.ask(cfg["report"]["prompt"], cfg["report"]["enable_thinking"], cfg["report"]["max_new_tokens"])
        (case_dir / "report_raw.txt").write_text(r["raw"], encoding="utf-8")      # before any parsing
        (case_dir / "report.txt").write_text(split_thinking(r["raw"])[1], encoding="utf-8")
        rec.update({"report_tokens": r["output_tokens"], "report_seconds": r["seconds"], "report_truncated": r["truncated"]})
        if r["truncated"]:
            rec["issues"].append("report reached the token limit")

    q = cfg["questions"]
    rows, raw_lines, errors, warnings = [], [], [], []
    conditions = {}
    for c in q["conditions"]:
        a = asker.ask(q["presence_template"].format(phrase=c["phrase"]), q["enable_thinking"], q["max_new_tokens"], score_yes_no=True)
        status = read_yes_no(a["raw"])
        row = {"key": c["key"], "model_label": c["model_label"], "question": a["prompt"], "raw": a["raw"],
               "status": status, "score_yes": a["score"]["p_yes_of_yes_no"], "score": a["score"], "seconds": a["seconds"]}
        raw_lines.append(f"{c['key']}\tpresence\t{a['raw']!r}")
        if status is None:
            errors.append(f"{c['key']}: answer {a['raw']!r} is neither Yes nor No")
        entry = {"status": status, "side": "not_applicable", "score_yes": a["score"]["p_yes_of_yes_no"]}
        if status == "present" and c["side"]:
            s = asker.ask(q["side_template"].format(phrase=c["phrase"]), q["enable_thinking"], q["max_new_tokens"])
            side = read_side(s["raw"])
            row.update({"side_question": s["prompt"], "side_raw": s["raw"], "side": side})
            raw_lines.append(f"{c['key']}\tside\t{s['raw']!r}")
            if side is None:
                warnings.append(f"{c['key']}: side answer {s['raw']!r} not read; side left out")
                entry.pop("side")
            else:
                entry["side"] = side
        conditions[c["key"]] = entry
        rows.append(row)
    (case_dir / "vlm_raw.txt").write_text("\n".join(raw_lines), encoding="utf-8")   # before assembling
    (case_dir / "questions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    assembled = {"assessability": "assessable", "conditions": conditions,
                 "limitations": ["Assembled by lung_ct from separate yes/no answers; the model did not write this object.",
                                 "The model was not asked whether the scan can be assessed and was not offered 'cannot tell'."]}
    parsed = {"status": "invalid", "errors": errors, "warnings": warnings}
    if not errors:
        p = parse_response(json.dumps(assembled), False, cfg["prompt"]["schema"])   # same validator as the other readers
        parsed = {"status": p["status"], "errors": p["errors"], "warnings": warnings + p["warnings"]}
    n_present = sum(v["status"] == "present" for v in conditions.values())
    rec.update({"parse_status": parsed["status"], "extraction_method": "assembled_from_yes_no",
                "errors": parsed["errors"], "warnings": parsed["warnings"], "truncated": False,
                "n_present": n_present, "n_calls": len(asker.calls),
                "input_tokens": asker.calls[0]["input_tokens"], "image_tokens": asker.calls[0]["image_tokens"],
                "output_tokens": sum(c["output_tokens"] for c in asker.calls),
                "generate_seconds": round(time.time() - t0, 1),
                "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                "peak_gpu_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20)})
    (case_dir / "vlm_result.json").write_text(json.dumps(rec, indent=2))
    if parsed["status"] == "valid":
        (case_dir / "vlm_parsed.json").write_text(json.dumps(assembled, indent=2))
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
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "nvreason_ct_v1.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-report", action="store_true", help="yes/no questions only")
    args = ap.parse_args()

    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    keys = [c["key"] for c in cfg["questions"]["conditions"]]
    if keys != list(READER_CONDITIONS):
        raise SystemExit("config conditions differ from lung_ct.vlm.checklist_schema.READER_CONDITIONS")
    qc = json.loads((ROOT / args.qc_run / "run_manifest.json").read_text())
    qc_cases = {c["case_id"]: c for c in qc["cases"]}
    order = args.cases if args.cases else list(qc_cases)
    eligible, skipped = [], []
    for cid in order:
        c = qc_cases.get(cid)
        if c is None or c["status"] == "failed":
            skipped.append({"case_id": cid, "reason": "absent or failed in Phase 1 QC"})
        elif not c["ct"]["hu_valid"]:
            skipped.append({"case_id": cid, "reason": f"non-HU: {c['ct']['hu_reason']}"})
        elif "laterality" in c and not c["laterality"].get("verified"):
            skipped.append({"case_id": cid, "reason": "left / right not verified from anatomy; the model takes it from the file"})
        else:
            eligible.append(cid)
    if args.n:
        eligible = eligible[:args.n]

    import importlib.metadata as md
    tag = "P3_nvreason_dry" if args.dry_run else "P3_nvreason"
    run_id = f"{tag}_{len(eligible)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    pkgs = ("torch", "transformers", "accelerate", "monai", "dynamic-network-architectures", "huggingface_hub",
            "tokenizers", "pillow", "nibabel", "numpy", "scipy")
    manifest = {"run_id": run_id,
                "provenance": {"started_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
                               "qc_run": str(args.qc_run), "config_file": str(cfg_path.relative_to(ROOT)),
                               "config_sha256": sha256_file(cfg_path), "python": platform.python_version(),
                               "platform": platform.platform(),
                               "packages": {p: md.version(p) for p in pkgs}, "dry_run": args.dry_run,
                               "with_report": not args.no_report,
                               "laterality_verified_in_qc": any("laterality" in c for c in qc["cases"])},
                "config": cfg, "skipped": skipped, "cases": []}
    print(f"{run_id}: {len(eligible)} cases, {len(skipped)} skipped")

    model = None
    try:
        local = snapshot_dir(cfg["model"])
        processor = load_processor(local)
        if not args.dry_run:
            import torch
            manifest["provenance"]["gpu"] = torch.cuda.get_device_name(0)
            manifest["provenance"]["cuda_runtime"] = torch.version.cuda
            model, manifest["model_info"] = load_model(cfg["model"], local)
    except Exception as e:  # noqa: BLE001
        manifest["model_load_error"] = f"{type(e).__name__}: {e}"
        manifest["provenance"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"MODEL LOAD FAILED: {manifest['model_load_error']}")
        traceback.print_exc()
        return 2

    for cid in eligible:
        print(f"[{cid}] ...", end=" ", flush=True)
        case_dir = run_dir / cid
        case_dir.mkdir()
        try:
            rec = infer_case(cid, ROOT / qc_cases[cid]["ct"]["file"], cfg, case_dir, processor, model, not args.no_report)
        except Exception as e:  # noqa: BLE001
            rec = {"case_id": cid, "status": "failed", "issues": [f"{type(e).__name__}: {e}"],
                   "traceback": traceback.format_exc()}
            (case_dir / "vlm_result.json").write_text(json.dumps(rec, indent=2))
        manifest["cases"].append(rec)
        print(rec["status"], {k: rec[k] for k in ("parse_status", "n_present", "n_calls", "input_tokens", "generate_seconds",
                                                   "peak_gpu_memory_mb") if k in rec}, "; ".join(rec["issues"]))

    manifest["provenance"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    cols = ["case_id", "status", "parse_status", "n_present", "n_calls", "input_tokens", "image_tokens", "output_tokens",
            "report_tokens", "report_truncated", "generate_seconds", "peak_gpu_memory_mb"]
    with open(run_dir / "batch_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest["cases"])
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
