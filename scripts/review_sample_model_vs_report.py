"""Sample for a two-reader review: where NV-Reason-CT and the report disagree, who is right? (core env; runs no model)

Frame      the TR02 calibration scans answered by NV-Reason-CT (set A), conditions given with --conditions
Cells per condition (from the model answer and the CT-RATE report label; fixed before any reader sees a scan)
  model_only     model present, report silent        --n-model-only at random   (the question asked)
  report_only    report mentions it, model absent    --n-report-only at random  (the other direction)
  agree_present  both present                         --n-agree at random        (control)
  agree_absent   both absent                          --n-agree at random        (control)
Every cell is sampled at random with the seed; the probability of each item is recorded (cell size / cell count), so estimates for the
frame can be weighted. Items of one scan are put on one page. Readers do not see the cell, the model answer, the report or the label:
the page order is shuffled and every page carries all sampled conditions of that scan as plain questions.

Usage: .venv/Scripts/python.exe scripts/review_sample_model_vs_report.py --nvreason-run runs/TR02_nvreason_calib400_<time> --out runs/RS01_review_sample_<date>
Writes <out>/{sample.json (with cells: for the analyst only), reader_list.csv (scan + questions only), README.md}
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from prompt_test_yesno import load_answers  # noqa: E402

CELLS = ("model_only", "report_only", "agree_present", "agree_absent")


def cell_of(model: str, labelled: bool) -> str | None:
    if model not in ("present", "absent"):
        return None
    return {("present", False): "model_only", ("absent", True): "report_only", ("present", True): "agree_present", ("absent", False): "agree_absent"}[(model, labelled)]


def draw(frame: dict, n: dict, seed: int) -> list[dict]:
    """frame: (condition, cell) -> sorted scans. Returns sampled items with their selection probability."""
    rng = random.Random(seed)
    out = []
    for (cond, cell), scans in sorted(frame.items()):
        k = min(n[cell], len(scans))
        for s in sorted(rng.sample(scans, k)):
            out.append({"scan": s, "condition": cond, "cell": cell, "cell_size": len(scans), "probability": round(k / len(scans), 4) if scans else None})
    return out


def draw_scans(frame: dict, need: dict, seed: int) -> list[str]:
    """Scan-level sample: every page asks ALL conditions. Needs (condition, cell) -> minimum number of chosen scans in that cell are met in a
    seeded random order; a scan added for one need counts for every cell it falls in. Inclusion is not a simple probability: estimates are per condition, descriptive."""
    rng = random.Random(seed)
    chosen: list[str] = []
    keys = sorted(need)
    rng.shuffle(keys)
    for key in keys:
        pool = [s for s in frame.get(key, []) if s not in chosen]
        rng.shuffle(pool)
        while sum(s in chosen for s in frame.get(key, [])) < min(need[key], len(frame.get(key, []))) and pool:
            chosen.append(pool.pop())
    return sorted(chosen)


def all_conditions(args, gt: dict, cfg: dict, ans: dict) -> int:
    conds = cfg["conditions"]                                   # every condition is asked, also those without a CT-RATE label (ground-glass)
    frame = {}
    for c in conds:
        if not c["ct_rate_label"]:
            continue
        for s, a in ans["A"][c["key"]].items():
            cl = cell_of(a, c["ct_rate_label"] in gt[s]["labels"])
            if cl:
                frame.setdefault((c["key"], cl), []).append(s)
    for key in frame:
        frame[key].sort()
    need = {}
    for c in conds:
        if c["ct_rate_label"]:
            k = c["key"]
            need.update({(k, "model_only"): args.n_focus if k in args.conditions else args.n_model_only, (k, "report_only"): args.n_report_only,
                         (k, "agree_present"): args.n_agree})
    scans = draw_scans(frame, need, args.seed)
    order = list(scans)
    random.Random(args.seed + 1).shuffle(order)
    got = {f"{k}|{cl}": sum(s in scans for s in v) for (k, cl), v in sorted(frame.items())}
    cells = {s: {c["key"]: next((cl for (k, cl), v in frame.items() if k == c["key"] and s in v), "no label") for c in conds} for s in scans}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "design": "scan-level; every page asks every condition", "nvreason_run": args.nvreason_run,
        "cases": args.cases, "seed": args.seed, "needs": {f"{k}|{c}": v for (k, c), v in sorted(need.items())}, "focus_conditions": args.conditions,
        "frame_counts": {f"{k}|{c}": len(v) for (k, c), v in sorted(frame.items())}, "sampled_counts": got,
        "pages": [{"page": i + 1, "scan": s} for i, s in enumerate(order)], "cells_by_scan": cells,
        "note": "Analyst file: contains the cells (model answer x report label). Do not give it to the readers. Inclusion probabilities are not simple: "
                "results are per condition and descriptive."}, indent=1), encoding="utf-8")
    with open(out / "reader_list.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page", "scan"])
        for i, s in enumerate(order):
            w.writerow([i + 1, s])
    mo = {c["key"]: got.get(f"{c['key']}|model_only", 0) for c in conds if c["ct_rate_label"]}
    (out / "README.md").write_text(
        f"# Review sample: NV-Reason-CT against the report, every condition\n\n{len(scans)} scans; each page asks all {len(conds)} conditions. Seed {args.seed}.\n"
        f"'Model present, report silent' items per condition: {mo}.\nReaders see the scan and the questions only; `sample.json` must not be shared with them.\n",
        encoding="utf-8")
    print(json.dumps({"scans": len(scans), "model_only_per_condition": mo, "sampled_counts": got}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nvreason-run", required=True)
    ap.add_argument("--cases", default="D:/datasets/ct_rate/calib400/cases.json")
    ap.add_argument("--prompt-sets", default="configs/nvreason_prompt_sets_v1.json")
    ap.add_argument("--conditions", nargs="+", default=["bronchiectasis", "hiatal_hernia", "medical_material"])
    ap.add_argument("--n-model-only", type=int, default=8)
    ap.add_argument("--n-report-only", type=int, default=2)
    ap.add_argument("--n-agree", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-conditions", action="store_true",
                    help="scan-level sample: every page asks every condition; needs per labelled condition: --n-model-only (--n-focus for --conditions), "
                         "--n-report-only, --n-agree both present")
    ap.add_argument("--n-focus", type=int, default=8, help="--all-conditions: 'model present, report silent' scans needed for each condition in --conditions")
    args = ap.parse_args()
    gt = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    cfg = json.loads((ROOT / args.prompt_sets).read_text(encoding="utf-8"))
    ans, _, _ = load_answers(ROOT / args.nvreason_run)
    if args.all_conditions:
        return all_conditions(args, gt, cfg, ans)
    cond = {c["key"]: c for c in cfg["conditions"] if c["key"] in args.conditions}
    frame = {}
    for k, c in cond.items():
        for s, a in ans["A"][k].items():
            cl = cell_of(a, c["ct_rate_label"] in gt[s]["labels"])
            if cl:
                frame.setdefault((k, cl), []).append(s)
    for key in frame:
        frame[key].sort()
    n = {"model_only": args.n_model_only, "report_only": args.n_report_only, "agree_present": args.n_agree, "agree_absent": args.n_agree}
    items = draw(frame, n, args.seed)
    pages = {}
    for it in items:
        pages.setdefault(it["scan"], []).append(it["condition"])
    order = sorted(pages)
    random.Random(args.seed + 1).shuffle(order)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "nvreason_run": args.nvreason_run, "cases": args.cases, "seed": args.seed,
        "per_cell": n, "frame_counts": {f"{k}|{c}": len(v) for (k, c), v in sorted(frame.items())}, "items": items,
        "pages": [{"page": i + 1, "scan": s, "conditions": sorted(pages[s])} for i, s in enumerate(order)],
        "note": "Analyst file: contains the cells. Do not give it to the readers."}, indent=1), encoding="utf-8")
    with open(out / "reader_list.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page", "scan", "questions"])
        for i, s in enumerate(order):
            w.writerow([i + 1, s, " | ".join(f"Is {cond[k]['phrase']} present?" for k in sorted(pages[s]))])
    counts = {c: sum(it["cell"] == c for it in items) for c in CELLS}
    (out / "README.md").write_text(
        f"# Review sample: NV-Reason-CT against the report\n\n{len(items)} scan-questions on {len(pages)} scans, conditions {', '.join(sorted(cond))}.\n"
        f"Cells (analyst only): {counts}. Seed {args.seed}. Readers see the scan and the questions only; `sample.json` must not be shared with them.\n", encoding="utf-8")
    print(json.dumps({"items": len(items), "pages": len(pages), "cells": counts, "frame": {f"{k}|{c}": len(v) for (k, c), v in sorted(frame.items())}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
