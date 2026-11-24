"""DRAFT sampling table for the hiatal-hernia reader feasibility study (core env). Reads stored scores; runs no model.

Frame      the local CT-RATE patients with stored model scores, one reconstruction per patient, MINUS the four patients whose
           pages were already graded with the tool's output visible (they cannot be read blind again by that reader)
Unit       patient
Score used only to spread the sample over the score range: the mean of the patient's percentile ranks of the hiatal-hernia
           score in COLIPRI, CT-CLIP zero-shot, VocabFine and ClassFine (models without a score for the patient are left out)
Strata (fixed before any reader label exists; they are sampling strata, not diagnostic cut-offs)
  C  report mentions hiatal hernia                     all taken            probability 1
  H  12 highest composite scores among the rest        all taken            probability 1
  M  next 20                                           8 at random          probability 8 / 20
  L  all others (the low-score patients)               8 at random          probability 8 / n_L
The sample is enriched for high scores. Any estimate for the frame must weight each patient by 1 / probability.
Readers do not see the stratum, any model score, the report or the report label.
Usage: python scripts/hh_reader_study_sample.py --analysis runs/E21_ctclip_pilot_20261124 --out runs/E22_hh_reader_feasibility_DRAFT --seed 20261124
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PREVIOUSLY_GRADED = ("valid_301", "valid_541", "valid_832", "valid_988")
N_H, N_M, TAKE_M, TAKE_L = 12, 20, 8, 8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args()
    src = ROOT / args.analysis / "hiatal_hernia_per_patient.csv"
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    col = [float(r["COLIPRI_score"]) if r["COLIPRI_score"] != "" else None for r in rows]
    have = np.array([v for v in col if v is not None])
    for r, v in zip(rows, col):
        pcs = [float(r[f"{k}_percentile"]) for k in ("zero-shot", "vocabfine", "classfine")]
        if v is not None:
            pcs.append(100.0 * float((have <= v).mean()))
        r["composite_percentile"] = round(float(np.mean(pcs)), 1)
        r["models_in_composite"] = len(pcs)
    frame = [r for r in rows if r["patient"] not in PREVIOUSLY_GRADED]
    C = [r for r in frame if r["report_label_hiatal_hernia_mentioned"] == "1"]
    rest = sorted((r for r in frame if r not in C), key=lambda r: (-r["composite_percentile"], r["patient"]))
    H, M, L = rest[:N_H], rest[N_H:N_H + N_M], rest[N_H + N_M:]
    rng = random.Random(args.seed)
    picked_m = set(r["patient"] for r in rng.sample(M, TAKE_M))
    picked_l = set(r["patient"] for r in rng.sample(L, TAKE_L))
    table = []
    for name, grp, prob, picked in (("C", C, 1.0, None), ("H", H, 1.0, None), ("M", M, TAKE_M / len(M), picked_m), ("L", L, TAKE_L / len(L), picked_l)):
        for r in grp:
            sel = picked is None or r["patient"] in picked
            table.append({"patient": r["patient"], "case_id": r["case_id"], "stratum": name, "sampling_probability": round(prob, 4), "selected": int(sel),
                          "weight_if_selected": round(1 / prob, 3) if sel else "", "composite_percentile": r["composite_percentile"], "models_in_composite": r["models_in_composite"]})
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sampling_table_ANALYST_ONLY.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    sel = [t for t in table if t["selected"]]
    order = sel[:]
    random.Random(args.seed + 1).shuffle(order)                     # reading order carries no information about the stratum
    with open(out / "reading_list_FOR_READERS.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reading_order", "case_id"])
        for i, t in enumerate(order, 1):
            w.writerow([i, t["case_id"]])
    summ = {"status": "DRAFT - not issued; no reader has seen it", "created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
            "source": str(src.relative_to(ROOT)).replace("\\", "/"), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "frame_patients": len(frame), "excluded_previously_graded_open_pages": list(PREVIOUSLY_GRADED),
            "strata": {n: {"patients": len(g), "selected": sum(1 for t in sel if t["stratum"] == n), "probability": round(p, 4)}
                       for n, g, p in (("C", C, 1.0), ("H", H, 1.0), ("M", M, TAKE_M / len(M)), ("L", L, TAKE_L / len(L)))},
            "selected_total": len(sel), "not_selected_local_patients": len(table) - len(sel),
            "composite_percentile_range_selected": [min(t["composite_percentile"] for t in sel), max(t["composite_percentile"] for t in sel)],
            "files": {"sampling_table_ANALYST_ONLY.csv": "strata, probabilities, weights: never shown to readers", "reading_list_FOR_READERS.csv": "shuffled case ids only"}}
    (out / "sampling_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
