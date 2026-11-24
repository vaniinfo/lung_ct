"""Paired tables for the pleural-effusion model comparison (core env). Reads the two metrics CSVs; changes nothing.

Rules fixed before any Arm B result existed:
  - reference = PleThora first-reviewer file; all 77 effusion scans stay in, empty predictions count as Dice 0
  - "improved" = Dice(B) - Dice(A) >= +0.10, "worsened" = <= -0.10, otherwise "similar"  (descriptive label only)
  - volume error = predicted mL - reference mL (signed) and its absolute value
  - scans without an expert outline are described by mask state only; they are not negatives
  - CT-RATE: report label (text-derived, "not mentioned" is not "absent") and clinician grade (4 scans) are shown
    in their own tables and never mixed with the PleThora expert-mask evaluation
  - mask states use the unchanged provisional limits (description, not assessment)
Usage: python scripts/pleural_compare_report.py --a <metrics_A.csv> --b <metrics_B.csv> --manifest <armB manifest> --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STATES = ("empty", "under_5_mL", "5_to_20_mL", "20_mL_or_more")
DELTA = 0.10


def f(x):
    return float(x) if x not in ("", None) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="runs/E14_plethora_pleural_effusion_20261123_112031/failure_groups.csv")
    args = ap.parse_args()
    A = {r["case_id"]: r for r in csv.DictReader(open(ROOT / args.a, encoding="utf-8"))}
    B = {r["case_id"]: r for r in csv.DictReader(open(ROOT / args.b, encoding="utf-8"))}
    man = {c["case_id"]: c for c in json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))["cases"]}
    grp = {r["patient_id"]: r["group"] for r in csv.DictReader(open(ROOT / args.groups, encoding="utf-8"))}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ids = [c for c in man if c in A and c in B]
    failed = [{"case_id": c, "arm_a": A.get(c, {}).get("status", "missing"), "arm_b": B.get(c, {}).get("status", "missing"), "issue_b": B.get(c, {}).get("issue", "")}
              for c in man if c not in A or c not in B or A[c]["status"] != "ok" or B[c]["status"] != "ok"]
    ok = [c for c in ids if A[c]["status"] == "ok" and B[c]["status"] == "ok"]
    pos = [c for c in ok if man[c]["set"] == "plethora_effusion"]
    paired = []
    for c in pos:
        a, b = A[c], B[c]
        da, db = f(a["first_dice"]), f(b["first_dice"])
        ref = f(a["first_reference_ml"])
        row = {"case_id": c, "e14_group": grp.get(c, ""), "reference_ml": ref, "a_ml": f(a["predicted_ml"]), "b_ml": f(b["predicted_ml"]), "a_dice": da, "b_dice": db,
               "dice_b_minus_a": db - da, "label": "improved" if db - da >= DELTA else "worsened" if db - da <= -DELTA else "similar",
               "a_share_outlined": f(a["first_share_of_reference_outlined"]), "b_share_outlined": f(b["first_share_of_reference_outlined"]),
               "a_volume_error_ml": f(a["predicted_ml"]) - ref, "b_volume_error_ml": f(b["predicted_ml"]) - ref,
               "a_state": a["mask_state_provisional_limits"], "b_state": b["mask_state_provisional_limits"],
               "a_share_in_tumour": f(a.get("share_of_prediction_inside_tumour_outline")), "b_share_in_tumour": f(b.get("share_of_prediction_inside_tumour_outline"))}
        for rv in ("second", "third"):
            if b.get(f"{rv}_dice", "") != "":
                row[f"a_dice_{rv}_reviewer"], row[f"b_dice_{rv}_reviewer"] = f(a[f"{rv}_dice"]), f(b[f"{rv}_dice"])
        paired.append(row)
    cols = list(dict.fromkeys(k for r in paired for k in r))
    with open(out / "paired_reference_positive.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(paired, key=lambda r: r["dice_b_minus_a"]))

    def stats(rows, arm):
        d = np.array([r[f"{arm}_dice"] for r in rows])
        e = np.array([r[f"{arm}_volume_error_ml"] for r in rows])
        return {"n": len(rows), "empty_predictions": sum(r[f"{arm}_ml"] == 0 for r in rows), "dice_median": float(np.median(d)), "dice_mean": float(d.mean()),
                "dice_at_least_0.7": int((d >= 0.7).sum()), "abs_volume_error_ml_median": float(np.median(np.abs(e))), "abs_volume_error_ml_mean": float(np.abs(e).mean()),
                "signed_volume_error_ml_median": float(np.median(e)), "mask_state": {s: sum(r[f"{arm}_state"] == s for r in rows) for s in STATES}}

    def table(cases, key=lambda c: A[c]["mask_state_provisional_limits"], key_b=lambda c: B[c]["mask_state_provisional_limits"]):
        t = collections.Counter((key(c), key_b(c)) for c in cases)
        return {a: {b: t[(a, b)] for b in STATES} for a in STATES}

    summ = {"rules": __doc__.split("Usage")[0].strip(), "scans_in_manifest": len(man), "scans_with_both_arms_ok": len(ok), "failures": failed,
            "plethora_expert_mask_evaluation": {
                "all": {"arm_a": stats(paired, "a"), "arm_b": stats(paired, "b"), "label_counts": dict(collections.Counter(r["label"] for r in paired)),
                        "mask_state_a_rows_b_columns": table(pos)},
                "by_e14_group": {g: {"n": len(rs), "arm_a_dice_median": float(np.median([r["a_dice"] for r in rs])), "arm_b_dice_median": float(np.median([r["b_dice"] for r in rs])),
                                     "arm_a_abs_volume_error_median": float(np.median([abs(r["a_volume_error_ml"]) for r in rs])),
                                     "arm_b_abs_volume_error_median": float(np.median([abs(r["b_volume_error_ml"]) for r in rs])),
                                     "labels": dict(collections.Counter(r["label"] for r in rs))}
                                 for g in sorted({r["e14_group"] for r in paired}) for rs in [[r for r in paired if r["e14_group"] == g]]},
                "improved": [r["case_id"] for r in paired if r["label"] == "improved"], "worsened": [r["case_id"] for r in paired if r["label"] == "worsened"],
                "sensitivity_without_LUNG1-143": {"arm_a": stats([r for r in paired if r["case_id"] != "LUNG1-143"], "a"), "arm_b": stats([r for r in paired if r["case_id"] != "LUNG1-143"], "b")}},
            "plethora_coded_no_effusion_not_confirmed_negative": {}, "ctrate_development": {}}
    neg = [c for c in ok if man[c]["set"] == "plethora_coded_no_effusion"]
    summ["plethora_coded_no_effusion_not_confirmed_negative"] = {
        "n": len(neg), "mask_state_a_rows_b_columns": table(neg),
        "arm_a_20_mL_or_more": sorted(c for c in neg if A[c]["mask_state_provisional_limits"] == "20_mL_or_more"),
        "arm_b_20_mL_or_more": sorted(c for c in neg if B[c]["mask_state_provisional_limits"] == "20_mL_or_more"),
        "arm_b_20_mL_or_more_detail": [{"case_id": c, "b_ml": f(B[c]["predicted_ml"]), "a_ml": f(A[c]["predicted_ml"]), "b_share_in_tumour": f(B[c].get("share_of_prediction_inside_tumour_outline"))}
                                       for c in sorted(neg) if B[c]["mask_state_provisional_limits"] == "20_mL_or_more"]}
    ctr = [c for c in ok if man[c]["set"] == "ctrate_development"]

    def by(cases, arm, field):
        t = collections.Counter((str(man[c][field]), arm[c]["mask_state_provisional_limits"]) for c in cases)
        return {k: {s: t[(k, s)] for s in STATES} for k in sorted({str(man[c][field]) for c in cases})}
    graded = [c for c in ctr if man[c].get("clinician_grade")]
    summ["ctrate_development"] = {
        "n": len(ctr), "note": "no expert outline; nothing here is a segmentation score",
        "agreement_between_arms_mask_state_a_rows_b_columns": table(ctr),
        "report_label_comparison": {"note": "label from the report text; 'False' = not mentioned, not confirmed absent", "arm_a": by(ctr, A, "report_label_pleural_effusion"), "arm_b": by(ctr, B, "report_label_pleural_effusion")},
        "clinician_grade_comparison": {"note": "four development cases graded on earlier pages (v3_2 / v3_3)",
                                       "cases": [{"case_id": c, "clinician_grade": man[c]["clinician_grade"], "a_ml": f(A[c]["predicted_ml"]), "a_state": A[c]["mask_state_provisional_limits"],
                                                  "b_ml": f(B[c]["predicted_ml"]), "b_state": B[c]["mask_state_provisional_limits"]} for c in graded]}}
    (out / "comparison_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "rules"}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
