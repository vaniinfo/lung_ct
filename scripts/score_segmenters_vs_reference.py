"""Score TotalSegmentator and NV-Segment-CT against EXPERT lung masks (COVID-19 CT Lung and Infection Segmentation set; core env).

Unlike compare_nvsegment_totalseg.py (agreement between two models), this uses a reference drawn by people:
left lung (label 1) and right lung (label 2). It can judge whole lungs only: the reference has no lobes and no airways.

Per case and side, for each model: Dice with the reference lung, volume error, and how much of the reference INFECTED
lung (Infection_Mask inside the reference lung) the model's lung mask covers - diseased lung is where lung masks fail.
Predictions and references stay separate; a missing or empty mask is reported and never scored as 0.

Only HU-valid cases are scored (the ten 'coronacases'); the NV-Segment-CT run already skips the others.

Writes runs/E26_segmenters_vs_expert_lungs_<time>/ : scores.csv, summary.json, SUMMARY.md

Usage:
    .venv/Scripts/python.exe scripts/score_segmenters_vs_reference.py --nvsegment-run runs/P2_nvsegment_10cases_<ts>
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lung_ct.evaluation.lung_seg import LEFT_LOBES, RIGHT_LOBES, same_grid  # noqa: E402

DATA_DIR = ROOT / "data" / "raw" / "covid_ct"
NV = {"left": [28, 29], "right": [30, 31, 32]}
TS = {"left": LEFT_LOBES, "right": RIGHT_LOBES}
REF = {"left": 1, "right": 2}


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    s = int(a.sum()) + int(b.sum())
    return None if s == 0 else 2.0 * int((a & b).sum()) / s


def side_scores(pred: np.ndarray, ref: np.ndarray, infected: np.ndarray, vox_ml: float) -> dict:
    """Dice, volumes and coverage of the infected part of the reference lung for one side and one model."""
    inf = infected & ref
    n_inf = int(inf.sum())
    return {"dice": dice(pred, ref), "pred_ml": int(pred.sum()) * vox_ml, "ref_ml": int(ref.sum()) * vox_ml,
            "volume_error_pct": 100.0 * (int(pred.sum()) - int(ref.sum())) / int(ref.sum()) if ref.any() else None,
            "ref_infected_ml": n_inf * vox_ml, "infected_covered_pct": 100.0 * int((pred & inf).sum()) / n_inf if n_inf else None,
            "outside_ref_ml": int((pred & ~ref).sum()) * vox_ml, "missed_ref_ml": int((ref & ~pred).sum()) * vox_ml}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nvsegment-run", required=True, type=Path)
    args = ap.parse_args()
    run = args.nvsegment_run if args.nvsegment_run.is_absolute() else ROOT / args.nvsegment_run
    man = json.loads((run / "run_manifest.json").read_text())
    out = ROOT / "runs" / f"E26_segmenters_vs_expert_lungs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    rows, not_scored = [], [{"case_id": s["case_id"], "reason": s["reason"]} for s in man["skipped"]]
    for c in man["cases"]:
        cid = c["case_id"]
        if c["status"] != "ok":
            not_scored.append({"case_id": cid, "reason": "NV-Segment-CT failed: " + "; ".join(c["issues"])})
            continue
        ref_img = nib.load(str(DATA_DIR / "Lung_Mask" / f"{cid}.nii.gz"))
        inf_img = nib.load(str(DATA_DIR / "Infection_Mask" / f"{cid}.nii.gz"))
        nv_img = nib.load(str(run / cid / "nvseg_labels.nii.gz"))
        if not (same_grid(nv_img, ref_img) and same_grid(inf_img, ref_img)):
            not_scored.append({"case_id": cid, "reason": "masks are not on one grid"})
            continue
        ref, infected, nv = np.asanyarray(ref_img.dataobj), np.asanyarray(inf_img.dataobj) > 0, np.asanyarray(nv_img.dataobj)
        vox_ml = float(abs(np.linalg.det(np.asarray(ref_img.affine, float)[:3, :3]))) / 1000.0
        seg = ROOT / "data" / "derived" / cid / "segmentation" / "total"
        for side in ("left", "right"):
            ref_m = ref == REF[side]
            ts_parts = []
            for n in TS[side]:
                p = seg / f"{n}.nii.gz"
                img = nib.load(str(p)) if p.exists() else None
                ts_parts.append(None if img is None or not same_grid(img, ref_img) else np.asanyarray(img.dataobj) > 0)
            preds = {"NV-Segment-CT": np.isin(nv, NV[side]),
                     "TotalSegmentator": None if any(x is None for x in ts_parts) else np.logical_or.reduce(ts_parts)}
            for model, pred in preds.items():
                if pred is None or not pred.any():
                    not_scored.append({"case_id": cid, "reason": f"{model} {side} lung: mask missing, off-grid or empty"})
                    continue
                rows.append({"case_id": cid, "side": side, "model": model, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                                              for k, v in side_scores(pred, ref_m, infected, vox_ml).items()}})
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def col(model, key, side=None):
        return [r[key] for r in rows if r["model"] == model and r[key] is not None and (side is None or r["side"] == side)]
    models = ["TotalSegmentator", "NV-Segment-CT"]
    summ = {}
    for m in models:
        d, cov, ve = col(m, "dice"), col(m, "infected_covered_pct"), [abs(x) for x in col(m, "volume_error_pct")]
        summ[m] = {"lungs_scored": len(d), "dice_min": round(min(d), 4), "dice_median": round(float(np.median(d)), 4), "dice_mean": round(float(np.mean(d)), 4), "dice_max": round(max(d), 4),
                   "abs_volume_error_pct_median": round(float(np.median(ve)), 2), "abs_volume_error_pct_max": round(max(ve), 2),
                   "infected_lung_covered_pct_min": round(min(cov), 1), "infected_lung_covered_pct_median": round(float(np.median(cov)), 1),
                   "missed_ref_ml_total": round(sum(col(m, "missed_ref_ml")), 1), "outside_ref_ml_total": round(sum(col(m, "outside_ref_ml")), 1)}
    pair = {(r["case_id"], r["side"]): {} for r in rows}
    for r in rows:
        pair[(r["case_id"], r["side"])][r["model"]] = r
    both = [v for v in pair.values() if len(v) == 2]
    diff = [v["NV-Segment-CT"]["dice"] - v["TotalSegmentator"]["dice"] for v in both]
    rng = np.random.default_rng(20261126)
    cases = sorted({r["case_id"] for r in rows})
    by_case = {c: [v["NV-Segment-CT"]["dice"] - v["TotalSegmentator"]["dice"] for (cc, _), v in pair.items() if cc == c and len(v) == 2] for c in cases}
    boots = [float(np.mean(np.concatenate([by_case[c] for c in rng.choice(cases, len(cases))]))) for _ in range(5000)]      # patients resampled, both lungs kept together
    paired = {"lungs": len(both), "patients": len(cases), "mean_dice_difference_nv_minus_ts": round(float(np.mean(diff)), 4),
              "interval_95_patient_bootstrap": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
              "lungs_nv_higher_by_0_005": sum(x > 0.005 for x in diff), "lungs_ts_higher_by_0_005": sum(x < -0.005 for x in diff), "lungs_within_0_005": sum(abs(x) <= 0.005 for x in diff)}
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "nvsegment_run": str(args.nvsegment_run).replace("\\", "/"),
               "reference": "COVID-19 CT Lung and Infection Segmentation Dataset (Zenodo 3757476): expert left / right lung masks and infection masks; ten HU-valid cases",
               "what_this_can_judge": "whole left and right lungs only; not lobes, not airways, not any other class", "models": summ, "paired": paired, "not_scored": not_scored}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    md = [f"# TotalSegmentator and NV-Segment-CT against expert lung masks: {len(cases)} scans, {len(both)} lungs", "",
          "Reference: expert left / right lung masks of the COVID-19 CT set (HU-valid cases only). Whole lungs only: no lobes, no airways.", "",
          "| Model | Dice lowest | Dice middle | Dice mean | Dice highest | Volume error, middle | Volume error, largest | Infected lung covered, lowest | Infected lung covered, middle |", "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {m} | {s['dice_min']:.3f} | {s['dice_median']:.3f} | {s['dice_mean']:.3f} | {s['dice_max']:.3f} | {s['abs_volume_error_pct_median']} % | {s['abs_volume_error_pct_max']} % | "
           f"{s['infected_lung_covered_pct_min']} % | {s['infected_lung_covered_pct_median']} % |" for m, s in summ.items()]
    md += ["", f"Paired, same lungs: mean Dice difference (NV-Segment-CT minus TotalSegmentator) {paired['mean_dice_difference_nv_minus_ts']:+.4f}, "
           f"95 % interval {paired['interval_95_patient_bootstrap'][0]:+.4f} to {paired['interval_95_patient_bootstrap'][1]:+.4f} (patients resampled). "
           f"NV higher on {paired['lungs_nv_higher_by_0_005']} lungs, TotalSegmentator higher on {paired['lungs_ts_higher_by_0_005']}, within 0.005 on {paired['lungs_within_0_005']}.", "",
           "| Scan | Side | TS Dice | NV Dice | TS infected covered | NV infected covered | Infected lung (reference) |", "|---|---|---|---|---|---|---|"]
    for (cid, side), v in sorted(pair.items()):
        if len(v) == 2:
            t, n = v["TotalSegmentator"], v["NV-Segment-CT"]
            f = lambda x: "n/a" if x is None else f"{x:.1f} %"  # noqa: E731
            md.append(f"| {cid} | {side} | {t['dice']:.3f} | {n['dice']:.3f} | {f(t['infected_covered_pct'])} | {f(n['infected_covered_pct'])} | {t['ref_infected_ml']:.0f} mL |")
    md += ["", "Not scored: " + ("; ".join(f"{x['case_id']} ({x['reason']})" for x in not_scored) or "none")]
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
