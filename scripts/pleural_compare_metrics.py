"""Reference-based metrics for one arm of the pleural-effusion model comparison (core env). Reads masks; changes nothing.

Rules fixed before any Arm B result exists (docs/PLEURAL_MODEL_COMPARISON_PLAN.md):
  - Primary reference = PleThora `first_reviewer` file. Second / third reviewer files are reported in their own
    columns and never merged.
  - Empty predictions are INCLUDED in the segmentation metrics: Dice = 0 and share outlined = 0 when the reference
    has voxels and the prediction has none. Nothing is dropped for being empty.
  - Missing reference (PleThora code 0, CT-RATE): NOT a confirmed negative. Only the prediction is described
    (voxels, mL, mask state); no segmentation score, no false-positive rate.
  - Verified empty reference (an expert confirmed that there is no effusion; manifest field
    "reference_verified_empty": true): Dice is not defined when the reference is empty and is left blank; the predicted
    voxels and mL are reported as "predicted on a verified-empty reference" and counted in their own group. A scan is
    never given a Dice of 1 for an empty prediction on an empty reference. No scan in the current sets has this status.
  - Reference file present but with 0 voxels and no verification: treated as a missing reference and flagged.
  - Mask state uses the unchanged provisional limits (20 mL / 5 mL). It is a description, not an assessment.
  - Volumes are unrounded in the CSV.

Manifest (json): {"arm": "...", "cases": [{"case_id", "set", "ct", "prediction", "references": {"first": path, ...},
                  "clinician_grade": "present|absent|null"}]}
Usage: python scripts/pleural_compare_metrics.py --manifest <json> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRESENT_ML, LOW_ML = 20.0, 5.0     # unchanged provisional reporting limits


def mask_state(ml: float) -> str:
    return "empty" if ml == 0 else "under_5_mL" if ml < LOW_ML else "5_to_20_mL" if ml < PRESENT_ML else "20_mL_or_more"


def case_metrics(pred: np.ndarray, refs: dict, vml: float, verified_empty: bool = False) -> dict:
    """Metrics for one scan. refs: name -> bool array (may be empty dict). Pure function, unit-tested."""
    n = int(pred.sum())
    r = {"voxel_ml": vml, "predicted_voxels": n, "predicted_ml": n * vml, "mask_state_provisional_limits": mask_state(n * vml)}
    nonempty = {k: v for k, v in refs.items() if int(v.sum()) > 0}
    if verified_empty:
        r["reference_status"] = "verified empty reference (no segmentation score; prediction reported)"
        r["predicted_ml_on_verified_empty_reference"] = n * vml
    elif not refs:
        r["reference_status"] = "no expert outline (not a confirmed negative)"
    elif not nonempty:
        r["reference_status"] = "reference file has 0 voxels and is not verified (treated as missing)"
    else:
        r["reference_status"] = "expert outline"
    for name, ref in nonempty.items():
        m, inter = int(ref.sum()), int((ref & pred).sum())
        r.update({f"{name}_reference_voxels": m, f"{name}_reference_ml": m * vml, f"{name}_overlap_voxels": inter,
                  f"{name}_dice": 2 * inter / (n + m),                          # empty prediction on a non-empty reference -> 0
                  f"{name}_share_of_reference_outlined": inter / m, f"{name}_predicted_minus_reference_ml": (n - m) * vml})
    return r


def load_bool(path: Path, like: nib.Nifti1Image) -> np.ndarray:
    im = nib.as_closest_canonical(nib.load(path))
    if im.shape != like.shape or not np.allclose(im.affine, like.affine, atol=1e-2):
        raise ValueError(f"{path.name}: grid differs from the CT after reorientation")
    return np.asarray(im.dataobj) > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in man["cases"]:
        r = {"arm": man["arm"], "case_id": c["case_id"], "set": c["set"], "status": "ok", "issue": "", "clinician_grade": c.get("clinician_grade") or ""}
        try:
            ct = nib.as_closest_canonical(nib.load(ROOT / c["ct"]))
            vml = float(np.prod(ct.header.get_zooms()[:3])) / 1000.0
            pp = ROOT / c["prediction"]
            if not pp.exists():
                raise FileNotFoundError("prediction file missing (run failed or not run)")
            pred = load_bool(pp, ct)
            n = int(pred.sum())
            r.update(case_metrics(pred, {name: load_bool(ROOT / rp, ct) for name, rp in (c.get("references") or {}).items()}, vml,
                                  bool(c.get("reference_verified_empty"))))
            if "gtv" in c and c["gtv"] and (ROOT / c["gtv"]).exists() and n:
                r["share_of_prediction_inside_tumour_outline"] = float((pred & load_bool(ROOT / c["gtv"], ct)).sum()) / n
        except Exception as e:  # noqa: BLE001  reported, never dropped
            r.update(status="failed", issue=f"{type(e).__name__}: {e}")
        rows.append(r)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / f"metrics_{man['arm']}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    ref = [r for r in ok if r["reference_status"] == "expert outline" and r.get("first_reference_voxels")]
    noref = [r for r in ok if r["reference_status"].startswith(("no expert", "reference file has 0"))]
    vempty = [r for r in ok if r["reference_status"].startswith("verified empty")]
    d = np.array([float(r["first_dice"]) for r in ref]) if ref else np.array([])
    summ = {"arm": man["arm"], "cases": len(rows), "failed": [{"case_id": r["case_id"], "issue": r["issue"]} for r in rows if r["status"] != "ok"],
            "with_expert_outline": {"n": len(ref), "empty_predictions_included": sum(r["predicted_voxels"] == 0 for r in ref),
                                    "dice_first_reviewer_median": float(np.median(d)) if d.size else None, "dice_first_reviewer_mean": float(d.mean()) if d.size else None,
                                    "mask_state": {s: sum(r["mask_state_provisional_limits"] == s for r in ref) for s in ("empty", "under_5_mL", "5_to_20_mL", "20_mL_or_more")}},
            "without_expert_outline_not_confirmed_negative": {"n": len(noref), "mask_state": {s: sum(r["mask_state_provisional_limits"] == s for r in noref)
                                                                                                  for s in ("empty", "under_5_mL", "5_to_20_mL", "20_mL_or_more")}},
            "verified_empty_reference": {"n": len(vempty), "with_any_predicted_voxels": sum(r["predicted_voxels"] > 0 for r in vempty)},
            "rules": "empty predictions count as Dice 0; scans without an expert outline get no Dice and are not treated as negatives; limits unchanged and provisional"}
    (out / f"summary_{man['arm']}.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
