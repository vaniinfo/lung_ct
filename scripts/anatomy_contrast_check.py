"""Anatomical left/right check and contrast detection after segmentation (core env).

For each case in a QC run, using TotalSegmentator masks (heart, liver, aorta, lung lobes):
  laterality  RAS world centroids: the heart must lie on the patient's LEFT of the lung midline
              (smaller x) and the liver on the RIGHT (larger x). Both reversed -> the image is
              mirrored: the case FAILS (never "fixed" silently). Missing masks -> unverified.
  contrast    median HU inside the aorta (mask eroded by 1 voxel to avoid the wall):
              < 100 non-contrast, 100-150 indeterminate, > 150 contrast. Thresholds are
              conventional ranges (unenhanced blood ~30-60 HU), not validated here.

Writes runs/P1_qc_verified_<source run>_<time>/run_manifest.json: a copy of the QC records with
"laterality" and "contrast_measured" filled in; cases failing laterality get status "failed", so
downstream stages skip them.

Usage: python scripts/anatomy_contrast_check.py --qc-run runs/P1_qc_ctrate_10cases_<time>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
LOBES = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
         "lung_middle_lobe_right", "lung_lower_lobe_right"]
CONTRAST_HU = {"non_contrast_below": 100, "contrast_above": 150}


def centroid(mask: np.ndarray, affine: np.ndarray) -> np.ndarray | None:
    """World (RAS, mm) centroid; None if empty."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    return (affine @ np.append(idx.mean(axis=0), 1.0))[:3]


def centroid_x(mask: np.ndarray, affine: np.ndarray) -> float | None:
    c = centroid(mask, affine)
    return None if c is None else float(c[0])


def check_case(cid: str, ct_file: str) -> dict:
    ct = nib.load(str(ROOT / ct_file.replace("\\", "/")))
    seg = ROOT / "data" / "derived" / cid / "segmentation" / "total"

    def load(name):
        p = seg / f"{name}.nii.gz"
        if not p.exists():
            return None
        img = nib.load(str(p))
        if img.shape != ct.shape or not np.allclose(img.affine, ct.affine, atol=1e-3):
            raise ValueError(f"{name} mask not on the CT grid")
        return np.asanyarray(img.dataobj) > 0

    out = {}
    lungs = [load(n) for n in LOBES]
    heart, liver, aorta = load("heart"), load("liver"), load("aorta")
    if any(m is None for m in lungs) or heart is None or liver is None:
        out["laterality"] = {"verified": False, "reason": "heart, liver or lung lobe masks missing"}
    else:
        left = lungs[0] | lungs[1]
        right = lungs[2] | lungs[3] | lungs[4]
        xl, xr = centroid_x(left, ct.affine), centroid_x(right, ct.affine)
        xh, xv = centroid_x(heart, ct.affine), centroid_x(liver, ct.affine)
        mid = (xl + xr) / 2 if None not in (xl, xr) else None
        rec = {"x_mm": {"left_lung": xl, "right_lung": xr, "heart": xh, "liver": xv, "lung_midline": mid}}
        if None in (xl, xr, xh, xv):
            rec.update(verified=False, reason="an organ mask is empty")
        else:
            lungs_ok = xr > xl                      # RAS: +x = patient right
            heart_left, liver_right = xh < mid, xv > mid
            # front/back and top/bottom (RAS: +y anterior, +z superior); a wrong header can rotate
            # the image 180 degrees (both x and y flipped) or flip one axis only
            cl, ch, cv = centroid(left | right, ct.affine), centroid(heart, ct.affine), centroid(liver, ct.affine)
            trach = load("trachea")
            ct_ = centroid(trach, ct.affine) if trach is not None else None
            heart_anterior = bool(ch[1] > cl[1])
            liver_below_heart = bool(cv[2] < ch[2])
            trachea_above_heart = bool(ct_[2] > ch[2]) if ct_ is not None else None
            rec["checks"] = {"heart_left_of_midline": bool(heart_left), "liver_right_of_midline": bool(liver_right),
                             "heart_anterior_to_lungs": heart_anterior, "liver_below_heart": liver_below_heart,
                             "trachea_above_heart": trachea_above_heart}
            ap_si_ok = heart_anterior and liver_below_heart and trachea_above_heart is not False
            if lungs_ok and heart_left and liver_right and ap_si_ok:
                rec.update(verified=True, reason=f"heart {mid - xh:.0f} mm left of lung midline, "
                                                 f"liver {xv - mid:.0f} mm right; heart anterior, "
                                                 f"trachea above, liver below")
            elif lungs_ok and heart_left and liver_right:
                rec.update(verified=False, reason=f"front/back or top/bottom orientation inconsistent: {rec['checks']}")
            elif (not heart_left) and (not liver_right):
                rec.update(verified=False, mirrored=True,
                           reason="heart on the right and liver on the left: image appears MIRRORED")
            else:
                rec.update(verified=False, reason="inconsistent organ positions (check the scan; "
                                                  "situs anomaly or segmentation error)")
        out["laterality"] = rec

    if aorta is None or not aorta.any():
        out["contrast_measured"] = {"status": "unknown", "reason": "no aorta mask"}
    else:
        core = ndimage.binary_erosion(aorta, iterations=1)
        core = core if core.sum() > 100 else aorta
        hu = np.asanyarray(ct.dataobj)[core].astype(np.float32)
        med = float(np.median(hu))
        status = ("non_contrast" if med < CONTRAST_HU["non_contrast_below"]
                  else "contrast" if med > CONTRAST_HU["contrast_above"] else "indeterminate")
        out["contrast_measured"] = {"status": status, "aorta_median_hu": round(med, 1),
                                    "aorta_p10_p90_hu": [round(float(np.percentile(hu, 10)), 1),
                                                         round(float(np.percentile(hu, 90)), 1)],
                                    "voxels": int(core.sum()), "thresholds_hu": CONTRAST_HU}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qc-run", required=True)
    args = ap.parse_args()
    src = json.loads((ROOT / args.qc_run / "run_manifest.json").read_text(encoding="utf-8"))
    run = ROOT / "runs" / f"P1_qc_verified_{Path(args.qc_run).name}_{datetime.now().strftime('%H%M%S')}"
    run.mkdir(parents=True)
    cases, bad = [], 0
    for c in src["cases"]:
        c = dict(c)
        if c.get("status") == "failed":
            cases.append(c)
            continue
        try:
            res = check_case(c["case_id"], c["ct"]["file"])
            c.update(res)
            if not res["laterality"].get("verified"):
                c["status"] = "failed"
                c.setdefault("issues", []).append(f"laterality not verified: {res['laterality']['reason']}")
                bad += 1
            declared = c.get("contrast")
            meas = res["contrast_measured"]["status"]
            if declared and "non-contrast" in declared and meas == "contrast":
                c.setdefault("issues", []).append("dataset says non-contrast but aorta looks enhanced")
            print(f"[{c['case_id']}] laterality {res['laterality'].get('verified')}: {res['laterality']['reason']} | "
                  f"contrast {meas} ({res['contrast_measured'].get('aorta_median_hu')} HU)")
        except Exception as e:  # recorded, never dropped
            c["status"] = "failed"
            c.setdefault("issues", []).append(f"anatomy/contrast check failed: {type(e).__name__}: {e}")
            bad += 1
            print(f"[{c['case_id']}] FAILED {e}")
        cases.append(c)
    (run / "run_manifest.json").write_text(json.dumps(
        {"run_id": run.name, "source_qc_run": args.qc_run,
         "provenance": {**src.get("provenance", {}), "verified_utc": datetime.now(timezone.utc).isoformat()},
         "cases": cases}, indent=2), encoding="utf-8")
    print(f"done: {len(cases) - bad} verified, {bad} failed -> {run}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
