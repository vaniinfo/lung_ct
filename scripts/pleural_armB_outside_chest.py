"""Where does Arm B put its "effusion"? Descriptive only (core env). Changes no mask and no score.

Arm B reads the whole CT; Arm A reads only a box around the lungs (+50 mm). For each scan this script reports how much
Arm B "effusion" lies on slices outside the chest. "Chest slices" = slices where Arm A's own lung label
(lung_pleural.nii.gz, TotalSegmentator) has voxels, widened by 50 mm above and below (the margin Arm A uses).
This is a description of the output, NOT a filter: the primary comparison keeps every Arm B voxel.
The "inside chest slices" Dice is an exploratory what-if, labelled as such, and is not a result of the fixed comparison.
Usage: python scripts/pleural_armB_outside_chest.py --manifest <armB manifest> --armA-manifest <armA manifest> --out <dir>
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
MARGIN_MM = 50.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--armA-manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))["cases"]
    arm_a = {c["case_id"]: c["prediction"] for c in json.loads((ROOT / args.armA_manifest).read_text(encoding="utf-8"))["cases"]}
    rows = []
    for c in man:
        cid = c["case_id"]
        r = {"case_id": cid, "set": c["set"]}
        try:
            ct = nib.as_closest_canonical(nib.load(ROOT / c["ct"]))
            zmm = float(ct.header.get_zooms()[2])
            vml = float(np.prod(ct.header.get_zooms()[:3])) / 1000.0
            eff = np.asarray(nib.as_closest_canonical(nib.load(ROOT / c["prediction"])).dataobj) > 0
            lung = np.asarray(nib.as_closest_canonical(nib.load((ROOT / arm_a[cid]).parent / "lung_pleural.nii.gz")).dataobj) > 0
            r.update(scan_length_mm=ct.shape[2] * zmm, b_ml=float(eff.sum()) * vml)
            zs = np.flatnonzero(lung.any((0, 1)))
            if zs.size == 0:
                r["note"] = "Arm A lung label empty"
            else:
                k = int(round(MARGIN_MM / zmm))
                lo, hi = max(0, zs[0] - k), min(ct.shape[2], zs[-1] + k + 1)
                inside = np.zeros_like(eff)
                inside[:, :, lo:hi] = eff[:, :, lo:hi]
                r.update(chest_length_mm=(hi - lo) * zmm, b_ml_inside_chest_slices=float(inside.sum()) * vml,
                         b_ml_outside_chest_slices=float(eff.sum() - inside.sum()) * vml)
                if c.get("references", {}).get("first"):
                    ref = np.asarray(nib.as_closest_canonical(nib.load(ROOT / c["references"]["first"])).dataobj) > 0
                    if ref.any():
                        r["reference_ml"] = float(ref.sum()) * vml
                        r["reference_ml_outside_chest_slices"] = float(ref.sum() - ref[:, :, lo:hi].sum()) * vml
                        r["b_dice_all_voxels"] = 2 * float((ref & eff).sum()) / float(ref.sum() + eff.sum())
                        r["b_dice_exploratory_inside_chest_slices_only"] = 2 * float((ref & inside).sum()) / float(ref.sum() + inside.sum())
        except Exception as e:  # noqa: BLE001
            r["note"] = f"{type(e).__name__}: {e}"
        rows.append(r)
    out = ROOT / args.out
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "armB_outside_chest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    done = [r for r in rows if "b_ml_outside_chest_slices" in r]
    big = sorted((r for r in done if r["b_ml_outside_chest_slices"] >= 5.0), key=lambda r: -r["b_ml_outside_chest_slices"])
    pos = [r for r in done if "b_dice_all_voxels" in r]
    summ = {"note": __doc__.split("Usage")[0].strip(), "scans": len(rows), "measured": len(done), "not_measured": [r["case_id"] for r in rows if r not in done],
            "scans_with_any_b_voxels_outside_chest_slices": sum(r["b_ml_outside_chest_slices"] > 0 for r in done),
            "scans_with_5_mL_or_more_outside_chest_slices": [{"case_id": r["case_id"], "set": r["set"], "scan_length_mm": r["scan_length_mm"],
                                                              "b_ml": r["b_ml"], "b_ml_outside_chest_slices": r["b_ml_outside_chest_slices"]} for r in big],
            "by_set_outside_5_mL_or_more": {s: sum(r["set"] == s for r in big) for s in sorted({r["set"] for r in rows})},
            "scan_length_mm_median_by_set": {s: float(np.median([r["scan_length_mm"] for r in done if r["set"] == s])) for s in sorted({r["set"] for r in done})},
            "reference_positive": {"n": len(pos), "reference_ml_outside_chest_slices_max": max(r["reference_ml_outside_chest_slices"] for r in pos),
                                   "b_dice_median_all_voxels": float(np.median([r["b_dice_all_voxels"] for r in pos])),
                                   "b_dice_median_exploratory_inside_chest_slices_only": float(np.median([r["b_dice_exploratory_inside_chest_slices_only"] for r in pos])),
                                   "b_dice_mean_all_voxels": float(np.mean([r["b_dice_all_voxels"] for r in pos])),
                                   "b_dice_mean_exploratory_inside_chest_slices_only": float(np.mean([r["b_dice_exploratory_inside_chest_slices_only"] for r in pos]))}}
    (out / "armB_outside_chest_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "note"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
