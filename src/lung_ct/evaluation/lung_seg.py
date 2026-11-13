"""Compare predicted lung lobe masks with reference left/right lung masks.

Predictions (TotalSegmentator) and references (expert masks) are kept separate:
every output key is prefixed `pred_` or `ref_`. Missing or empty predicted masks
are reported explicitly and never counted as 0 mL.
"""
from typing import Optional

import nibabel as nib
import numpy as np

from lung_ct.metrics.volumes import dice_coefficient, voxel_volume_from_affine
from lung_ct.qc.laterality import check_lung_laterality

LEFT_LOBES = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
RIGHT_LOBES = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
LOBES = LEFT_LOBES + RIGHT_LOBES


def same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image, atol: float = 1e-3) -> bool:
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=atol)


def evaluate_lung_segmentation(
    ref_lung_img: nib.Nifti1Image,
    lobe_imgs: dict[str, Optional[nib.Nifti1Image]],
    ref_left_label: int = 1,
    ref_right_label: int = 2,
) -> dict:
    """Lobe volumes, predicted left/right lung, Dice and volume error vs reference.

    lobe_imgs maps each name in LOBES to an image, or None if the file is missing.
    Left/right Dice is only reported when all lobes of that side are present and
    on the reference grid.
    """
    vox_ml = voxel_volume_from_affine(ref_lung_img.affine) / 1000.0
    ref = np.asanyarray(ref_lung_img.dataobj)
    issues = []

    lobes, masks = {}, {}
    for name in LOBES:
        img = lobe_imgs.get(name)
        if img is None:
            lobes[name] = {"status": "missing"}
            issues.append(f"{name}: file missing")
            continue
        if not same_grid(img, ref_lung_img):
            lobes[name] = {"status": "geometry_mismatch"}
            issues.append(f"{name}: not on reference grid")
            continue
        m = np.asanyarray(img.dataobj) > 0
        n = int(m.sum())
        lobes[name] = {"status": "ok" if n else "empty", "voxels": n,
                       "volume_ml": n * vox_ml}
        if not n:
            issues.append(f"{name}: empty mask")
        masks[name] = m

    out = {"pred_lobes": lobes, "voxel_volume_ml": vox_ml}
    sides = {"left": (LEFT_LOBES, ref_left_label), "right": (RIGHT_LOBES, ref_right_label)}
    side_masks = {}
    for side, (names, ref_label) in sides.items():
        ref_m = ref == ref_label
        ref_ml = int(ref_m.sum()) * vox_ml
        rec = {"ref_volume_ml": ref_ml}
        if all(lobes[n].get("status") == "ok" for n in names):
            pred_m = np.logical_or.reduce([masks[n] for n in names])
            side_masks[side] = pred_m
            pred_ml = int(pred_m.sum()) * vox_ml
            rec.update({
                "status": "ok",
                "pred_volume_ml": pred_ml,
                "volume_error_ml": pred_ml - ref_ml,
                "volume_error_pct": 100.0 * (pred_ml - ref_ml) / ref_ml if ref_ml else None,
                "dice": dice_coefficient(pred_m, ref_m),
            })
        else:
            rec["status"] = "not_computed: incomplete lobes"
        out[f"lung_{side}"] = rec

    if len(side_masks) == 2:
        both = np.zeros(ref.shape, dtype=np.uint8)
        both[side_masks["left"]] = 1
        both[side_masks["right"]] = 2
        lat = check_lung_laterality(nib.Nifti1Image(both, ref_lung_img.affine))
        out["pred_laterality"] = lat
        if not lat["verified"]:
            issues.append(f"predicted laterality: {lat['reason']}")
        pred_all, ref_all = both > 0, ref > 0
        out["lung_total"] = {
            "pred_volume_ml": int(pred_all.sum()) * vox_ml,
            "ref_volume_ml": int(ref_all.sum()) * vox_ml,
            "dice": dice_coefficient(pred_all, ref_all),
        }
    out["issues"] = issues
    return out
