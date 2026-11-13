"""Physical volume calculations from segmentation masks."""
import numpy as np
import nibabel as nib


def mask_volume_ml(mask_data: np.ndarray, voxel_volume_mm3: float) -> float:
    """Calculate physical volume in mL from a binary mask.

    Volume = voxel_count * voxel_volume_mm3 / 1000
    """
    voxel_count = int(np.sum(mask_data > 0))
    volume_ml = voxel_count * voxel_volume_mm3 / 1000.0
    return volume_ml


def voxel_volume_from_affine(affine: np.ndarray) -> float:
    """Compute voxel volume in mm^3 from the affine matrix."""
    return float(np.abs(np.linalg.det(affine[:3, :3])))


def compute_lung_volumes(lung_mask_img: nib.Nifti1Image) -> dict:
    """Compute left/right lung volumes from a lung mask.

    Assumes: left lung = 1, right lung = 2 (COVID-19-CT-Seg convention).
    Returns volumes in mL and voxel counts.
    """
    data = lung_mask_img.get_fdata()
    voxel_vol = voxel_volume_from_affine(lung_mask_img.affine)

    unique_labels = np.unique(data[data > 0]).astype(int).tolist()

    results = {
        "voxel_volume_mm3": voxel_vol,
        "unique_labels": unique_labels,
        "total_lung_voxels": int(np.sum(data > 0)),
        "total_lung_volume_ml": mask_volume_ml(data, voxel_vol),
    }

    for label in unique_labels:
        label_mask = (data == label)
        count = int(np.sum(label_mask))
        vol = count * voxel_vol / 1000.0
        results[f"label_{label}_voxels"] = count
        results[f"label_{label}_volume_ml"] = vol

    return results


def infection_by_side(infection_mask_img: nib.Nifti1Image, lung_mask_img: nib.Nifti1Image,
                      left_label: int, right_label: int) -> dict:
    """Reference infection volume within each lung (expert masks only).

    The caller must pass labels whose laterality was verified from world coordinates;
    side is never inferred from array index here.
    """
    inf = np.asanyarray(infection_mask_img.dataobj) > 0
    lung = np.asanyarray(lung_mask_img.dataobj)
    vml = voxel_volume_from_affine(lung_mask_img.affine) / 1000.0
    out = {"reference_source": "expert_masks", "left_label": left_label,
           "right_label": right_label}
    for side, lab in (("left", left_label), ("right", right_label)):
        lm = lung == lab
        n_lung, n_inf = int(lm.sum()), int((inf & lm).sum())
        out[side] = {"infection_ml": n_inf * vml, "lung_ml": n_lung * vml,
                     "infection_pct_of_lung": 100.0 * n_inf / n_lung if n_lung else None}
    return out


def compute_infection_burden(
    infection_mask_img: nib.Nifti1Image,
    lung_mask_img: nib.Nifti1Image,
) -> dict:
    """Compute reference infection burden from expert masks.

    infection_burden_pct = 100 * volume(infection AND lung) / volume(lung)
    Also reports infection voxels outside lung as QC.
    """
    inf_data = infection_mask_img.get_fdata()
    lung_data = lung_mask_img.get_fdata()
    voxel_vol = voxel_volume_from_affine(lung_mask_img.affine)

    lung_mask = lung_data > 0
    inf_mask = inf_data > 0

    inf_in_lung = np.sum(inf_mask & lung_mask)
    inf_outside_lung = np.sum(inf_mask & ~lung_mask)
    total_lung = np.sum(lung_mask)

    burden_pct = 0.0
    if total_lung > 0:
        burden_pct = 100.0 * inf_in_lung / total_lung

    return {
        "reference_source": "expert_mask",
        "infection_voxels_in_lung": int(inf_in_lung),
        "infection_voxels_outside_lung": int(inf_outside_lung),
        "infection_volume_in_lung_ml": float(inf_in_lung * voxel_vol / 1000.0),
        "total_lung_voxels": int(total_lung),
        "total_lung_volume_ml": float(total_lung * voxel_vol / 1000.0),
        "infection_burden_pct": float(burden_pct),
        "note": "Calculated from supplied expert masks, not predicted by any model",
    }


def dice_coefficient(pred: np.ndarray, ref: np.ndarray) -> float:
    """Compute Dice similarity coefficient between two binary masks.

    Returns 1.0 for two empty masks (documented convention).
    Returns 0.0 if only one is empty.
    """
    pred_bool = pred > 0
    ref_bool = ref > 0

    pred_sum = np.sum(pred_bool)
    ref_sum = np.sum(ref_bool)

    if pred_sum == 0 and ref_sum == 0:
        return 1.0
    if pred_sum == 0 or ref_sum == 0:
        return 0.0

    intersection = np.sum(pred_bool & ref_bool)
    return float(2.0 * intersection / (pred_sum + ref_sum))
