"""Verify lung label laterality from world coordinates (never from array index)."""
import nibabel as nib
import numpy as np


def label_centroids_ras(mask_img: nib.Nifti1Image, labels=(1, 2)) -> dict:
    """World-space (RAS+, mm) centroid of each label; None if the label is empty."""
    data = np.asanyarray(mask_img.dataobj)
    out = {}
    for lab in labels:
        idx = np.argwhere(data == lab)
        if idx.size == 0:
            out[lab] = None
            continue
        out[lab] = (mask_img.affine @ np.append(idx.mean(axis=0), 1.0))[:3].tolist()
    return out


def check_lung_laterality(mask_img: nib.Nifti1Image, left_label: int = 1,
                          right_label: int = 2) -> dict:
    """Check that `left_label` lies on the patient's left of `right_label`.

    NIfTI world space is RAS+: +x points to the patient's Right, so the left lung
    must have the smaller x centroid.
    """
    c = label_centroids_ras(mask_img, (left_label, right_label))
    cl, cr = c[left_label], c[right_label]
    if cl is None or cr is None:
        missing = [lab for lab, v in c.items() if v is None]
        return {"verified": False, "reason": f"empty label(s): {missing}",
                "centroids_ras_mm": c}
    dx = cr[0] - cl[0]
    ok = dx > 0
    return {
        "verified": bool(ok),
        "reason": (f"right-label centroid is {dx:.1f} mm toward patient right"
                   if ok else
                   f"label {left_label} is NOT on patient left (dx={dx:.1f} mm)"),
        "centroids_ras_mm": c,
    }
