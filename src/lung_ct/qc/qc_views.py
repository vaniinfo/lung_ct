"""Generate QC views: axial, coronal, sagittal with orientation labels and overlays."""
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


# Standard CT display windows
CT_WINDOWS = {
    "lung": {"center": -600, "width": 1500},      # [-1350, -150]
    "mediastinum": {"center": 40, "width": 400},   # [-160, 240]
    "bone": {"center": 400, "width": 1800},        # [-500, 1300]
    "soft_tissue": {"center": 50, "width": 350},   # [-125, 225]
    # Not an HU window: shows display-scaled 8-bit volumes as stored (QC only).
    "raw8bit_nonHU": {"center": 127.5, "width": 255},
}

# Screen labels for radiological display. These are only valid for slices produced by
# `display_slices`, which reorients to canonical RAS and applies fixed flips:
#   axial/coronal: patient Right on screen left; sagittal: Anterior on screen left.
DISPLAY_LABELS = {
    "axial": {"left": "R", "right": "L", "top": "A", "bottom": "P"},
    "coronal": {"left": "R", "right": "L", "top": "S", "bottom": "I"},
    "sagittal": {"left": "A", "right": "P", "top": "S", "bottom": "I"},
}


def apply_window(data: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply CT display window (center/width) and return normalized 0-1 array."""
    low = center - width / 2
    high = center + width / 2
    windowed = np.clip(data, low, high)
    return (windowed - low) / (high - low)


def display_slices(img: nib.Nifti1Image, index_ras: Optional[tuple] = None) -> dict:
    """Extract mid (or given) axial/coronal/sagittal slices in radiological display order.

    The volume is reoriented to closest canonical RAS (axis permutation/flip only, no
    interpolation), so screen orientation does not depend on the on-disk orientation.

    Returns {view: (array_2d, aspect)} where array_2d is indexed [row, col] for
    `imshow(origin="lower")` and aspect = vertical_spacing / horizontal_spacing.
    """
    ras = nib.as_closest_canonical(img)
    data = np.asanyarray(ras.dataobj)
    sp = np.sqrt((ras.affine[:3, :3] ** 2).sum(axis=0))
    i, j, k = index_ras if index_ras is not None else [n // 2 for n in data.shape]

    # RAS axes: 0 -> Right(+), 1 -> Anterior(+), 2 -> Superior(+).
    # Transpose so rows=vertical axis; flip columns where screen-left must be +axis.
    axial = data[:, :, k].T[:, ::-1]      # cols: R..L, rows (origin lower): P..A
    coronal = data[:, j, :].T[:, ::-1]    # cols: R..L, rows: I..S
    sagittal = data[i, :, :].T[:, ::-1]   # cols: A..P, rows: I..S
    return {
        "axial": (axial, sp[1] / sp[0]),
        "coronal": (coronal, sp[2] / sp[0]),
        "sagittal": (sagittal, sp[2] / sp[1]),
    }


def generate_qc_views(
    img: nib.Nifti1Image,
    case_id: str,
    output_dir: Path,
    mask_img: Optional[nib.Nifti1Image] = None,
    window_name: str = "lung",
    mask_alpha: float = 0.3,
    mask_color: tuple = (1, 0, 0),
    label_colors: Optional[dict] = None,
    index_ras: Optional[tuple] = None,
    views: Optional[tuple] = None,
) -> list[Path]:
    """Generate axial, coronal, sagittal QC views at mid-volume.

    Args:
        img: CT NIfTI image
        case_id: identifier for filenames
        output_dir: where to save PNGs
        mask_img: optional overlay mask
        window_name: CT window preset name
        mask_alpha: overlay transparency
        mask_color: RGB tuple for mask overlay (all non-zero labels)
        label_colors: optional {label_value: (rgb, name)}; colours each label separately
            and adds a legend (labels not listed are not drawn)
        index_ras: optional slice indices in canonical RAS voxel space (default: centre)
        views: optional subset of ("axial", "coronal", "sagittal")

    Returns:
        List of saved PNG paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ornt = "".join(nib.orientations.aff2axcodes(img.affine))

    if mask_img is not None:
        if mask_img.shape[:3] != img.shape[:3] or not np.allclose(
                mask_img.affine, img.affine, atol=1e-3):
            raise ValueError(
                f"{case_id}: mask geometry does not match CT (shape {mask_img.shape} vs "
                f"{img.shape}); refusing to overlay")

    window = CT_WINDOWS[window_name]
    ct_views = display_slices(img, index_ras)
    mask_views = display_slices(mask_img, index_ras) if mask_img is not None else {}
    if views is not None:
        ct_views = {k: v for k, v in ct_views.items() if k in views}

    saved = []
    for view_name, (slc, aspect) in ct_views.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(apply_window(slc.astype(np.float32), window["center"], window["width"]),
                  cmap="gray", origin="lower", aspect=aspect, vmin=0, vmax=1)

        if view_name in mask_views:
            mslc = mask_views[view_name][0]
            overlay = np.zeros((*mslc.shape, 4))
            if label_colors:
                from matplotlib.patches import Patch
                handles = []
                for lab, (rgb, name) in label_colors.items():
                    overlay[mslc == lab] = (*rgb, mask_alpha)
                    handles.append(Patch(color=rgb, label=name))
                ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.6)
            else:
                overlay[mslc > 0] = (*mask_color, mask_alpha)
            ax.imshow(overlay, origin="lower", aspect=aspect)

        vl = DISPLAY_LABELS[view_name]
        ax.text(0.02, 0.5, vl["left"], transform=ax.transAxes,
                color="yellow", fontsize=14, va="center", fontweight="bold")
        ax.text(0.98, 0.5, vl["right"], transform=ax.transAxes,
                color="yellow", fontsize=14, va="center", ha="right", fontweight="bold")
        ax.text(0.5, 0.98, vl["top"], transform=ax.transAxes,
                color="yellow", fontsize=14, ha="center", va="top", fontweight="bold")
        ax.text(0.5, 0.02, vl["bottom"], transform=ax.transAxes,
                color="yellow", fontsize=14, ha="center", fontweight="bold")

        ax.set_title(f"{case_id} - {view_name} [{window_name} window] "
                     f"(source {ornt}, displayed radiological)", fontsize=10, color="white")
        ax.axis("off")

        out_path = output_dir / f"{case_id}_{view_name}_{window_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        saved.append(out_path)

    return saved
