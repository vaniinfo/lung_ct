"""Build the MedGemma 1.5 model-visible CT representation from a HU NIfTI volume.

Reproduces the official notebook (Google-Health/medgemma @ ea92d0f,
high_dimensional_ct_hugging_face.ipynb):
  * uniform sampling of up to 85 axial slices with the notebook's index formula
  * three HU windows stacked as RGB: (-1024, 1024), (-135, 215), (0, 80)
  * clip -> scale to 0..255 -> round -> uint8, then JPEG encoding

The notebook reads DICOM slices (pixel rows anterior->posterior, columns patient
right->left, ordered by InstanceNumber). Here the NIfTI is reoriented explicitly from its
affine (axis permutation/flip only, no interpolation) so that every slice has that same
radiological in-plane layout regardless of on-disk orientation; slice order is fixed to
superior->inferior and logged per slice with world coordinates.

Only numpy / nibabel / Pillow: usable in both the core and vlm environments.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import nibabel as nib
import numpy as np
from PIL import Image

from lung_ct.ingest.nifti_loader import assess_hu

# Array axes after reorientation: rows -> Posterior, cols -> patient Left, slices -> Inferior
DISPLAY_AXCODES = ("P", "L", "I")
NOTEBOOK_WINDOWS_HU = ((-1024, 1024), (-135, 215), (0, 80))


def notebook_slice_indices(n_slices: int, max_slices: int = 85) -> list[int]:
    """Slice indices exactly as the notebook samples them (including its rounding).

    Note the notebook iterates i = 1..max_slices, so index 0 is never selected when
    subsampling. Python's round() (banker's rounding) is used, as in the notebook.
    """
    if n_slices <= 0:
        raise ValueError("volume has no slices")
    if n_slices <= max_slices:
        return list(range(n_slices))
    return [int(round(i / max_slices * (n_slices - 1))) for i in range(1, max_slices + 1)]


def _norm(hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Notebook `norm`: clip to [lo, hi], scale to 0..255 (float32)."""
    v = np.clip(hu, lo, hi).astype(np.float32)
    v -= lo
    v /= (hi - lo)
    v *= 255.0
    return v


def window_rgb(hu_slice: np.ndarray, windows=NOTEBOOK_WINDOWS_HU) -> np.ndarray:
    """HU slice (H, W) -> uint8 (H, W, 3) with one window per channel."""
    stacked = np.stack([_norm(hu_slice, lo, hi) for lo, hi in windows], axis=-1)
    return np.round(stacked, 0).astype(np.uint8)


def encode_image(rgb: np.ndarray, fmt: str = "jpeg", quality: int = 75) -> bytes:
    with io.BytesIO() as buf:
        with Image.fromarray(rgb) as im:
            if fmt == "jpeg":
                im.save(buf, format="JPEG", quality=quality)
            elif fmt == "png":
                im.save(buf, format="PNG")
            else:
                raise ValueError(f"unsupported image format {fmt!r}")
        return buf.getvalue()


def reorient_for_display(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient (no interpolation) so array axes are (Posterior, Left, Inferior)."""
    src = nib.orientations.io_orientation(img.affine)
    tgt = nib.orientations.axcodes2ornt(DISPLAY_AXCODES)
    out = img.as_reoriented(nib.orientations.ornt_transform(src, tgt))
    got = nib.orientations.aff2axcodes(out.affine)
    if tuple(got) != DISPLAY_AXCODES:
        raise RuntimeError(f"reorientation produced {got}, expected {DISPLAY_AXCODES}")
    return out


@dataclass
class CTRepresentation:
    images: list[bytes]                 # encoded model-visible images, in prompt order
    slices: list[dict]                  # per-image provenance, same order
    info: dict = field(default_factory=dict)


def build_representation(img: nib.Nifti1Image, rep_cfg: dict) -> CTRepresentation:
    """Volume -> ordered list of encoded RGB slices plus provenance.

    Raises ValueError for volumes whose intensities are not verified HU: the windows are
    defined in HU and must never be applied to display-scaled data.
    """
    if rep_cfg.get("resample"):
        raise NotImplementedError("resampling is not part of the baseline representation")
    if rep_cfg.get("slice_order") != "superior_to_inferior":
        raise NotImplementedError(f"slice_order {rep_cfg.get('slice_order')!r}")

    disp = reorient_for_display(img)
    hu = disp.get_fdata(dtype=np.float32)
    ok, reason = assess_hu(hu)
    if not ok:
        raise ValueError(f"not HU, refusing to window: {reason}")

    sp = np.sqrt((disp.affine[:3, :3] ** 2).sum(axis=0))
    n = hu.shape[2]
    idx = notebook_slice_indices(n, rep_cfg["max_slices"])
    windows = [tuple(w) for w in rep_cfg["rgb_windows_hu"]]
    inv_src = np.linalg.inv(img.affine)
    ci, cj = (hu.shape[0] - 1) / 2, (hu.shape[1] - 1) / 2

    images, slices = [], []
    for num, k in enumerate(idx, 1):
        rgb = window_rgb(hu[:, :, k], windows)
        data = encode_image(rgb, rep_cfg["image_format"], rep_cfg.get("jpeg_quality", 75))
        world = disp.affine @ np.array([ci, cj, k, 1.0])        # slice centre, RAS mm
        src_vox = inv_src @ world
        images.append(data)
        slices.append({
            "sample_number": num,
            "label": rep_cfg["slice_label_template"].format(n=num),
            "display_slice_index": int(k),
            "source_voxel_center": [round(float(v), 2) for v in src_vox[:3]],
            "world_ras_center_mm": [round(float(v), 2) for v in world[:3]],
            "image_shape": list(rgb.shape),
            "image_sha256": hashlib.sha256(data).hexdigest(),
        })

    info = {
        "representation": rep_cfg["name"],
        "source_shape": list(img.shape),
        "source_axcodes": "".join(nib.orientations.aff2axcodes(img.affine)),
        "display_axcodes": "".join(DISPLAY_AXCODES),
        "display_spacing_mm": [round(float(s), 4) for s in sp],
        "in_plane_pixel_aspect": round(float(sp[0] / sp[1]), 4),
        "n_source_slices": int(n),
        "n_images": len(images),
        "superior_most_z_mm": slices[0]["world_ras_center_mm"][2],
        "inferior_most_z_mm": slices[-1]["world_ras_center_mm"][2],
        "hu_check": reason,
        "windows_hu": [list(w) for w in windows],
        "image_format": rep_cfg["image_format"],
    }
    return CTRepresentation(images=images, slices=slices, info=info)
