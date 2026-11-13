"""NIfTI file loading with geometry and intensity validation."""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np


@dataclass
class VolumeMetadata:
    """Metadata extracted from a NIfTI volume."""
    file_path: str
    sha256: str
    shape: tuple
    spacing_mm: tuple  # voxel dimensions in mm
    origin: tuple
    direction: np.ndarray  # 3x3 direction matrix
    affine: np.ndarray  # 4x4 affine
    dtype: str
    voxel_volume_mm3: float
    intensity_min: float
    intensity_max: float
    hu_valid: bool  # whether intensity range is plausible for HU
    hu_reason: str
    orientation_code: str  # e.g. 'RAS', 'LPS'
    extra: dict = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


HU_AIR_RANGE = (-1100.0, -900.0)   # air / outside-body floor expected in HU CT
HU_STANDARD_MAX = 3071.0           # 12-bit HU ceiling; above this = metal/contrast/extended scale


def assess_hu(data: np.ndarray) -> tuple[bool, str]:
    """Decide whether intensities are plausibly Hounsfield units.

    Never rescales anything; only classifies. Rules (first match wins):
      - all values in [0, 255]            -> display-scaled, not HU
      - minimum >= 0                      -> unsigned / missing rescale intercept, not HU
      - minimum < -2000                   -> outside expected HU bounds (unhandled padding)
      - minimum above air range (> -900)  -> no air-like values, cannot confirm HU
      - otherwise HU; values > 3071 are reported but do not invalidate.
    """
    imin, imax = float(data.min()), float(data.max())
    if imin >= 0 and imax <= 255:
        return False, f"range [{imin:.0f}, {imax:.0f}] looks like display-scaled (0-255), not HU"
    if imin >= 0:
        return False, f"range [{imin:.0f}, {imax:.0f}] has no negative values (missing HU offset?)"
    if imin < -2000:
        return False, f"range [{imin:.0f}, {imax:.0f}] minimum below -2000, outside expected HU bounds"
    if imin > HU_AIR_RANGE[1]:
        return False, f"range [{imin:.0f}, {imax:.0f}] has no air-like values (min > -900)"
    reason = f"plausible HU range [{imin:.0f}, {imax:.0f}]"
    if imax > HU_STANDARD_MAX:
        n = int(np.count_nonzero(data > HU_STANDARD_MAX))
        reason += f"; {n} voxels > {HU_STANDARD_MAX:.0f} (metal/dense contrast or extended scale)"
    return True, reason


def load_nifti(path: Path) -> tuple[nib.Nifti1Image, VolumeMetadata]:
    """Load a NIfTI file and extract validated metadata.

    Returns the nibabel image and a VolumeMetadata dataclass.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    img = nib.load(str(path))
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    # Voxel spacing from affine (handles anisotropic)
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    # Direction cosines (normalized columns of rotation part)
    direction = affine[:3, :3] / spacing[np.newaxis, :]

    # Origin
    origin = tuple(affine[:3, 3].tolist())

    # Voxel volume
    voxel_vol = float(np.abs(np.linalg.det(affine[:3, :3])))

    # Orientation code (e.g. RAS, LPS)
    ornt = nib.orientations.aff2axcodes(affine)
    orientation_code = "".join(ornt)

    # Intensity validation for HU
    imin, imax = float(data.min()), float(data.max())
    hu_valid, hu_reason = assess_hu(data)

    file_hash = sha256_file(path)

    meta = VolumeMetadata(
        file_path=str(path),
        sha256=file_hash,
        shape=tuple(data.shape),
        spacing_mm=tuple(spacing.tolist()),
        origin=origin,
        direction=direction,
        affine=affine,
        dtype=str(data.dtype),
        voxel_volume_mm3=voxel_vol,
        intensity_min=imin,
        intensity_max=imax,
        hu_valid=hu_valid,
        hu_reason=hu_reason,
        orientation_code=orientation_code,
    )

    return img, meta


def check_geometry_match(meta_a: VolumeMetadata, meta_b: VolumeMetadata,
                         spacing_tol: float = 0.01, origin_tol: float = 0.5) -> dict:
    """Check if two volumes have matching geometry for overlay.

    Returns a dict with match status and any mismatches found.
    """
    issues = []

    if meta_a.shape != meta_b.shape:
        issues.append(f"shape mismatch: {meta_a.shape} vs {meta_b.shape}")

    spacing_diff = np.abs(np.array(meta_a.spacing_mm) - np.array(meta_b.spacing_mm))
    if np.any(spacing_diff > spacing_tol):
        issues.append(f"spacing mismatch (max diff {spacing_diff.max():.4f}mm): "
                      f"{meta_a.spacing_mm} vs {meta_b.spacing_mm}")

    origin_diff = np.abs(np.array(meta_a.origin) - np.array(meta_b.origin))
    if np.any(origin_diff > origin_tol):
        issues.append(f"origin mismatch (max diff {origin_diff.max():.2f}mm): "
                      f"{meta_a.origin} vs {meta_b.origin}")

    dir_diff = np.abs(meta_a.direction - meta_b.direction).max()
    if dir_diff > 0.01:
        issues.append(f"direction mismatch (max diff {dir_diff:.4f})")

    return {
        "match": len(issues) == 0,
        "issues": issues,
        "file_a": meta_a.file_path,
        "file_b": meta_b.file_path,
    }
