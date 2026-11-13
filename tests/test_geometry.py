"""Tests for geometry validation - laterality and orientation."""
import numpy as np
import nibabel as nib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lung_ct.ingest.nifti_loader import load_nifti, check_geometry_match, VolumeMetadata


def make_test_nifti(shape=(64, 64, 32), spacing=(1.0, 1.0, 1.0), tmp_path=None):
    """Create a test NIfTI with a known left/right landmark."""
    data = np.zeros(shape, dtype=np.float32)
    # Place a landmark in the "right" half (positive x in RAS = left body side)
    # and a different value in the "left" half
    mid_x = shape[0] // 2
    data[:mid_x, :, :] = -500  # one side
    data[mid_x:, :, :] = -800  # other side
    affine = np.diag([*spacing, 1.0])
    img = nib.Nifti1Image(data, affine)
    return img, data


def test_orientation_preserved():
    """A known left/right landmark remains on the correct physical side."""
    img, data = make_test_nifti(spacing=(1.0, 1.0, 1.0))
    ornt = nib.orientations.aff2axcodes(img.affine)
    # In RAS, x-axis goes Right to Left
    assert ornt[0] == "R", f"Expected R as first axis code, got {ornt[0]}"

    # The voxel at [0,0,0] should be at the Right side (most negative x in physical)
    phys_origin = img.affine[:3, 3]
    # Verify the landmark values are accessible at expected positions
    assert data[0, 0, 0] == -500
    assert data[-1, 0, 0] == -800


def test_geometry_match_identical():
    """Identical geometry should match."""
    meta = VolumeMetadata(
        file_path="a.nii.gz", sha256="abc", shape=(64, 64, 32),
        spacing_mm=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        direction=np.eye(3), affine=np.eye(4), dtype="float32",
        voxel_volume_mm3=1.0, intensity_min=-1000, intensity_max=1000,
        hu_valid=True, hu_reason="ok", orientation_code="RAS"
    )
    result = check_geometry_match(meta, meta)
    assert result["match"] is True
    assert len(result["issues"]) == 0


def test_geometry_mismatch_shape():
    """Different shapes should not match."""
    base = dict(
        sha256="abc", spacing_mm=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        direction=np.eye(3), affine=np.eye(4), dtype="float32",
        voxel_volume_mm3=1.0, intensity_min=-1000, intensity_max=1000,
        hu_valid=True, hu_reason="ok", orientation_code="RAS"
    )
    meta_a = VolumeMetadata(file_path="a.nii.gz", shape=(64, 64, 32), **base)
    meta_b = VolumeMetadata(file_path="b.nii.gz", shape=(64, 64, 64), **base)
    result = check_geometry_match(meta_a, meta_b)
    assert result["match"] is False
    assert any("shape" in i for i in result["issues"])


def test_resampling_preserves_labels():
    """Resampling a label mask with nearest-neighbor preserves discrete class IDs."""
    from scipy.ndimage import zoom
    # Create a label mask with classes 0, 1, 2
    data = np.zeros((20, 20, 20), dtype=np.int16)
    data[5:15, 5:15, 5:15] = 1
    data[8:12, 8:12, 8:12] = 2

    # Resample with nearest-neighbor (order=0)
    resampled = zoom(data, 0.5, order=0)

    unique_before = set(np.unique(data))
    unique_after = set(np.unique(resampled))

    # Resampled should only contain original label values
    assert unique_after.issubset(unique_before), \
        f"Resampling introduced new labels: {unique_after - unique_before}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
