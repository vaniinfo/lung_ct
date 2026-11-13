"""Tests for physical volume calculations - protects scientific correctness."""
import numpy as np
import nibabel as nib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lung_ct.metrics.volumes import (
    mask_volume_ml,
    voxel_volume_from_affine,
    dice_coefficient,
)


def make_nifti(data, spacing=(1.0, 1.0, 1.0)):
    """Create a NIfTI image with given spacing."""
    affine = np.diag([*spacing, 1.0])
    return nib.Nifti1Image(data.astype(np.float32), affine)


def test_isotropic_cube_volume():
    """A 10x10x10 isotropic 1mm cube should be 1.0 mL."""
    data = np.ones((10, 10, 10))
    vol = mask_volume_ml(data, voxel_volume_mm3=1.0)
    assert abs(vol - 1.0) < 1e-6, f"Expected 1.0 mL, got {vol}"


def test_anisotropic_cuboid_volume():
    """An anisotropic synthetic cuboid has the expected physical volume in mL.

    10x10x10 voxels with spacing 0.5x0.5x2.0mm = voxel volume 0.5mm^3
    Total volume = 1000 * 0.5 / 1000 = 0.5 mL
    """
    data = np.ones((10, 10, 10))
    spacing = (0.5, 0.5, 2.0)
    voxel_vol = 0.5 * 0.5 * 2.0  # 0.5 mm^3
    vol = mask_volume_ml(data, voxel_volume_mm3=voxel_vol)
    assert abs(vol - 0.5) < 1e-6, f"Expected 0.5 mL, got {vol}"


def test_voxel_volume_from_affine():
    """Voxel volume from affine matches spacing product for orthogonal image."""
    affine = np.diag([0.7, 0.7, 3.0, 1.0])
    vol = voxel_volume_from_affine(affine)
    expected = 0.7 * 0.7 * 3.0
    assert abs(vol - expected) < 1e-6, f"Expected {expected}, got {vol}"


def test_voxel_volume_negative_det():
    """Voxel volume uses absolute determinant (handles flipped axes)."""
    affine = np.diag([-0.7, 0.7, 3.0, 1.0])
    vol = voxel_volume_from_affine(affine)
    expected = 0.7 * 0.7 * 3.0
    assert abs(vol - expected) < 1e-6


def test_dice_identical():
    """Identical masks give Dice = 1.0."""
    a = np.array([1, 1, 0, 0, 1])
    assert dice_coefficient(a, a) == 1.0


def test_dice_disjoint():
    """Disjoint masks give Dice = 0.0."""
    a = np.array([1, 1, 0, 0, 0])
    b = np.array([0, 0, 1, 1, 0])
    assert dice_coefficient(a, b) == 0.0


def test_dice_both_empty():
    """Both empty masks give Dice = 1.0 (documented convention)."""
    a = np.zeros(10)
    b = np.zeros(10)
    assert dice_coefficient(a, b) == 1.0


def test_dice_one_empty():
    """One empty mask gives Dice = 0.0."""
    a = np.array([1, 1, 0])
    b = np.zeros(3)
    assert dice_coefficient(a, b) == 0.0


def test_dice_partial_overlap():
    """Partial overlap gives expected Dice."""
    a = np.array([1, 1, 1, 0, 0])
    b = np.array([0, 1, 1, 1, 0])
    # intersection=2, |a|=3, |b|=3, Dice = 2*2/(3+3) = 4/6
    expected = 4.0 / 6.0
    assert abs(dice_coefficient(a, b) - expected) < 1e-6


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
