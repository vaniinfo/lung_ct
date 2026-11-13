import nibabel as nib
import numpy as np
import pytest

from lung_ct.qc.laterality import check_lung_laterality


def _mask(x_flip: bool, swap: bool = False):
    """Two blobs; label 1 at patient-left in world space unless swap=True."""
    data = np.zeros((20, 10, 10), dtype=np.uint8)
    lo, hi = (slice(2, 6), slice(14, 18))
    affine = np.diag([-1.0 if x_flip else 1.0, 1.0, 1.0, 1.0])  # LAS vs RAS
    # world x = +index (RAS) or -index (LAS); patient-left = smaller world x
    left_idx, right_idx = (hi, lo) if x_flip else (lo, hi)
    if swap:
        left_idx, right_idx = right_idx, left_idx
    data[left_idx] = 1
    data[right_idx] = 2
    return nib.Nifti1Image(data, affine)


@pytest.mark.parametrize("x_flip", [False, True])
def test_correct_laterality_verified_for_either_storage_order(x_flip):
    assert check_lung_laterality(_mask(x_flip))["verified"]


@pytest.mark.parametrize("x_flip", [False, True])
def test_swapped_labels_detected(x_flip):
    res = check_lung_laterality(_mask(x_flip, swap=True))
    assert not res["verified"]
    assert "NOT on patient left" in res["reason"]


def test_empty_label_reported():
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[2:4] = 1
    res = check_lung_laterality(nib.Nifti1Image(data, np.eye(4)))
    assert not res["verified"]
    assert "empty" in res["reason"]
