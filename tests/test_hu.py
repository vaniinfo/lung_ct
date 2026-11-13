import numpy as np
import pytest

from lung_ct.ingest.nifti_loader import assess_hu


def _vol(lo, hi, extra=None):
    d = np.linspace(lo, hi, 1000).reshape(10, 10, 10)
    if extra is not None:
        d.flat[0] = extra
    return d


@pytest.mark.parametrize("lo,hi,valid,needle", [
    (-1024, 2000, True, "plausible"),
    (0, 255, False, "display-scaled"),
    (0, 4095, False, "no negative"),
    (-3024, 2000, False, "below -2000"),
    (-500, 1500, False, "no air-like"),
])
def test_classification(lo, hi, valid, needle):
    ok, reason = assess_hu(_vol(lo, hi))
    assert ok is valid
    assert needle in reason


def test_metal_does_not_invalidate_but_is_reported():
    ok, reason = assess_hu(_vol(-1023, 1500, extra=9567))
    assert ok
    assert "1 voxels > 3071" in reason
