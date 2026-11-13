"""Review-package viewer orientation: laterality on screen must come from world coordinates."""
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("brp", ROOT / "scripts" / "build_review_package.py")
brp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brp)


@pytest.mark.parametrize("axcodes", ["LAS", "RAS", "LPS", "RPI"])
def test_axial_viewer_is_radiological(axcodes):
    ras_axes = {"R": (0, 1), "L": (0, -1), "A": (1, 1), "P": (1, -1), "S": (2, 1), "I": (2, -1)}
    shape = (40, 50, 30)
    aff = np.eye(4)
    for col, c in enumerate(axcodes):
        a, s = ras_axes[c]
        aff[:3, col] = 0
        aff[a, col] = s
    aff[:3, 3] = -aff[:3, :3] @ (np.array(shape) / 2)
    world = np.array([12.0, 15.0, 0.0])                 # patient right, anterior
    vox = np.round(np.linalg.inv(aff) @ np.append(world, 1)).astype(int)[:3]
    d = np.zeros(shape, np.int16)
    d[tuple(slice(v - 2, v + 3) for v in vox)] = 1
    canon = nib.as_closest_canonical(nib.Nifti1Image(d, aff))
    data = np.asanyarray(canon.dataobj)
    k = int(np.argwhere(data)[:, 2].mean().round())
    disp = brp.axial_display(data[:, :, k])
    rows, cols = np.nonzero(disp)
    assert rows.mean() / disp.shape[0] < 0.5, "anterior must be at the top"
    assert cols.mean() / disp.shape[1] < 0.5, "patient right must be on screen left"


def test_scoring_columns_have_no_prefilled_judgements():
    for col in ("assessability", "opacity_presence", "laterality"):
        assert col in brp.V1_COLS
    for col in ("agreement", "error_type", "comments"):
        assert col in brp.V2_COLS
    for col in ("proximal_landmark_correct", "distal_landmark_correct"):
        assert col in brp.AIR_COLS
