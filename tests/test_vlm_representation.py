"""MedGemma CT representation: notebook-faithful sampling/windowing, explicit orientation."""
import io
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from lung_ct.vlm.ct_representation import (
    build_representation, notebook_slice_indices, reorient_for_display, window_rgb)

CFG = json.loads((Path(__file__).resolve().parent.parent / "configs" /
                  "medgemma_ct_v1.json").read_text(encoding="utf-8"))
REP = CFG["representation"]
SHAPE = (40, 50, 120)
SPACING = (0.8, 0.8, 2.0)


def test_config_matches_notebook_windows():
    assert REP["rgb_windows_hu"] == [[-1024, 1024], [-135, 215], [0, 80]]
    assert REP["max_slices"] == 85 and REP["image_format"] == "jpeg"


def test_slice_indices_follow_notebook_formula():
    idx = notebook_slice_indices(301, 85)
    assert len(idx) == 85
    assert idx == [int(round(i / 85 * 300)) for i in range(1, 86)]
    assert idx[0] == 4 and idx[-1] == 300        # index 0 is skipped, last slice included
    assert all(b > a for a, b in zip(idx, idx[1:]))
    assert notebook_slice_indices(60, 85) == list(range(60))
    with pytest.raises(ValueError):
        notebook_slice_indices(0)


def test_window_rgb_matches_notebook_norm():
    hu = np.array([[-2000, -1024, -135, 0, 80, 215, 1024, 3000]], dtype=np.float32)
    rgb = window_rgb(hu)
    assert rgb.dtype == np.uint8 and rgb.shape == (1, 8, 3)
    for c, (lo, hi) in enumerate([(-1024, 1024), (-135, 215), (0, 80)]):
        expect = np.round((np.clip(hu, lo, hi).astype(np.float32) - lo) / (hi - lo) * 255.0)
        assert np.array_equal(rgb[..., c], expect.astype(np.uint8))
    assert list(rgb[0, 0]) == [0, 0, 0] and list(rgb[0, -1]) == [255, 255, 255]


def _make_img(axcodes: str, value_fn=None):
    """HU-like volume (air -1000, body 40) with a bright marker in the patient's
    Right-Anterior-Superior region, placed via world coordinates."""
    ras_axes = {"R": (0, 1), "L": (0, -1), "A": (1, 1), "P": (1, -1), "S": (2, 1), "I": (2, -1)}
    affine = np.eye(4)
    for col, code in enumerate(axcodes):
        world_axis, sign = ras_axes[code]
        affine[:3, col] = 0
        affine[world_axis, col] = sign * SPACING[col]
    affine[:3, 3] = -affine[:3, :3] @ (np.array(SHAPE) / 2)
    extent = np.abs(affine[:3, :3]) @ np.array(SHAPE, dtype=float)
    world = 0.3 * extent
    vox = np.round(np.linalg.inv(affine) @ np.append(world, 1)).astype(int)[:3]
    data = np.full(SHAPE, -1000, dtype=np.int16)
    data[5:-5, 5:-5, 5:-5] = 40
    data[tuple(slice(v - 2, v + 3) for v in vox)] = 1000
    return nib.Nifti1Image(data, affine)


@pytest.mark.parametrize("axcodes", ["LAS", "RAS", "LPS", "RPI"])
def test_display_layout_is_radiological_for_any_orientation(axcodes):
    disp = reorient_for_display(_make_img(axcodes))
    arr = np.asanyarray(disp.dataobj)
    r, c, k = [x.mean() / n for x, n in zip(np.nonzero(arr == 1000), arr.shape)]
    assert r < 0.5, "anterior must be at the top (low row index)"
    assert c < 0.5, "patient right must be on image left (low column index)"
    assert k < 0.5, "superior must come first in slice order"


def test_build_representation_end_to_end():
    rep = build_representation(_make_img("LAS"), REP)
    assert rep.info["n_images"] == 85 == len(rep.images) == len(rep.slices)
    assert rep.info["source_axcodes"] == "LAS" and rep.info["display_axcodes"] == "PLI"
    z = [s["world_ras_center_mm"][2] for s in rep.slices]
    assert all(b < a for a, b in zip(z, z[1:])), "slices must run superior -> inferior"
    assert [s["label"] for s in rep.slices[:2]] == ["SLICE 1", "SLICE 2"]
    with Image.open(io.BytesIO(rep.images[0])) as im:
        # LAS input: rows come from the A/P axis (50), columns from L/R (40); PIL size=(w, h)
        assert im.format == "JPEG" and im.mode == "RGB" and im.size == (SHAPE[0], SHAPE[1])
    # source voxel provenance maps back to integer slice positions of the LAS array
    ks = [s["source_voxel_center"][2] for s in rep.slices]
    assert all(abs(k - round(k)) < 1e-6 for k in ks)


def test_refuses_display_scaled_volume():
    img = nib.Nifti1Image(np.random.default_rng(0).integers(0, 256, SHAPE).astype(np.uint8),
                          np.diag([*SPACING, 1.0]))
    with pytest.raises(ValueError, match="not HU"):
        build_representation(img, REP)
