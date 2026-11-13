"""Display orientation must depend on world coordinates, not on-disk array order."""
import nibabel as nib
import numpy as np
import pytest

from lung_ct.qc.qc_views import DISPLAY_LABELS, display_slices, generate_qc_views

SHAPE = (40, 50, 30)
SPACING = (0.8, 0.8, 2.0)


def _make_img(axcodes: str) -> nib.Nifti1Image:
    """Build a volume with a marker in the patient's Right-Anterior-Superior corner.

    The marker is placed via world coordinates so it is the same anatomy for any axcodes.
    """
    ras_axes = {"R": (0, 1), "L": (0, -1), "A": (1, 1), "P": (1, -1), "S": (2, 1), "I": (2, -1)}
    affine = np.eye(4)
    for col, code in enumerate(axcodes):
        world_axis, sign = ras_axes[code]
        affine[:3, col] = 0
        affine[world_axis, col] = sign * SPACING[col]
    affine[:3, 3] = -affine[:3, :3] @ (np.array(SHAPE) / 2)  # centre at world origin

    # world extent per RAS axis (accounts for axis permutation, e.g. PIR)
    extent = np.abs(affine[:3, :3]) @ np.array(SHAPE, dtype=float)
    world = 0.25 * extent  # right (+x), anterior (+y), superior (+z), inside the volume

    vox = np.round(np.linalg.inv(affine) @ np.append(world, 1)).astype(int)[:3]
    data = np.zeros(SHAPE, dtype=np.int16)
    data[tuple(slice(v - 2, v + 3) for v in vox)] = 1
    return nib.Nifti1Image(data, affine), world


def _marker_rc(arr):
    rows, cols = np.nonzero(arr)
    assert rows.size, "marker not visible in slice"
    return rows.mean() / arr.shape[0], cols.mean() / arr.shape[1]


@pytest.mark.parametrize("axcodes", ["RAS", "LAS", "LPS", "LPI", "PIR"])
def test_marker_lands_on_labelled_side(axcodes):
    img, world = _make_img(axcodes)
    assert "".join(nib.aff2axcodes(img.affine)) == axcodes
    # slice through the marker in canonical RAS index space
    ras = nib.as_closest_canonical(img)
    idx = tuple(np.round(np.linalg.inv(ras.affine) @ np.append(world, 1)).astype(int)[:3])
    views = display_slices(img, index_ras=idx)

    r, c = _marker_rc(views["axial"][0])
    assert DISPLAY_LABELS["axial"]["left"] == "R" and c < 0.5   # patient right -> screen left
    assert DISPLAY_LABELS["axial"]["top"] == "A" and r > 0.5    # origin=lower: high row = top

    r, c = _marker_rc(views["coronal"][0])
    assert c < 0.5 and r > 0.5                                  # R on left, S on top

    r, c = _marker_rc(views["sagittal"][0])
    assert DISPLAY_LABELS["sagittal"]["left"] == "A" and c < 0.5
    assert r > 0.5


def test_aspect_is_vertical_over_horizontal_spacing():
    img, _ = _make_img("LAS")
    views = display_slices(img)
    assert views["axial"][1] == pytest.approx(0.8 / 0.8)
    assert views["coronal"][1] == pytest.approx(2.0 / 0.8)
    assert views["sagittal"][1] == pytest.approx(2.0 / 0.8)


def test_overlay_refuses_mismatched_geometry(tmp_path):
    img, _ = _make_img("LAS")
    same = nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine.copy())
    generate_qc_views(img, "ok", tmp_path, mask_img=same)

    aff = img.affine.copy()
    aff[0, 3] += 5.0
    bad = nib.Nifti1Image(np.asanyarray(img.dataobj), aff)
    with pytest.raises(ValueError, match="geometry"):
        generate_qc_views(img, "bad", tmp_path, mask_img=bad)
