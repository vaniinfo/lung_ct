import nibabel as nib
import numpy as np
import pytest

from lung_ct.evaluation.lung_seg import LOBES, evaluate_lung_segmentation

AFF = np.diag([-2.0, 2.0, 2.0, 1.0])  # LAS, 8 mm^3 voxels
SHAPE = (20, 10, 10)


def _ref():
    d = np.zeros(SHAPE, dtype=np.uint8)
    d[12:18] = 1   # LAS: high index = patient left
    d[2:8] = 2
    return nib.Nifti1Image(d, AFF)


# axial (z) ranges that tile each lung without overlap
Z = {"left": {"upper": slice(5, 10), "lower": slice(0, 5)},
     "right": {"upper": slice(6, 10), "middle": slice(3, 6), "lower": slice(0, 3)}}


def _lobes(ref):
    r = np.asanyarray(ref.dataobj)
    out = {}
    for name in LOBES:
        _, part, _, side = name.split("_")       # lung_<part>_lobe_<side>
        lab = 1 if side == "left" else 2
        m = np.zeros(SHAPE, dtype=np.uint8)
        zs = Z[side][part]
        m[:, :, zs] = (r[:, :, zs] == lab)
        out[name] = nib.Nifti1Image(m, AFF)
    return out


def test_perfect_prediction_gives_dice_one_and_zero_error():
    ref = _ref()
    res = evaluate_lung_segmentation(ref, _lobes(ref))
    assert res["issues"] == []
    for side in ("left", "right"):
        assert res[f"lung_{side}"]["dice"] == pytest.approx(1.0)
        assert res[f"lung_{side}"]["volume_error_ml"] == pytest.approx(0.0)
    assert res["lung_left"]["ref_volume_ml"] == pytest.approx(6 * 10 * 10 * 8 / 1000)
    assert res["pred_laterality"]["verified"]


def test_missing_lobe_reported_not_zero():
    ref = _ref()
    lobes = _lobes(ref)
    lobes["lung_middle_lobe_right"] = None
    res = evaluate_lung_segmentation(ref, lobes)
    assert res["pred_lobes"]["lung_middle_lobe_right"] == {"status": "missing"}
    assert res["lung_right"]["status"].startswith("not_computed")
    assert "dice" not in res["lung_right"]
    assert res["lung_left"]["status"] == "ok"
    assert "lung_total" not in res
    assert any("missing" in i for i in res["issues"])


def test_empty_lobe_flagged():
    ref = _ref()
    lobes = _lobes(ref)
    lobes["lung_upper_lobe_left"] = nib.Nifti1Image(np.zeros(SHAPE, np.uint8), AFF)
    res = evaluate_lung_segmentation(ref, lobes)
    assert res["pred_lobes"]["lung_upper_lobe_left"]["status"] == "empty"
    assert res["lung_left"]["status"].startswith("not_computed")


def test_off_grid_prediction_rejected():
    ref = _ref()
    lobes = _lobes(ref)
    aff = AFF.copy()
    aff[0, 3] = 4.0
    lobes["lung_lower_lobe_left"] = nib.Nifti1Image(
        np.asanyarray(lobes["lung_lower_lobe_left"].dataobj), aff)
    res = evaluate_lung_segmentation(ref, lobes)
    assert res["pred_lobes"]["lung_lower_lobe_left"]["status"] == "geometry_mismatch"


def test_swapped_prediction_fails_laterality():
    ref = _ref()
    lobes = _lobes(ref)
    swapped = {}
    for n, img in lobes.items():
        swapped[n] = nib.Nifti1Image(np.asanyarray(img.dataobj)[::-1].copy(), AFF)
    res = evaluate_lung_segmentation(ref, swapped)
    assert not res["pred_laterality"]["verified"]
    assert res["lung_left"]["dice"] == pytest.approx(0.0)
