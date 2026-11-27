"""Viewer v6 (Axial 2.5 mm mean slab, MIP 10 mm, native slices under Details): slab arithmetic and page text.
What is drawn in a browser is checked by scripts/test_slab_viewer.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import viewer_v4 as v4  # noqa: E402
import viewer_v5 as v5  # noqa: E402
import viewer_v6 as v6  # noqa: E402


@pytest.mark.parametrize("dz,images,mm", [
    (0.75, 3, 2.25), (1.25, 2, 2.5), (0.5, 5, 2.5), (0.625, 4, 2.5), (1.0, 3, 3.0),      # 2.5 images rounds up to 3
    (1.5, 2, 3.0), (2.0, 1, 2.0), (2.5, 1, 2.5), (3.0, 1, 3.0), (5.0, 1, 5.0),          # from 1.67 mm spacing: one image, nothing averaged
])
def test_axial_2_5_mm_is_a_whole_number_of_images_from_physical_spacing(dz, images, mm):
    p = v6.slab_plan(50, 200, dz, v6.AXIAL_MM)
    assert p["planned_images"] == images and p["actual_mm"] == pytest.approx(mm) and p["k0"] <= 50 <= p["k1"]


def test_only_two_slab_sizes_exist():
    assert (v6.AXIAL_MM, v6.MIP_MM) == (2.5, 10)
    assert "data-mip" not in v6.VIEWER_HTML and "5 mm</button>" not in v6.VIEWER_HTML.replace("2.5 mm</button>", "")
    assert ">Axial 2.5 mm</button>" in v6.VIEWER_HTML and ">MIP 10 mm</button>" in v6.VIEWER_HTML and ">Details</button>" in v6.VIEWER_HTML


def test_mean_is_exact_sum_then_one_division_with_halves_up():
    a = np.zeros((2, 2, 4), np.int16)
    a[0, 0] = [1, 2, 0, 0]          # mean of first two = 1.5 -> 2
    a[0, 1] = [-1, -2, 0, 0]        # -1.5 -> -1 (halves up, towards +infinity)
    a[1, 0] = [3071, 3071, 3071, -1024]
    a[1, 1] = [-1000, 2500, -1000, 0]
    m2 = v6.slab_mean(a, 0, 1)
    assert m2[0, 0] == 2 and m2[0, 1] == -1 and m2[1, 0] == 3071
    m3 = v6.slab_mean(a, 0, 2)
    assert m3[1, 1] == 167 and m3[1, 0] == 3071          # (-1000 + 2500 - 1000) / 3 = 166.67 -> 167
    assert np.array_equal(v6.slab_mean(a, 2, 2), a[:, :, 2])           # one image: unchanged
    assert m3.dtype == np.int16


def test_mean_does_not_overflow_int16_sums():
    a = np.full((1, 1, 20), 3071, np.int16)
    assert v6.slab_mean(a, 0, 19)[0, 0] == 3071


def test_mean_differs_from_skipping_slices():
    rng = np.random.default_rng(0)
    a = rng.integers(-1000, 400, (8, 8, 9)).astype(np.int16)
    p = v6.slab_plan(4, 9, 0.75, 2.5)
    assert p["images"] == 3
    assert not np.array_equal(v6.slab_mean(a, p["k0"], p["k1"]), a[:, :, 4])


@pytest.mark.parametrize("text,want", [
    ("Non-contrast images were taken in the axial plane with a section thickness of 1.5 mm.", 1.5),
    ("with a slice thickness of 1,25 mm", 1.25),
    ("Sections were taken without contrast medium and reconstructions were made at the workstation.", None),
    ("", None), (None, None),
])
def test_thickness_is_taken_only_from_an_explicit_statement(text, want):
    assert v6.reported_thickness_mm(text) == want


def test_page_text_separates_spacing_from_thickness_and_says_the_view_is_derived():
    h = v6.VIEWER_HTML
    assert "it is not the slice thickness" in h and "it is not a reconstruction from the scanner" in h
    assert "before brightness / contrast" in h and "Outlines can be shown only here" in h
    assert "from the file" in v6.JS and "as stated in the radiology report (not in the image file)" in v6.JS and "not recorded" in v6.JS


def test_opening_view_is_the_mean_slab_and_outlines_are_blocked_in_slab_views():
    assert "const SL = {mode: 'mean'" in v6.JS
    assert "if (key && slabOn()) return;" in v6.JS and "if (slabOn()) setSlab('native');" in v6.JS
    assert "body.slabmode #ovbar{display:none}" in v6.CSS


def test_earlier_viewers_are_not_changed_by_v6():
    assert "slabOn" not in v4.JS and "slabOn" not in v5.JS and "sl25" not in v5.VIEWER_HTML
    assert v5.VIEWER_BUILD == "checklist_viewer_v5_2_canvas2d_mip" and v6.VIEWER_BUILD == "checklist_viewer_v6_canvas2d_axial25_mip10"
    assert v6.slab_plan is v5.slab_plan


def test_standalone_page_carries_build_and_slab_information():
    arr = np.zeros((6, 6, 6), np.int16)
    page = v6.standalone_page("t", arr, np.diag((1.0, 1.0, 0.75, 1.0)), slab={"reported_thickness_mm": 1.5, "reconstruction_note": "note"})
    assert v6.VIEWER_BUILD in page and '"reported_thickness_mm": 1.5' in page and 'id="slbar"' in page and 'id="mipbar"' not in page
