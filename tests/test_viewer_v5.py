"""Viewer v5 (axial sliding-slab MIP): the slab arithmetic, and that viewer v4 is left exactly as it was.
Browser-side checks (what is drawn) are in scripts/test_mip_viewer.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import viewer_v4 as v4  # noqa: E402
import viewer_v5 as v5  # noqa: E402


@pytest.mark.parametrize("dz,mm,images,actual", [
    (0.75, 5, 7, 5.25), (0.75, 8, 11, 8.25), (0.75, 10, 13, 9.75),      # CT-RATE scans with 0.75 mm spacing
    (1.25, 5, 4, 5.0), (1.25, 8, 6, 7.5), (1.25, 10, 8, 10.0),
    (1.0, 5, 5, 5.0), (1.0, 10, 10, 10.0),
    (2.5, 5, 2, 5.0), (2.5, 8, 3, 7.5), (2.5, 10, 4, 10.0),
    (3.0, 5, 2, 6.0), (4.0, 10, 3, 12.0),                                # 2.5 images rounds up to 3
    (6.0, 5, 1, 6.0), (20.0, 10, 1, 20.0),                               # spacing larger than the request: one image
])
def test_thickness_is_rounded_to_whole_images_from_physical_spacing(dz, mm, images, actual):
    p = v5.slab_plan(50, 200, dz, mm)
    assert (p["images"], p["planned_images"]) == (images, images)
    assert p["actual_mm"] == pytest.approx(actual) and p["planned_mm"] == pytest.approx(actual)
    assert not p["truncated"] and p["k0"] <= 50 <= p["k1"]


def test_slab_is_centred_and_an_even_count_puts_the_extra_image_on_the_superior_side():
    odd = v5.slab_plan(50, 200, 0.75, 10)          # 13 images
    assert (odd["k0"], odd["k1"]) == (44, 56)
    even = v5.slab_plan(50, 200, 1.25, 10)         # 8 images: 3 below (inferior), 4 above (superior; larger index)
    assert (even["k0"], even["k1"]) == (47, 54)


def test_slab_is_shortened_at_the_scan_edges_never_shifted_or_padded():
    lo = v5.slab_plan(0, 100, 0.75, 10)
    assert (lo["k0"], lo["k1"], lo["images"], lo["truncated"]) == (0, 6, 7, True) and lo["actual_mm"] == pytest.approx(5.25)
    assert lo["planned_images"] == 13 and lo["planned_mm"] == pytest.approx(9.75)
    hi = v5.slab_plan(99, 100, 0.75, 10)
    assert (hi["k0"], hi["k1"], hi["images"], hi["truncated"]) == (93, 99, 7, True)
    near = v5.slab_plan(3, 100, 0.75, 10)
    assert (near["k0"], near["k1"], near["truncated"]) == (0, 9, True)
    tiny = v5.slab_plan(1, 3, 0.75, 10)            # scan shorter than the slab
    assert (tiny["k0"], tiny["k1"], tiny["images"]) == (0, 2, 3)


def test_every_centre_image_is_inside_its_slab_and_neighbouring_slabs_overlap():
    for dz in (0.5, 0.75, 1.25, 2.5, 5.0):
        for mm in v5.MIP_OPTIONS_MM:
            prev = None
            for k in range(40):
                p = v5.slab_plan(k, 40, dz, mm)
                assert 0 <= p["k0"] <= k <= p["k1"] <= 39 and p["images"] == p["k1"] - p["k0"] + 1 <= p["planned_images"]
                if prev is not None and p["planned_images"] > 1:
                    assert p["k0"] <= prev["k1"]        # shares at least one image with the previous slab
                prev = p


def test_reference_maximum_uses_voxel_values_and_only_the_slab():
    rng = np.random.default_rng(3)
    a = rng.integers(-1024, 400, size=(9, 8, 20)).astype(np.int16)
    a[4, 3, 10] = 3000
    a[2, 2, 0:20] = -1024
    assert v5.slab_max(a, 7, 12)[4, 3] == 3000 and v5.slab_max(a, 11, 15)[4, 3] < 3000
    assert v5.slab_max(a, 0, 19)[2, 2] == -1024
    assert np.array_equal(v5.slab_max(a, 5, 5), a[:, :, 5])


def test_viewer_v4_is_untouched_and_v5_only_adds_to_it():
    assert v4.VIEWER_BUILD == "checklist_viewer_v4_canvas2d" and v5.VIEWER_BUILD == "checklist_viewer_v5_2_canvas2d_mip"
    for text in (v4.JS, v4.VIEWER_HTML, v4.CSS):
        assert "mip" not in text.lower()
    assert v5.JS != v4.JS and len(v5.JS) > len(v4.JS)
    assert v5.EARLY_JS == v4.EARLY_JS and v5.CSS.startswith(v4.CSS)
    for fn in ("canonical_transform", "to_canonical", "ct_payload", "overlay_payload"):      # same geometry and payload code
        assert getattr(v5, fn) is getattr(v4, fn)


def test_page_opens_on_a_single_slice_with_10_mm_selected_and_no_graphics_library():
    html = v5.viewer_html(True)
    assert '<button id="mipoff" class="on"' in html and '<button id="mipon" title' in html
    assert '<button data-mip="5">5 mm</button><button data-mip="8">8 mm</button><button data-mip="10" class="on">10 mm</button>' in html
    assert "const MIP = {on: false, mm: 10," in v5.JS
    assert "webgl" not in (v5.JS + html + v5.CSS).lower() and "niivue" not in (v5.JS + html + v5.CSS).lower()
    assert "if (mipShown(view)) fillMip(s, b.u); else fillSlice(view, s, b.u, true);" in v5.JS       # no outline is drawn over a slab
    assert "if (key && MIP.on) return;" in v5.JS


def test_synthetic_page_builds():
    a = np.zeros((6, 5, 4), np.int16)
    page = v5.standalone_page("t", a, np.diag((1.0, 1.0, 2.0, 1.0)))
    assert v5.VIEWER_BUILD in page and 'id="mipbar"' in page and page.count("const META = ") == 1
