"""2D canvas viewer (viewer v4): geometry is decided at build time from the affine. These tests use an asymmetric
phantom defined in PHYSICAL (patient) terms and re-store it in all 48 axis orders / flips with hand-built affines
(no nibabel involved in making the variants), with anisotropic spacing. Every variant must come back as the same
RAS+ array. Oblique affines must be refused, never shown as if axis-aligned."""
import base64
import hashlib
import itertools
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import viewer_v3 as v3  # noqa: E402
import viewer_v4 as v4  # noqa: E402

SHAPE = (11, 9, 7)                       # RAS+ grid: axis 0 -> Right, 1 -> Anterior, 2 -> Superior
SPACING = (0.8, 1.3, 2.5)                # mm, anisotropic
ORIGIN = (-37.0, 12.5, -140.0)           # mm position of voxel (0, 0, 0)


def phantom():
    """Base RAS+ volume: every voxel has a different value; a bright block at Left-Anterior-Superior."""
    i, j, k = np.indices(SHAPE)
    a = (i + 13 * j + 131 * k - 1000).astype(np.int16)
    a[0:3, 6:9, 5:7] = 3000              # low i = patient LEFT, high j = ANTERIOR, high k = SUPERIOR
    mask = np.zeros(SHAPE, np.uint8)
    mask[1:4, 5:8, 4:6] = 1              # overlay near the same corner, not symmetric
    aff = np.diag(SPACING + (1.0,))
    aff[:3, 3] = ORIGIN
    return a, mask, aff


def variant(arr, aff, perm, flips):
    """Store the same physical volume with another axis order / direction. Affine built by hand."""
    out = np.transpose(arr, perm)
    m = aff[:3, :3][:, list(perm)].copy()
    t = aff[:3, 3].copy()
    for ax in range(3):
        if flips[ax]:
            out = np.flip(out, axis=ax)
            t = t + m[:, ax] * (out.shape[ax] - 1)
            m[:, ax] = -m[:, ax]
    v = np.eye(4)
    v[:3, :3], v[:3, 3] = m, t
    return np.ascontiguousarray(out), v


ALL = [(p, f) for p in itertools.permutations(range(3)) for f in itertools.product((0, 1), repeat=3)]


def test_every_axis_order_and_flip_returns_the_same_ras_volume():
    base, mask, aff = phantom()
    for perm, flips in ALL:
        a, v = variant(base, aff, perm, flips)
        m, _ = variant(mask, aff, perm, flips)
        ornt, can, codes, ang = v4.canonical_transform(v, a.shape)
        assert ang == pytest.approx(0, abs=1e-9)
        back = v4.to_canonical(a, ornt)
        assert back.shape == SHAPE and np.array_equal(back, base), (perm, flips, codes)
        assert np.array_equal(v4.to_canonical(m, ornt), mask), (perm, flips)
        assert np.allclose(can, aff, atol=1e-9), (perm, flips)            # same physical position for every voxel
    # the physical statement behind the phantom: the bright block is at patient Left, Anterior, Superior
    _, can, _, _ = v4.canonical_transform(aff, SHAPE)
    c = np.argwhere(base == 3000).mean(0)
    mm = can @ np.append(c, 1)
    centre = can @ np.append((np.array(SHAPE) - 1) / 2, 1)
    assert mm[0] < centre[0] and mm[1] > centre[1] and mm[2] > centre[2]  # RAS+: smaller x = Left, larger y = Anterior, larger z = Superior


def test_meta_spacing_affine_and_named_orientation():
    base, mask, aff = phantom()
    a, v = variant(base, aff, (1, 0, 2), (1, 1, 0))       # LPS-like storage with swapped in-plane axes
    ornt, can, codes, ang = v4.canonical_transform(v, a.shape)
    html, meta = v4.ct_payload(v4.to_canonical(a, ornt), can, codes, ang)
    assert meta["dims"] == list(SHAPE) and meta["spacing"] == pytest.approx(SPACING)
    assert np.allclose(np.array(meta["affine"]) @ np.array(meta["inv_affine"]), np.eye(4), atol=1e-9)
    assert meta["original_axis_codes"] == codes and len(codes) == 3 and meta["obliquity_deg"] == 0
    assert meta["min"] == int(base.min()) and meta["max"] == 3000 and "clipped" in meta["clip_note"]
    # lossless: the embedded stream decodes to the canonical array, and the per-piece hashes cover every voxel byte
    b64 = "".join(x.split(">")[1].split("<")[0] for x in html.strip().split("\n"))
    stream = zlib.decompress(base64.b64decode(b64), wbits=31)
    assert np.array_equal(v3.decode_ct_stream(stream, meta), base)
    raw = base.astype("<i2").tobytes(order="F")
    assert meta["chunk_sha256"] == [hashlib.sha256(raw[i:i + meta["hash_chunk"]]).hexdigest() for i in range(0, len(raw), meta["hash_chunk"])]
    assert meta["sha256"] == hashlib.sha256(raw).hexdigest()


def test_overlay_bounding_box_is_exact():
    _, mask, _ = phantom()
    html, m = v4.overlay_payload("x", mask)
    raw = zlib.decompress(base64.b64decode(html.split(">")[1].split("<")[0]), wbits=31)
    sub = np.frombuffer(raw, np.uint8).reshape(m["dims"], order="F")
    back = np.zeros(SHAPE, np.uint8)
    o, d = m["offset"], m["dims"]
    back[o[0]:o[0] + d[0], o[1]:o[1] + d[1], o[2]:o[2] + d[2]] = sub
    assert np.array_equal(back, mask) and m["offset"] == [1, 5, 4] and m["dims"] == [3, 3, 2] and m["voxels_on"] == 18


def _rot(deg, axis):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    r = np.eye(4)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def test_oblique_volumes_are_refused_not_assumed_axis_aligned():
    base, _, aff = phantom()
    for deg, axis in ((5.0, 0), (12.0, 1), (1.0, 2), (30.0, 0)):
        tilted = _rot(deg, axis) @ aff
        assert v4.obliquity_deg(tilted) == pytest.approx(deg, abs=1e-6)
        with pytest.raises(v4.ObliqueOrientationError, match="oblique"):
            v4.canonical_transform(tilted, SHAPE)
        with pytest.raises(v4.ObliqueOrientationError):
            v4.standalone_page("t", base, tilted)
    small = _rot(0.2, 0) @ aff                                   # within the stated tolerance: accepted and recorded
    _, _, _, ang = v4.canonical_transform(small, SHAPE)
    assert 0.19 < ang < 0.21 <= v4.MAX_OBLIQUE_DEG
    with pytest.raises(v4.ObliqueOrientationError):
        v4.canonical_transform(np.diag([1.0, 1.0, 0.0, 1.0]) + np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), SHAPE)


def test_page_has_no_webgl_library_and_states_the_display_convention():
    base, mask, aff = phantom()
    page = v4.standalone_page("phantom", base, aff, {"m": ("Marker", "#ff0000", mask)}, {"m": [0.0, 0.0, 0.0]})
    low = page.lower()
    assert "niivue" not in low and "webgl" not in low and "getcontext('2d')" in low
    assert "left = Right, top = Anterior" in page and "left = Anterior, top = Superior" in page
    assert page.count('class="vp" data-vol="ct"') >= 1 and 'data-vol="m"' in page


def test_embedded_page_scripts_are_valid_javascript(tmp_path):
    """A syntax error in an embedded script silently removes the grading form or the viewer from every page built."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js not installed")
    import viewer_grading as g1
    import viewer_grading_v2 as g2
    scripts = {"grading_v1": g1.JS.replace("__GRADING__", "{}"), "grading_v2": g2.JS.replace("__GRADING__", "{}"),
               "viewer_v3": v3.JS.replace("__META__", "{}"), "viewer_v4": v4.JS.replace("__META__", "{}"), "early": v3.EARLY_JS, "early_v4": v4.EARLY_JS}
    for name, js in scripts.items():
        f = tmp_path / f"{name}.js"
        f.write_text(js, encoding="utf-8")
        r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{name}: {r.stderr[:400]}"
