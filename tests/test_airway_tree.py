"""Airway tree prototype on synthetic tubes with known geometry."""
import nibabel as nib
import numpy as np
import pytest

from lung_ct.airway.tree import analyze_airway_tree

R = 3.0  # tube radius, mm
# world (RAS mm) segments: trachea top->carina, left & right main bronchi
CARINA = np.array([0.0, 0.0, 0.0])
SEGMENTS = [(np.array([0.0, 0.0, 40.0]), CARINA),
            (CARINA, np.array([-20.0, 0.0, -25.0])),      # patient left (-x in RAS)
            (CARINA, np.array([20.0, 0.0, -25.0]))]       # patient right


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _make(spacing=(1.0, 1.0, 1.0), axcodes_flip_x=False, spur=False):
    lo, hi = np.array([-30.0, -10.0, -40.0]), np.array([35.0, 10.0, 50.0])
    sp = np.array(spacing)
    shape = np.ceil((hi - lo) / sp).astype(int)
    affine = np.diag([*sp, 1.0])
    affine[:3, 3] = lo
    if axcodes_flip_x:                       # store as LAS: array x runs right->left
        affine[0, 0] = -sp[0]
        affine[0, 3] = hi[0]
    ijk = np.indices(shape).reshape(3, -1).T
    world = (affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
    d = np.min([_seg_dist(world, a, b) for a, b in SEGMENTS], axis=0)
    inside = d <= R
    if spur:   # short bump on the trachea wall
        inside |= np.linalg.norm(world - np.array([R + 1.0, 0, 25.0]), axis=1) <= 2.5
    return nib.Nifti1Image(inside.reshape(shape).astype(np.uint8), affine)


def _by_gen(res):
    return sorted((r for r in res.branches if r["in_tree"]), key=lambda r: r["generation"])


@pytest.mark.parametrize("spacing,flip", [((1.0, 1.0, 1.0), False), ((0.7, 0.7, 1.5), True)])
def test_y_tree_topology_lengths_side(spacing, flip):
    res = analyze_airway_tree(_make(spacing, flip))
    s = res.summary
    assert s["n_tree_branches"] == 3 and s["n_cycles"] == 0
    assert s["branches_per_generation"] == {0: 1, 1: 2}
    tr, b1, b2 = _by_gen(res)
    assert tr["kind"] == "root" and tr["proximal_ras_mm"][2] > 30   # root near top
    # skeleton ends ~R short of the tube end; junction position shifts a few mm
    assert 30 < tr["length_mm"] < 44
    for b in (b1, b2):
        assert 26 < b["length_mm"] < 36             # true centreline 32.0 mm
        assert b["tortuosity"] == pytest.approx(1.0, abs=0.08)
        assert not [k for k in b if "diam" in k or "radius" in k]   # no diameter output
    sides = {b["side"]: b["distal_ras_mm"][0] for b in (b1, b2)}
    assert sides["left"] < 0 < sides["right"]       # side from world x, not array index
    assert s["branches_by_side"] == {"left": 1, "right": 1}
    assert not [f for f in s["flags"] if "root" in f]


def test_short_spur_is_pruned():
    res = analyze_airway_tree(_make(spur=True))
    assert res.summary["n_tree_branches"] == 3
    assert res.summary["prune_iterations"] >= 1


def test_spur_kept_when_pruning_disabled():
    res = analyze_airway_tree(_make(spur=True), {"prune_min_mm": 0.0, "prune_radius_factor": 0.0})
    assert res.summary["n_tree_branches"] > 3


def test_fragments_discarded_and_reported():
    img = _make()
    data = np.asanyarray(img.dataobj).copy()
    data[1:3, 1:3, 1:3] = 1                          # disconnected fragment
    res = analyze_airway_tree(nib.Nifti1Image(data, img.affine))
    assert res.summary["preprocess"]["components"] == 2
    assert res.summary["preprocess"]["discarded_voxels"] == 8
    assert res.summary["n_tree_branches"] == 3


def test_empty_mask_raises():
    with pytest.raises(ValueError, match="empty"):
        analyze_airway_tree(nib.Nifti1Image(np.zeros((10, 10, 10), np.uint8), np.eye(4)))


def test_curved_tube_tortuosity_survives_smoothing():
    """Quarter circle, radius 30 mm: length 47.1 mm, chord 42.4 mm, tortuosity 1.111."""
    sp = np.array([0.7, 0.7, 1.5])
    lo, hi = np.array([-5.0, -8.0, -5.0]), np.array([40.0, 8.0, 40.0])
    shape = np.ceil((hi - lo) / sp).astype(int)
    affine = np.diag([*sp, 1.0])
    affine[:3, 3] = lo
    ijk = np.indices(shape).reshape(3, -1).T
    w = (affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
    # arc centred at (30, 0, 0) in the x-z plane from (0,0,0) up to (30,0,30)
    rel = w - np.array([30.0, 0.0, 0.0])
    ang = np.arctan2(rel[:, 2], -rel[:, 0])
    rxz = np.hypot(rel[:, 0], rel[:, 2])
    d = np.where((ang >= 0) & (ang <= np.pi / 2),
                 np.hypot(rxz - 30.0, rel[:, 1]), np.inf)
    img = nib.Nifti1Image((d <= 2.5).reshape(shape).astype(np.uint8), affine)
    res = analyze_airway_tree(img)
    assert res.summary["n_tree_branches"] == 1
    b = res.branches[0]
    assert 1.07 < b["tortuosity"] < 1.16


def test_small_loop_in_trachea_does_not_shift_generations():
    """A tiny hole through the trachea makes the skeleton loop; after contraction the
    loop node has two edges and must not count as a bifurcation (cases 002/005 bug)."""
    img = _make()
    data = np.asanyarray(img.dataobj).copy()
    ijk = np.indices(data.shape).reshape(3, -1).T
    w = (img.affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
    hole = np.hypot(w[:, 0], w[:, 2] - 25.0) <= 1.0       # along y through the tube axis
    data.reshape(-1)[hole] = 0
    res = analyze_airway_tree(nib.Nifti1Image(data, img.affine))
    s = res.summary
    assert s["branches_per_generation"] == {0: 1, 1: 2}
    assert s["branches_by_side"] == {"left": 1, "right": 1}
    tr = [r for r in res.branches if r["kind"] == "root"][0]
    assert 30 < tr["length_mm"] < 46                     # trachea not split in two


def test_raw_skan_length_reported_and_staircase_bias_documented():
    """Straight oblique tube at anisotropic spacing: the raw Skan pixel-path sum is biased
    upward by the voxel staircase; the smoothed length used for tortuosity is not."""
    res = analyze_airway_tree(_make((0.7, 0.7, 1.5), True))
    for b in (r for r in res.branches if r["generation"] == 1):
        assert b["skan_path_length_mm"] > b["length_mm"] * 1.1
        assert b["length_mm"] / b["endpoint_distance_mm"] == pytest.approx(b["tortuosity"], abs=1e-3)


def test_sitk_and_nibabel_agree_on_physical_coordinates(tmp_path):
    """Explicit LPS (SimpleITK/DICOM) <-> RAS (nibabel) conversion: x and y flip sign."""
    sitk = pytest.importorskip("SimpleITK")
    img = _make((0.7, 0.7, 1.5), True)            # stored LAS
    path = tmp_path / "m.nii.gz"
    nib.save(img, str(path))
    s = sitk.ReadImage(str(path))
    for ijk in [(0, 0, 0), (10, 5, 20), (40, 12, 50)]:
        ras = (img.affine @ np.append(ijk, 1.0))[:3]
        lps = np.array(s.TransformIndexToPhysicalPoint([int(v) for v in ijk]))
        assert np.allclose(lps, ras * np.array([-1, -1, 1]), atol=1e-4)


def test_two_level_bifurcation_generations_and_counts():
    """Trachea -> 2 main bronchi -> right bronchus splits again: gens {0:1, 1:2, 2:2}."""
    global SEGMENTS
    saved = SEGMENTS
    rb_end = np.array([20.0, 0.0, -25.0])
    SEGMENTS = saved + [(rb_end, np.array([10.0, 0.0, -33.0])),
                        (rb_end, np.array([28.0, 0.0, -33.0]))]
    try:
        res = analyze_airway_tree(_make((0.7, 0.7, 1.0)))
    finally:
        SEGMENTS = saved
    s = res.summary
    assert s["branches_per_generation"] == {0: 1, 1: 2, 2: 2}
    assert s["n_tree_branches"] == 5 and s["n_terminal_branches"] == 3
    g2 = [b for b in res.branches if b["generation"] == 2]
    assert all(b["side"] == "right" for b in g2)
    rb = [b for b in res.branches if b["generation"] == 1 and b["side"] == "right"][0]
    assert rb["kind"] == "internal" and 26 < rb["length_mm"] < 36


def test_sensitivity_rating_rules():
    from lung_ct.airway.sensitivity import rate, settings
    assert rate("gen1_branches", [2, 2, 2], 2) == ("stable", None)
    assert rate("gen1_branches", [2, 3, 2], 2)[0] == "unstable"
    assert rate("total_centreline_mm", [100, 104, 101], 100)[0] == "stable"
    assert rate("total_centreline_mm", [100, 110, 98], 100)[0] == "moderate"
    assert rate("total_centreline_mm", [100, 130, 98], 100)[0] == "unstable"
    assert rate("gen1_left_length_mm", [50, None], 50)[0] == "unstable"
    s = settings(1.5)
    assert s[0][0] == "default" and len(s) == 11
    assert ("smoothing_x_coarsest_spacing", 2.0, {"path_smoothing_sigma_mm": 3.0}) in s


def test_tortuosity_rated_on_excess_over_one():
    from lung_ct.airway.sensitivity import rate
    # 1.02 -> 1.05: 3% of the value but 150% of the excess
    assert rate("median_tortuosity_internal", [1.02, 1.05, 1.01], 1.02)[0] == "unstable"
    assert rate("median_tortuosity_internal", [1.020, 1.0205], 1.02)[0] == "stable"


def test_aggregate_and_central_statement_agree_with_per_case_rows():
    from lung_ct.airway.sensitivity import (CENTRAL_STATEMENT, aggregate_ratings,
                                            central_exceptions, central_summary,
                                            tortuosity_by_smoothing)
    rows = [
        {"case_id": "a", "metric": "trachea_length_mm", "rating": "stable",
         "default": "90", "min": "88", "max": "91", "most_influential": "x=1"},
        {"case_id": "b", "metric": "trachea_length_mm", "rating": "moderate",
         "default": "90", "min": "85", "max": "95", "most_influential": "x=1"},
        {"case_id": "b", "metric": "gen1_left_length_mm", "rating": "unstable",
         "default": "14.11", "min": "14.04", "max": "54.04",
         "most_influential": "prune_radius_factor=2.0"},
        {"case_id": "a", "metric": "n_tree_branches", "rating": "unstable",
         "default": "200", "min": "150", "max": "220", "most_influential": "p=1"},
    ]
    agg = aggregate_ratings(rows)
    assert agg["trachea_length_mm"] == {"stable": 1, "moderate": 1, "unstable": 0,
                                        "unstable_cases": []}
    assert sum(sum(v[k] for k in ("stable", "moderate", "unstable")) for v in agg.values()) == 4
    assert [r["case_id"] for r in central_exceptions(rows)] == ["b"]   # n_tree is not central
    s = central_summary(rows)
    assert s.startswith(CENTRAL_STATEMENT) and "b gen1_left_length_mm 14-54 mm" in s
    long = [{"param": "default", "value": "", "metric": "median_tortuosity_internal", "result": "1.02"},
            {"param": "smoothing_x_coarsest_spacing", "value": "0.5",
             "metric": "median_tortuosity_internal", "result": "1.05"},
            {"param": "prune_min_mm", "value": "1.0", "metric": "median_tortuosity_internal",
             "result": "1.03"}]
    t = tortuosity_by_smoothing(long)["median_tortuosity_internal"]
    assert set(t) == {"default (1.0 x coarsest spacing)", "0.5 x coarsest spacing"}
    assert t["0.5 x coarsest spacing"]["excess"][0] == pytest.approx(0.05)


def test_branch_paths_match_reported_endpoints():
    res = analyze_airway_tree(_make((0.7, 0.7, 1.5), True))
    for b in (r for r in res.branches if r["in_tree"]):
        pts = res.branch_paths[b["branch_id"]]
        assert np.allclose(pts[0], b["proximal_ras_mm"], atol=0.01)
        assert np.allclose(pts[-1], b["distal_ras_mm"], atol=0.01)
