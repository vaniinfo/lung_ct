"""Exploratory airway calibre on synthetic tubes with known geometry, and patient-level split rules.

First run 2026-11-23. Tolerances were set before the first run and were not changed after it; one fixture (the cone) was
moved off the voxel lattice because the baseline skeletoniser returned an empty skeleton for a perfectly symmetric object.
"""
import nibabel as nib
import numpy as np
import pytest

from lung_ct.airway import calibre, cohort
from lung_ct.airway.tree import analyze_airway_tree
from test_airway_tree import R, _make


def _cone(r_top=5.0, r_bottom=2.0, length=60.0, sp=(0.8, 0.8, 0.8)):
    """Straight tube along z whose radius falls linearly from r_top (z = length) to r_bottom (z = 0)."""
    lo, hi = np.array([-10.0, -10.0, -5.0]), np.array([10.0, 10.0, length + 5.0])
    sp = np.array(sp)
    shape = np.ceil((hi - lo) / sp).astype(int)
    affine = np.diag([*sp, 1.0])
    affine[:3, 3] = lo
    ijk = np.indices(shape).reshape(3, -1).T
    w = (affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
    z = np.clip(w[:, 2], 0, length)
    rad = r_bottom + (r_top - r_bottom) * z / length
    # axis slightly off the voxel lattice: with the axis exactly between voxels (a perfectly symmetric object) scikit-image's Lee
    # thinning returned an EMPTY skeleton at 0.8 mm (seen 2026-11-23) and the tree stage then fails; real masks are not symmetric
    inside = (np.hypot(w[:, 0] - 0.31, w[:, 1] - 0.17) <= rad) & (w[:, 2] >= 0) & (w[:, 2] <= length)
    return nib.Nifti1Image(inside.reshape(shape).astype(np.uint8), affine)


@pytest.mark.parametrize("spacing,tol", [((1.0, 1.0, 1.0), 0.6), ((0.7, 0.7, 1.5), 1.0)])
def test_y_tree_diameter_in_mm_matches_the_known_tube(spacing, tol):
    img = _make(spacing)
    res = analyze_airway_tree(img)
    rows = calibre.measure_branches(img, res)
    assert len(rows) == 3
    for r in rows:
        assert r["area_equivalent_diameter_mm"] == pytest.approx(2 * R, abs=tol)        # 6 mm tube, in mm whatever the voxel size
        assert r["mean_area_mm2"] == pytest.approx(np.pi * R ** 2, rel=0.25)
        assert 0.8 < r["calibre_ratio"] < 1.5 and "non_circular_or_merged_lumen" not in r["flags"]
        assert r["usable_length_mm"] < r["length_mm"]                                    # both ends are trimmed


def test_same_tube_at_two_voxel_sizes_gives_the_same_physical_diameter():
    d = []
    for sp in ((1.0, 1.0, 1.0), (0.7, 0.7, 1.5)):
        img = _make(sp)
        rows = calibre.measure_branches(img, analyze_airway_tree(img))
        d.append(np.median([r["area_equivalent_diameter_mm"] for r in rows]))
    assert d[0] == pytest.approx(d[1], abs=0.8)


def test_cone_tapers_distally_with_the_expected_sign_and_size():
    img = _cone()
    res = analyze_airway_tree(img)
    rows = calibre.measure_branches(img, res)
    root = next(r for r in rows if r["kind"] == "root")
    # diameter 10 -> 4 mm over 60 mm: d ln(d) / ds = ln(0.4) / 60 = -1.53 % per mm, reported as +1.53 (narrowing distally)
    assert root["n_taper_windows"] >= 3
    assert 1.0 < root["intratapering_pct_per_mm"] < 2.1


def test_uniform_tube_has_almost_no_intratapering_and_children_report_change_against_parent():
    img = _make((1.0, 1.0, 1.0))
    rows = calibre.measure_branches(img, analyze_airway_tree(img))
    for r in rows:
        if r["intratapering_pct_per_mm"] is not None:
            assert abs(r["intratapering_pct_per_mm"]) < 0.8
    kids = [r for r in rows if r["generation"] == 1]
    assert len(kids) == 2 and all(r["parent_branch_id"] is not None for r in kids)
    assert all(abs(r["intertapering_pct"]) < 15 for r in kids)                           # same radius as the parent tube


def test_lobe_comes_from_world_coordinates_and_central_when_outside():
    img = _make((1.0, 1.0, 1.0))
    res = analyze_airway_tree(img)
    lab = np.zeros(img.shape, np.uint8)
    ijk = np.indices(img.shape).reshape(3, -1).T
    w = (img.affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
    lab.reshape(-1)[(w[:, 2] < -5) & (w[:, 0] > 0)] = 1                                   # patient right, below the carina
    lab.reshape(-1)[(w[:, 2] < -5) & (w[:, 0] < 0)] = 4
    rows = calibre.measure_branches(img, res, lobe_img=nib.Nifti1Image(lab, img.affine), lobe_names={1: "RUL", 4: "LUL"})
    by = {r["side"]: r for r in rows if r["generation"] == 1}
    assert by["right"]["lobe"] == "RUL" and by["left"]["lobe"] == "LUL"
    assert next(r for r in rows if r["generation"] == 0)["lobe"] == "central"


def test_case_quality_flags_fragments_edges_and_default_spacing():
    img = _make((1.0, 1.0, 1.0))
    q = calibre.case_quality(img, analyze_airway_tree(img))
    assert not any(f.startswith("broken_airway_mask") for f in q["flags"])
    a = np.asanyarray(img.dataobj).copy()
    a[-12:, :12, -12:] = 1                                                                # a large separate piece at a corner far from the tubes
    img2 = nib.Nifti1Image(a, img.affine)
    q2 = calibre.case_quality(img2, analyze_airway_tree(img2))
    assert any(f.startswith("broken_airway_mask") for f in q2["flags"])
    assert any(f.startswith("airway_mask_touches_image_edge") for f in q2["flags"])
    ident = nib.Nifti1Image(np.asanyarray(img.dataobj), np.eye(4))
    q3 = calibre.case_quality(ident, analyze_airway_tree(ident))
    assert any("spacing_may_be_a_default" in f for f in q3["flags"])


def test_no_voxel_unit_columns_and_method_never_claims_airquant():
    img = _make((1.0, 1.0, 1.0))
    rows = calibre.measure_branches(img, analyze_airway_tree(img))
    assert not [k for k in rows[0] if "voxel" in k or "pixel" in k]
    assert "NOT AirQuant" in calibre.METHOD and "unvalidated" in calibre.METHOD


def test_case_features_leave_out_blocked_branches():
    rows = [{"generation": 2, "lobe": "RLL", "flags": "", "area_equivalent_diameter_mm": 4.0, "mean_area_mm2": 12.6, "intratapering_pct_per_mm": 1.0, "intertapering_pct": 10.0},
            {"generation": 2, "lobe": "RLL", "flags": "non_circular_or_merged_lumen", "area_equivalent_diameter_mm": 40.0, "mean_area_mm2": 1256.0,
             "intratapering_pct_per_mm": None, "intertapering_pct": None}]
    f = {(x["feature"], x["group"]): x for x in calibre.case_features(rows)}
    assert f[("median_area_equivalent_diameter_mm", "generation_2")]["value"] == 4.0
    assert f[("median_area_equivalent_diameter_mm", "generation_2")]["n_branches"] == 1
    assert f[("n_branches_total", "lobe_RLL")]["value"] == 2


# ---------------- patient-level splits ----------------
def test_ctrate_patient_id_groups_scans_and_reconstructions():
    assert cohort.patient_id_from_case("valid_301_a_2") == cohort.patient_id_from_case("valid_301_b_1") == "valid_301"
    assert cohort.patient_id_from_case("coronacases_001") == "coronacases_001"


def test_splits_are_per_patient_deterministic_and_stratified():
    cases = [{"case_id": f"valid_{p}_a_{r}", "label": int(p % 5 == 0)} for p in range(1, 101) for r in (1, 2)]
    labels, conflict = cohort.patient_labels(cases)
    assert not conflict and len(labels) == 100
    s1, s2 = cohort.assign_splits(labels, 7), cohort.assign_splits(labels, 7)
    assert s1 == s2 and cohort.assign_splits(labels, 8) != s1
    case_patient = {c["case_id"]: cohort.patient_id_from_case(c["case_id"]) for c in cases}
    cohort.check_no_leakage({c: s1[p] for c, p in case_patient.items()}, case_patient)
    pos = [s1[p] for p, v in labels.items() if v == 1]
    assert {x: pos.count(x) for x in cohort.SPLITS} == {"train": 12, "validation": 4, "test": 4}


def test_leakage_and_conflicting_labels_are_caught():
    with pytest.raises(ValueError):
        cohort.check_no_leakage({"valid_1_a_1": "train", "valid_1_a_2": "test"}, {"valid_1_a_1": "valid_1", "valid_1_a_2": "valid_1"})
    labels, conflict = cohort.patient_labels([{"case_id": "valid_1_a_1", "label": 1}, {"case_id": "valid_1_a_2", "label": 0}])
    assert conflict == ["valid_1"] and labels == {}


def test_group_comparison_is_descriptive_and_signed():
    r = cohort.compare_groups([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0])
    assert r["rank_biserial"] == 1.0 and r["median_difference"] == 4.0 and r["n_cases"] == 4
    assert not [k for k in r if "sensitivity" in k or "specificity" in k or "auc" in k or "threshold" in k]
    assert cohort.compare_groups([5.0], [1.0, 2.0])["rank_biserial"] is None            # too few to compare
