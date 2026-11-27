"""Synthetic checks for reference-answer derivation and VLM scoring."""
import json

import nibabel as nib
import numpy as np
import pytest

from lung_ct.evaluation.reference_answers import assign_sides, reference_answers
from lung_ct.evaluation.score_vlm import score_case, summarise

SHAPE = (60, 40, 60)


def _lungs(affine):
    """Two box 'lungs'. Array axis 0 is world x under the given affine."""
    lung = np.zeros(SHAPE, np.uint8)
    lung[5:25, 5:35, 5:55] = 1    # low i
    lung[35:55, 5:35, 5:55] = 2   # high i
    return lung


def _imgs(lung, inf, affine):
    return nib.Nifti1Image(inf.astype(np.uint8), affine), nib.Nifti1Image(lung, affine)


RAS = np.diag([1.0, 1.0, 1.0, 1.0])            # +i = patient right
LAS = np.diag([-1.0, 1.0, 1.0, 1.0])           # +i = patient left (stored flipped)


def test_sides_from_world_x_not_index():
    lung = _lungs(RAS)
    s_ras = assign_sides(nib.Nifti1Image(lung, RAS))
    s_las = assign_sides(nib.Nifti1Image(lung, LAS))
    assert (s_ras["left_label"], s_ras["right_label"]) == (1, 2)
    assert (s_las["left_label"], s_las["right_label"]) == (2, 1)


def test_unilateral_right_lower_peripheral():
    lung = _lungs(RAS)
    inf = np.zeros(SHAPE, bool)
    inf[35:37, 5:35, 5:15] = True   # label 2 (right in RAS), lowest z, at lung edge
    r = reference_answers(*_imgs(lung, inf, RAS))
    a = r["answers"]
    assert a["opacity_presence"] == "present"
    assert a["distribution"]["laterality"] == "right"
    assert a["distribution"]["craniocaudal"] == "lower_predominant"
    assert a["distribution"]["axial"] == "peripheral"


def test_flipped_storage_changes_side():
    lung = _lungs(RAS)
    inf = np.zeros(SHAPE, bool)
    inf[35:37, 5:35, 5:15] = True
    a = reference_answers(*_imgs(lung, inf, LAS))["answers"]
    assert a["distribution"]["laterality"] == "left"


def test_bilateral_central_no_predominance():
    lung = _lungs(RAS)
    inf = np.zeros(SHAPE, bool)
    inf[13:17, 18:22, 10:50] = True   # centre of lung 1, spans all heights
    inf[43:47, 18:22, 10:50] = True   # centre of lung 2
    a = reference_answers(*_imgs(lung, inf, RAS))["answers"]
    assert a["distribution"] == {"laterality": "bilateral",
                                 "craniocaudal": "no_predominance", "axial": "central"}


def test_absent_when_no_infection():
    lung = _lungs(RAS)
    a = reference_answers(*_imgs(lung, np.zeros(SHAPE, bool), RAS))["answers"]
    assert a["opacity_presence"] == "absent"
    assert set(a["distribution"].values()) == {"not_applicable"}


def test_infection_outside_lung_ignored_but_reported():
    lung = _lungs(RAS)
    inf = np.zeros(SHAPE, bool)
    inf[0:3, 0:3, 0:3] = True
    r = reference_answers(*_imgs(lung, inf, RAS))
    assert r["answers"]["opacity_presence"] == "absent"
    assert r["measurements"]["infection_ml_outside_lung"] > 0


def test_grid_mismatch_raises():
    lung = _lungs(RAS)
    with pytest.raises(ValueError):
        reference_answers(nib.Nifti1Image(np.zeros(SHAPE, np.uint8), np.eye(4) * 2),
                          nib.Nifti1Image(lung, RAS))


def test_ctss_from_lobes():
    lung = _lungs(RAS)
    inf = np.zeros(SHAPE, bool)
    inf[35:45, 5:35, 5:55] = True   # half of lung 2
    lobes = {}
    names = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
             "lung_upper_lobe_left", "lung_lower_lobe_left"]
    for i, n in enumerate(names):
        m = np.zeros(SHAPE, np.uint8)
        if n == "lung_lower_lobe_right":
            m[35:55, 5:35, 5:55] = 1          # 50% involved -> score 3
        else:
            m[5:25, 5:35, 5 + 10 * i:15 + 10 * i] = 1   # uninvolved -> 0
        lobes[n] = nib.Nifti1Image(m, RAS)
    r = reference_answers(*_imgs(lung, inf, RAS), lobe_imgs=lobes)
    c = r["measurements"]["ct_severity_score"]
    assert c["lobes"]["lung_lower_lobe_right"]["score"] == 3
    assert c["total"] == 3


def test_scoring_outcomes_and_rates():
    ref = {"answers": {"opacity_presence": "present",
                       "distribution": {"laterality": "bilateral",
                                        "craniocaudal": "lower_predominant",
                                        "axial": "peripheral"}}}
    m1 = {"opacity_presence": "present",
          "distribution": {"laterality": "bilateral", "craniocaudal": "indeterminate",
                           "axial": "central"}}
    s1, s2 = score_case(m1, ref), score_case(None, ref)
    assert s1["laterality"]["outcome"] == "agree"
    assert s1["craniocaudal"]["outcome"] == "indeterminate"
    assert s1["axial"]["outcome"] == "disagree"
    assert s2["laterality"]["outcome"] == "no_valid_output"
    summ = summarise([s1, s2])
    assert summ["laterality"]["agreement_all"] == 0.5
    assert summ["laterality"]["agreement_committed"] == 1.0
    assert summ["craniocaudal"]["agreement_committed"] is None


def test_equal_volume_zones_remove_height_bias():
    """Lung wide at the bottom, narrow at the top: uniform disease is middle-heavy by height
    thirds but has no predominance with equal-volume zones."""
    from lung_ct.evaluation.reference_answers import load_rules
    lung = np.zeros(SHAPE, np.uint8)
    for z in range(5, 55):
        w = 2 + int(18 * (55 - z) / 50)          # wider inferiorly
        lung[25 - w:25, 5:35, z] = 1
        lung[35:35 + w, 5:35, z] = 2
    inf = lung > 0
    v1 = dict(load_rules())
    v2 = json.loads(json.dumps({k: v for k, v in v1.items() if not k.startswith("_")}))
    v2["craniocaudal"]["zone_method"] = "equal_volume"
    v2["rules_version"], v2["_sha256"] = "test_v2", "x"
    a2 = reference_answers(*_imgs(lung, inf, RAS), rules=v2)
    assert a2["answers"]["distribution"]["craniocaudal"] == "no_predominance"
    shares = a2["measurements"]["craniocaudal"]["infection_share_by_zone"]
    assert all(abs(s - 1 / 3) < 0.05 for s in shares.values())
