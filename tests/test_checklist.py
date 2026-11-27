"""Checklist measurements and decision rules on a synthetic scan stored in LPS order (like CT-RATE),
so 'right', 'front' and 'up' are only correct if the code reorients properly."""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lung_ct.agent.checklist import (banned_hits, build_checklist, load_checklist_config,  # noqa: E402
                                     rule_based_checklist_report, validate_checklist_report)
from lung_ct.agent.conditions import dependent_band, measure_conditions  # noqa: E402
from lung_ct.agent.report import validate_report  # noqa: E402

CK = load_checklist_config(ROOT / "configs" / "checklist_v1.json")
# LPS storage: index i -> patient LEFT, j -> POSTERIOR, k -> SUPERIOR; 2 mm voxels
AFF = np.diag([-2.0, -2.0, 2.0, 1.0])
SHAPE = (80, 60, 40)


def _scan(tmp: Path):
    hu = np.full(SHAPE, -1000, np.int16)
    hu[5:75, 5:55, 2:38] = 40                      # body
    right = np.zeros(SHAPE, bool); right[8:36, 10:50, 5:35] = True   # low i = patient right
    left = np.zeros(SHAPE, bool); left[44:72, 10:50, 5:35] = True
    hu[right | left] = -850
    # ground-glass block, ANTERIOR right lung (low j), upper part
    hu[12:30, 12:22, 22:32] = -500
    # same-size block at the BACK of the left lung (high j): gravity-dependent, must not count
    hu[50:68, 44:50, 8:14] = -500
    heart = np.zeros(SHAPE, bool); heart[34:50, 12:30, 8:20] = True
    hu[heart] = 40
    aorta = np.zeros(SHAPE, bool); aorta[40:44, 30:34, 5:35] = True
    hu[aorta] = 45
    hu[40, 30, 10:14] = 600                        # calcified wall spot
    eff = np.zeros(SHAPE, bool); eff[10:34, 46:50, 6:12] = True     # right, posterior
    seg = tmp / "seg"
    for sub, name, m in [("total", "lung_upper_lobe_right", right & (np.arange(40) >= 20)[None, None, :]),
                         ("total", "lung_middle_lobe_right", np.zeros(SHAPE, bool)),
                         ("total", "lung_lower_lobe_right", right & (np.arange(40) < 20)[None, None, :]),
                         ("total", "lung_upper_lobe_left", left & (np.arange(40) >= 20)[None, None, :]),
                         ("total", "lung_lower_lobe_left", left & (np.arange(40) < 20)[None, None, :]),
                         ("total", "heart", heart), ("total", "aorta", aorta),
                         ("pleural_pericard_effusion", "pleural_effusion", eff),
                         ("pleural_pericard_effusion", "pericardial_effusion", np.zeros(SHAPE, bool))]:
        (seg / sub).mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(m.astype(np.uint8), AFF), str(seg / sub / f"{name}.nii.gz"))
    ct = nib.Nifti1Image(hu, AFF)
    lab = np.zeros(SHAPE, np.uint8); lab[left] = 1; lab[right] = 2
    lobes = {n: np.asanyarray(nib.load(str(seg / "total" / f"{n}.nii.gz")).dataobj) > 0
             for n in ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
                       "lung_upper_lobe_left", "lung_lower_lobe_left"]}
    short = dict(zip(lobes, ["RUL", "RML", "RLL", "LUL", "LLL"]))
    return hu.astype(np.float32), ct, lab, lobes, short, seg


def test_measurements_respect_anatomy(tmp_path):
    hu, ct, lab, lobes, short, seg = _scan(tmp_path)
    rows, extra = measure_conditions(hu, ct, lab, lobes, short, seg, CK)
    r = {x["id"]: x for x in rows}
    assert extra["contrast"] == "non-contrast"
    assert r["E72"]["value"] > 0 and r["E71"]["value"] == 0          # effusion is on the patient's RIGHT
    loc = r["E82"]["value"]
    assert loc["side"] == "right" and loc["zone"] == "upper" and loc["leading_lobe"] == "RUL"
    assert r["E80"]["value"] > 0
    assert r["E90"]["value"] > 0                                     # the back-of-lung block is dependent
    assert r["E78"]["value"] > 0                                     # aortic calcium found
    assert r["E76"]["value"] == pytest.approx(32 / 128, abs=0.01)   # 16 vs 64 voxels wide


def test_dependent_band_is_posterior():
    lab = np.zeros((10, 20, 3), np.uint8); lab[2:8, 0:20, :] = 1      # RAS: y=0 back, y=19 front
    dep = dependent_band(lab, 0.2)
    assert dep[5, 0, 1] and dep[5, 3, 1] and not dep[5, 10, 1] and not dep[5, 19, 1]


def _ev(rows):
    return {"rows": rows}


def _row(i, v, **kw):
    return {"id": i, "group": kw.pop("group", "m"), "name": i, "value": v, "unit": "", "reliability": "x",
            "source": "s", **kw}


def test_decision_rules():
    rows = [_row("E70", 40, note="scan type: non-contrast"), _row("E71", 30.0), _row("E72", 0.0),
            _row("E73", 30.0), _row("E76", 0.48), _row("E30", 1.0), _row("E78", 0.5),
            _row("E81", 0.2), _row("E82", None), _row("E85", 2), _row("E87", None),
            _row("E50", 0.95, group="classifier", ckey="lung_nodule"),
            _row("E51", 0.9, group="classifier", ckey="bronchiectasis"),
            _row("E52", 0.1, group="classifier", ckey="hiatal_hernia"),
            _row("E53", 0.99, group="classifier", ckey="pleural_effusion")]
    readers = {"medgemma": {"conditions": {"bronchiectasis": {"status": "present"}}}, "hulumed": None}
    cl = build_checklist(_ev(rows), readers, CK)
    s = {i["key"]: i for i in cl["items"]}
    assert s["pleural_effusion"]["status"] == "present" and s["pleural_effusion"]["location"] == "left"
    assert s["pericardial_effusion"]["status"] == "cannot tell"       # grey zone 20-50 mL, no classifier
    assert s["cardiomegaly"]["status"] == "absent"
    assert s["lung_nodule"]["status"] == "cannot tell"                # classifier alone never makes 'present'
    assert s["bronchiectasis"]["status"] == "present"                 # classifier + reader agree
    assert s["hiatal_hernia"]["status"] == "absent"
    assert s["pulmonary_embolism"]["status"] == "cannot assess"
    assert s["lung_density"]["status"] == "absent"
    rep = rule_based_checklist_report(cl)
    assert not banned_hits(json.dumps(rep), CK)


def test_banned_words_rejected():
    rows = [_row("E70", 40, note="scan type: non-contrast"), _row("E71", 30.0), _row("E72", 0.0)]
    cl = build_checklist(_ev(rows), {}, CK)
    bad = {"impression": "Patchy opacities.", "findings": [{"text": "Pleural effusion present.", "evidence": ["E71"]}],
           "disagreements": [], "limitations": []}
    v = validate_checklist_report(bad, _ev(rows), cl, CK, validate_report)
    assert not v["valid"] and any("banned" in e for e in v["errors"])


CK2 = load_checklist_config(ROOT / "configs" / "checklist_v2.json")


def test_v2_readers_do_not_vote():
    rows = [_row("E70", 40, note="scan type: non-contrast"),
            _row("E51", 0.9, group="classifier", ckey="bronchiectasis"),
            _row("E52", 0.1, group="classifier", ckey="hiatal_hernia")]
    readers = {"medgemma": {"conditions": {"bronchiectasis": {"status": "present"},
                                           "hiatal_hernia": {"status": "present"}}}}
    s = {i["key"]: i for i in build_checklist(_ev(rows), readers, CK2)["items"]}
    assert s["bronchiectasis"]["status"] == "cannot tell"          # v1 would say present
    assert s["hiatal_hernia"]["status"] == "absent"
    assert any("do not vote" in f for f in s["hiatal_hernia"]["flags"])


def test_noise_smoothing(tmp_path):
    from lung_ct.agent.conditions import image_noise
    hu, ct, lab, lobes, short, seg = _scan(tmp_path)
    rng = np.random.default_rng(0)
    noisy = hu + rng.normal(0, 120, hu.shape).astype(np.float32)   # sharp-kernel-like grain
    out, rows, info = image_noise(noisy, ct, seg, CK2)
    assert rows[0]["id"] == "E69" and info["sd_hu"] > 100 and info["smoothed"]
    assert info["sd_after_hu"] < 0.75 * info["sd_hu"]   # 2 mm voxels: sigma 1 mm is only half a voxel
    out2, _, info2 = image_noise(hu, ct, seg, CK2)
    assert not info2["smoothed"] and out2 is hu                      # clean scan untouched


CK3 = load_checklist_config(ROOT / "configs" / "checklist_v3.json")


def test_v3_measurements(tmp_path):
    from lung_ct.agent.conditions import image_noise
    hu, ct, lab, lobes, short, seg = _scan(tmp_path)
    # dense lung touching the right-side effusion (posterior, high j) -> compressed lung, not consolidation
    hu[12:30, 40:46, 6:12] = 0
    (seg / "lung_vessels").mkdir(exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(SHAPE, np.uint8), AFF), str(seg / "lung_vessels" / "lung_airways.nii.gz"))
    out, nrows, noise = image_noise(hu, ct, seg, CK3)
    assert out is hu and noise["blocked"] == []                       # clean scan: nothing skipped, no smoothing
    rows, _ = measure_conditions(hu, ct, lab, lobes, short, seg, CK3, noise)
    r = {x["id"]: x for x in rows}
    assert r["E88"]["value"] > 0                                        # counted as collapsed lung
    assert r["E31"]["value"] is not None and "per_lobe" not in str(r["E31"]["value"])
    assert set(r["E32"]["value"]) >= {"RUL", "LLL"}
    assert "inflation" in r["E96"]["note"]


def test_v3_noise_blocks_not_smooths(tmp_path):
    from lung_ct.agent.conditions import blocked, image_noise
    hu, ct, lab, lobes, short, seg = _scan(tmp_path)
    noisy = hu + np.random.default_rng(1).normal(0, 52, hu.shape).astype(np.float32)
    out, _, noise = image_noise(noisy, ct, seg, CK3)
    assert out is noisy                                                 # never smoothed in v3
    assert blocked(noise, "emphysema") and blocked(noise, "calcium") and not blocked(noise, "consolidation")
    rows, _ = measure_conditions(noisy, ct, lab, lobes, short, seg, CK3, noise)
    r = {x["id"]: x for x in rows}
    assert r["E78"]["value"] is None and r["E80"]["value"] is None and r["E85"]["value"] is not None


def test_v3_rules():
    base = [_row("E70", 40, note="scan type: non-contrast"),
            _row("E55", 0.05, group="classifier", ckey="emphysema"),
            _row("E81", 5.0), _row("E82", {"side": "both", "right_share_pct": 50, "zone": "lower",
                                           "zone_shares_pct": {}, "leading_lobe": "RLL", "leading_lobe_share_pct": 40}),
            _row("E85", 0), _row("E87", None), _row("E88", 45)]
    good = base + [_row("E31", 1.0), _row("E96", -840, note="poor inflation if above -780 HU; inflation: adequate")]
    s = {i["key"]: i for i in build_checklist(_ev(good), {}, CK3)["items"]}
    assert s["emphysema"]["status"] == "cannot tell"     # low LAA is not proof, classifier alone cannot clear it
    assert s["atelectasis"]["status"] == "present"       # 45 mL next to fluid
    assert s["ground_glass"]["status"] == "present"
    poor = base + [_row("E31", 1.0), _row("E96", -740, note="poor inflation if above -780 HU; inflation: POOR")]
    s = {i["key"]: i for i in build_checklist(_ev(poor), {}, CK3)["items"]}
    assert s["ground_glass"]["status"] == "cannot tell" and "inflated" in s["ground_glass"]["basis"]
    high = base + [_row("E31", 9.0)]
    assert {i["key"]: i for i in build_checklist(_ev(high), {}, CK3)["items"]}["emphysema"]["status"] == "present"


def test_v3_1_heart_gate():
    ck = load_checklist_config(ROOT / "configs" / "checklist_v3_1.json")
    rows = [_row("E70", 40, note="scan type: non-contrast"), _row("E76", 0.62)]
    poor = rows + [_row("E96", -700, note="inflation: POOR")]
    good = rows + [_row("E96", -850, note="inflation: adequate")]
    get = lambda r: {i["key"]: i for i in build_checklist(_ev(r), {}, ck)["items"]}["cardiomegaly"]["status"]  # noqa: E731
    assert get(poor) == "cannot tell" and get(good) == "present"
    small = [_row("E70", 40, note="scan type: non-contrast"), _row("E76", 0.42), _row("E96", -700, note="inflation: POOR")]
    assert get(small) == "absent"            # a small ratio stays valid on a shallow breath


def test_v3_2_wording():
    ck = load_checklist_config(ROOT / "configs" / "checklist_v3_2.json")
    rows = [_row("E70", 40, note="scan type: non-contrast"), _row("E71", 0.0), _row("E72", 0.0),
            _row("E52", 0.1, group="classifier", ckey="hiatal_hernia")]
    cl = build_checklist(_ev(rows), {}, ck)
    s = {i["key"]: i for i in cl["items"]}
    assert s["pleural_effusion"]["status"] == "absent"                       # scoring unchanged
    assert s["pleural_effusion"]["display"] == "none measured"
    assert s["pleural_effusion"]["statement"].startswith("No measurable pleural effusion")
    assert s["hiatal_hernia"]["display"] == "not suggested"
    rep = rule_based_checklist_report(cl)
    assert "No measurable pleural effusion" in json.dumps(rep)
    bad = {"impression": "There is no pleural effusion.", "findings": [], "disagreements": [], "limitations": []}
    v = validate_checklist_report(bad, _ev(rows), cl, ck, validate_report)
    assert any("overclaims" in str(f) for f in v["flags"])
    ok = {"impression": "No measurable pleural effusion.", "findings": [], "disagreements": [], "limitations": []}
    v = validate_checklist_report(ok, _ev(rows), cl, ck, validate_report)
    assert not any("overclaims" in str(f) for f in v["flags"])


def test_v3_3_ground_glass_poor_inflation_limit():
    ck = load_checklist_config(ROOT / "configs" / "checklist_v3_3.json")
    old = load_checklist_config(ROOT / "configs" / "checklist_v3_2.json")

    def gg(pct, cfg, note="inflation: POOR"):
        rows = [_row("E70", 40, note="scan type: non-contrast"), _row("E81", pct), _row("E85", 0), _row("E96", -700, note=note)]
        return {i["key"]: i for i in build_checklist(_ev(rows), {}, cfg)["items"]}

    assert gg(39.9, old)["ground_glass"]["status"] == "cannot tell"          # v3_2: never present on poor inflation
    s = gg(39.9, ck)
    assert s["ground_glass"]["status"] == "present" and s["lung_density"]["status"] == "present"
    assert any("poorly inflated" in f for f in s["ground_glass"]["flags"])
    assert gg(24.0, ck)["ground_glass"]["status"] == "cannot tell"           # under the 30% limit: still gated
    assert gg(5.0, ck, note="inflation: adequate")["ground_glass"]["status"] == "present"
    heart = [_row("E70", 40, note="scan type: non-contrast"), _row("E76", 0.62), _row("E96", -700, note="inflation: POOR")]
    assert {i["key"]: i for i in build_checklist(_ev(heart), {}, ck)["items"]}["cardiomegaly"]["status"] == "cannot tell"


def test_pericardial_thickness_measure():
    import numpy as np
    from lung_ct.agent.conditions import greatest_thickness_mm, pericardial_thickness_row, thickness_grade
    aff = np.diag([0.7, 0.7, 1.5, 1.0])                       # anisotropic voxels: result must be in mm
    slab = np.zeros((80, 80, 40), bool)
    slab[10:70, 30:42, 5:35] = True                           # 12 voxels x 0.7 mm = 8.4 mm thick slab
    t = greatest_thickness_mm(slab, aff)
    assert 7.0 <= t <= 9.8
    assert greatest_thickness_mm(np.zeros((5, 5, 5), bool), aff) == 0.0 and greatest_thickness_mm(None, aff) is None
    assert [thickness_grade(x) for x in (0, 5, 10, 20, 25, None)] == ["no fluid outlined", "small", "moderate", "moderate", "large", None]
    ck = load_checklist_config(ROOT / "configs" / "checklist_v3_3.json")
    r = pericardial_thickness_row(slab, aff, ck)
    assert r["id"] == "E79" and r["unit"] == "mm" and "graded small" in r["note"] and "not used" in r["note"]
    assert pericardial_thickness_row(None, aff, ck)["value"] is None
    rows = [_row("E70", 40, note="scan type: non-contrast"), _row("E73", 9.5), _row("E79", 6.0)]
    it = {i["key"]: i for i in build_checklist(_ev(rows), {}, ck)["items"]}["pericardial_effusion"]
    assert it["status"] == "cannot tell" and "E79" in it["evidence"]      # low volume is never 'none' in v3_3


def test_v3_3_emphysema_low_value_wording():
    ck = load_checklist_config(ROOT / "configs" / "checklist_v3_3.json")
    old = load_checklist_config(ROOT / "configs" / "checklist_v3_2.json")

    def emph(pct, cfg):
        rows = [_row("E70", 40, note="scan type: non-contrast"), _row("E31", pct), _row("E96", -850, note="inflation: adequate")]
        return {i["key"]: i for i in build_checklist(_ev(rows), {}, cfg)["items"]}["emphysema"]

    new, was = emph(0.01, ck), emph(0.01, old)
    assert new["status"] == was["status"] == "cannot tell"                   # wording only: the answer does not change
    assert "cannot be ruled out" in new["statement"] and "0.01%" in new["statement"] and "from 6%" in new["statement"]
    assert was["statement"] == "Emphysema: cannot tell from this scan."
    assert emph(12.0, ck)["status"] == "present"
