"""checklist_v3_7: a predicted pleural-fluid mask below the reporting limit is 'cannot tell'; limits are provisional, unchanged."""
import json
from pathlib import Path

from lung_ct.agent.checklist import build_checklist, load_checklist_config, rule_based_checklist_report
from test_v3_6 import _ev36

ROOT = Path(__file__).resolve().parent.parent
V36 = load_checklist_config(ROOT / "configs" / "checklist_v3_6.json")
V37 = load_checklist_config(ROOT / "configs" / "checklist_v3_7.json")
CASES = ((0, 0), (1.5, 1.5), (10.0, 8.4), (40.0, 10.0))


def _pe(lr, ck):
    return next(i for i in build_checklist(_ev36(0.0, 0.0, lr), {}, ck)["items"] if i["key"] == "pleural_effusion")


def test_v3_7_changes_no_limit_and_no_measurement():
    assert V37["measurements"] == V36["measurements"]
    a, b = {c["key"]: c for c in V36["conditions"]}, {c["key"]: c for c in V37["conditions"]}
    assert a.keys() == b.keys()
    for k in a:
        for f in ("present_at_least", "absent_below", "poor_inflation", "poor_inflation_present_at_least", "allow_classifier_absent", "classifier_key", "measured_by"):
            assert a[k].get(f) == b[k].get(f), (k, f)
        if k != "pleural_effusion":
            assert a[k] == b[k], k
    assert "not been validated as a diagnostic boundary" in V37["limits_note"] or "validated as a diagnostic boundary" in V37["limits_note"]


def test_predicted_mask_below_the_limit_is_cannot_tell_and_the_historical_answer_is_kept():
    new = [_pe(v, V37) for v in CASES]
    old = [_pe(v, V36) for v in CASES]
    assert [i["status"] for i in old] == ["absent", "absent", "cannot tell", "present"]                    # v3_6 untouched
    assert [i["status"] for i in new] == ["cannot tell", "cannot tell", "cannot tell", "present"]
    assert [i["historical_rule_answer"] for i in new] == ["none", "none", "undecided", "present"]            # unchanged 20 / 5 mL rule
    assert [i["mask_state"] for i in new] == [i["mask_state"] for i in old]
    assert all("historical_rule_answer" not in i for i in old)
    assert [i.get("measured_value") for i in new] == [i.get("measured_value") for i in old]
    empty, low, mid, big = new
    assert empty["display"] == "cannot tell (no fluid outlined)" and "does not exclude" in empty["statement"]
    assert low["display"] == "cannot tell (low volume outlined)" and "neither confirms nor excludes" in low["statement"]
    assert "cannot establish absence" in empty["basis"] and "not validated" in empty["basis"]
    for it in (low, mid, big):
        assert "provisional" in it["statement"] and "estimate" in it["statement"]
    assert "not a validated diagnostic boundary" in big["statement"]
    for it in new:
        assert "No measurable" not in it["statement"] and "none measured" not in it.get("display", "")


def test_only_pleural_effusion_differs_from_v3_6():
    for lr in CASES:
        a = {i["key"]: i for i in build_checklist(_ev36(0.0, 0.0, lr), {}, V36)["items"]}
        b = {i["key"]: i for i in build_checklist(_ev36(0.0, 0.0, lr), {}, V37)["items"]}
        for k in a:
            if k != "pleural_effusion":
                assert json.dumps(a[k], sort_keys=True) == json.dumps(b[k], sort_keys=True), k


def test_report_never_lists_pleural_effusion_as_not_detected():
    rep = rule_based_checklist_report(build_checklist(_ev36(0.0, 0.0), {}, V37))
    txt = " ".join(f["text"] for f in rep["findings"])
    assert "does not exclude" in txt and "Pleural effusion" in txt
    assert not any(f["text"].startswith("Not detected by measurement") and "Pleural effusion" in f["text"] for f in rep["findings"])
