"""checklist_v3_8: a positive mask-based pleural-effusion answer is headlined 'suggested by segmentation'; threshold result kept apart."""
import json
from pathlib import Path

from lung_ct.agent.checklist import build_checklist, load_checklist_config, rule_based_checklist_report
from test_v3_6 import _ev36

ROOT = Path(__file__).resolve().parent.parent
V37 = load_checklist_config(ROOT / "configs" / "checklist_v3_7.json")
V38 = load_checklist_config(ROOT / "configs" / "checklist_v3_8.json")
CASES = ((0, 0), (1.5, 1.5), (10.0, 8.4), (40.0, 10.0))


def _pe(lr, ck):
    return next(i for i in build_checklist(_ev36(0.0, 0.0, lr), {}, ck)["items"] if i["key"] == "pleural_effusion")


def test_v3_8_changes_no_limit_measurement_or_status():
    assert V38["measurements"] == V37["measurements"]
    a, b = {c["key"]: c for c in V37["conditions"]}, {c["key"]: c for c in V38["conditions"]}
    for k in a:
        for f in ("present_at_least", "absent_below", "measured_by", "classifier_key"):
            assert a[k].get(f) == b[k].get(f), (k, f)
        if k != "pleural_effusion":
            assert a[k] == b[k], k
    for lr in CASES:
        x = {i["key"]: i for i in build_checklist(_ev36(0.0, 0.0, lr), {}, V37)["items"]}
        y = {i["key"]: i for i in build_checklist(_ev36(0.0, 0.0, lr), {}, V38)["items"]}
        assert [x[k]["status"] for k in x] == [y[k]["status"] for k in x]
        for k in x:
            if k != "pleural_effusion":
                assert json.dumps(x[k], sort_keys=True) == json.dumps(y[k], sort_keys=True), k


def test_positive_mask_based_answer_is_a_suggestion_and_the_threshold_result_is_separate():
    empty, low, mid, big = (_pe(v, V38) for v in CASES)
    assert big["headline"] == "Pleural effusion suggested by segmentation" and big["display"] == "suggested by segmentation"
    assert big["statement"].startswith("Pleural effusion suggested by segmentation") and "Not confirmed" in big["statement"]
    assert "present" not in big["statement"].lower() and big["decision_basis"] == "predicted mask only"
    assert [i["provisional_threshold_result"]["answer"] for i in (empty, low, mid, big)] == ["none", "none", "undecided", "present"]
    t = big["provisional_threshold_result"]
    assert (t["value"], t["present_at_least"], t["absent_below"]) == (50.0, 20, 5) and "not validated diagnostic" in t["note"]
    for it in (empty, low, mid):
        assert "headline" not in it and it["status"] == "cannot tell"
    assert "headline" not in _pe(CASES[3], V37) and _pe(CASES[3], V37)["display"] == "present"          # v3_7 untouched


def test_report_lists_it_as_suggested_not_as_a_confirmed_finding():
    rep = rule_based_checklist_report(build_checklist(_ev36(0.0, 0.0, (40.0, 10.0)), {}, V38))
    assert any(f["text"].startswith("Suggested by segmentation (not confirmed)") for f in rep["findings"])
    assert not any(f["text"].startswith("Pleural effusion: present") for f in rep["findings"])
    assert "Suggested by segmentation, not confirmed: pleural effusion" in rep["impression"]
    assert "Findings: pleural effusion" not in rep["impression"]


def test_headline_and_provisional_meaning_survive_prompt_validation_export_and_scoring():
    import csv
    import io
    from lung_ct.agent.checklist import (EXPORT_COLUMNS, assessment_category, build_checklist_prompt, export_row,
                                         validate_checklist_report)
    base = lambda o, e: {"errors": [], "flags": [], "valid": True}  # noqa: E731
    for lr, want in zip(CASES, ("cannot tell", "cannot tell", "cannot tell", "suggested")):
        ev = _ev36(0.0, 0.0, lr)
        cl = build_checklist(ev, {}, V38)
        it = next(i for i in cl["items"] if i["key"] == "pleural_effusion")
        assert assessment_category(it) == want                       # a historical 'none' never becomes 'absent'
        row = export_row("x", cl["config_version"], it)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS)
        w.writeheader()
        w.writerow(row)
        back = next(csv.DictReader(io.StringIO(buf.getvalue())))
        assert back["current_assessment"] == want and back["current_assessment"] not in ("absent", "none", "present")
        assert back["provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT"] == it["provisional_threshold_result"]["answer"]
        assert "provisional, not validated" in back["provisional_limits"] and back["checklist_version"] == "checklist_v3_8"
        prompt = build_checklist_prompt(cl, ev, V38)
        line = next(x for x in prompt.splitlines() if x.startswith("- Pleural effusion"))
        assert it["statement"] in line and "PRESENT" not in line.split("|")[0]
        assert "is NOT marked present" in prompt and "never write that it is absent" in prompt
    cl = build_checklist(_ev36(0.0, 0.0, CASES[3]), {}, V38)
    v = validate_checklist_report({"impression": "x", "findings": [{"text": "Pleural effusion on the left.", "evidence": []}]}, {}, cl, V38, base)
    assert any("suggested by segmentation" in f["flag"] for f in v["flags"])
    v = validate_checklist_report({"impression": "x", "findings": [{"text": "Pleural effusion suggested by segmentation, left.", "evidence": []}]}, {}, cl, V38, base)
    assert not any("suggested by segmentation" in f["flag"] for f in v["flags"])
    cl0 = build_checklist(_ev36(0.0, 0.0, CASES[0]), {}, V38)
    v = validate_checklist_report({"impression": "There is no pleural effusion.", "findings": []}, {}, cl0, V38, base)
    assert any("overclaims absence of pleural effusion" in f["flag"] for f in v["flags"])
    # older configs: prompt carries no new instruction
    assert "is NOT marked present" not in build_checklist_prompt(build_checklist(_ev36(0.0, 0.0, CASES[3]), {}, V37), _ev36(0.0, 0.0, CASES[3]), V37)
