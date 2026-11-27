"""checklist_v3_11: AI reader NV-Reason-CT decides 'present' for medical material, hiatal hernia and bronchiectasis (never 'absent')."""
from pathlib import Path

import pytest

from lung_ct.agent.checklist import assessment_category, build_checklist, export_row, load_checklist_config
from lung_ct.vlm.checklist_schema import READER_CONDITIONS
from test_v3_6 import _ev36

ROOT = Path(__file__).resolve().parent.parent
V310 = load_checklist_config(ROOT / "configs" / "checklist_v3_10.json")
V311 = load_checklist_config(ROOT / "configs" / "checklist_v3_11.json")
R = "NV-Reason-CT"
KEYS = ["medical_material", "hiatal_hernia", "bronchiectasis"]


def _reader(present=(), score=None, name=R):
    return {name: {"assessability": "assessable", "limitations": [],
                   "conditions": {k: {"status": "present" if k in present else "absent", "side": "not_applicable",
                                      **({"score_yes": score} if score is not None else {})} for k in READER_CONDITIONS}}}


def _items(ck, readers):
    return {i["key"]: i for i in build_checklist(_ev36(0.0, 0.0), readers, ck)["items"]}


def test_only_the_three_conditions_differ_from_v3_10():
    a, b = {c["key"]: c for c in V310["conditions"]}, {c["key"]: c for c in V311["conditions"]}
    assert list(a) == list(b) and V310["measurements"] == V311["measurements"] and V311["readers_vote"] is False
    for k in a:
        if k in KEYS:
            assert b[k] == {**a[k], "reader_decides_present": R}
        else:
            assert a[k] == b[k], k
    assert V311["trusted_reader"]["conditions"] == KEYS and V311["trusted_reader"]["reader"] == R
    assert V311["config_version"] == "checklist_v3_11" and V311["derived_from"] == "checklist_v3_10"


@pytest.mark.parametrize("key", KEYS)
def test_reader_present_gives_present_with_the_source_named(key):
    it = _items(V311, _reader({key}, 0.97))[key]
    assert it["status"] == "present" and assessment_category(it) == "present" and it["display"] == "present"
    assert it["reader_decided"] == {"reader": R, "answer": "present", "score_yes": 0.97, "side": "not_applicable"}
    assert f"AI reader {R} answers present, score 0.97 (0 to 1, uncalibrated)" in it["basis"] and "not validated" in it["basis"]
    assert it["statement"] == f"{it['label']}: present, according to AI reader {R} (not measured by the tool)."
    assert it["decision_basis"] == f"AI reader {R}"
    assert export_row("c", "checklist_v3_11", it)["decision_basis"] == f"AI reader {R}"


@pytest.mark.parametrize("key", KEYS)
def test_reader_absent_never_gives_absent(key):
    it = _items(V311, _reader(set(), 0.0))[key]
    assert it["status"] == "cannot tell" and "reader_decided" not in it
    assert f"AI reader {R} answers absent, score 0.00 (0 to 1, uncalibrated); a reader's 'absent' cannot establish absence" in it["basis"]


@pytest.mark.parametrize("key", KEYS)
def test_without_that_reader_the_answer_is_as_before(key):
    for readers in ({}, _reader({key}, 0.99, name="medgemma")):          # no reader, or another reader saying present
        it = _items(V311, readers)[key]
        assert it["status"] == "cannot tell" and "reader_decided" not in it and f"no answer from AI reader {R}" in it["basis"]
    assert _items(V310, _reader({key}, 0.99))[key]["status"] == "cannot tell"      # v3_10 itself is not changed by the new code


def test_no_other_condition_is_changed_by_the_reader():
    everything = _reader(set(READER_CONDITIONS), 1.0)
    with_r, without = _items(V311, everything), _items(V311, {})
    for k in with_r:
        if k not in KEYS:
            assert with_r[k]["status"] == without[k]["status"], k
            assert "reader_decided" not in with_r[k]
    assert all(with_r[k]["status"] == "present" for k in KEYS)


def test_score_is_not_used_as_a_limit_and_may_be_missing():
    low = _items(V311, _reader({"hiatal_hernia"}, 0.51))["hiatal_hernia"]
    none = _items(V311, _reader({"hiatal_hernia"}))["hiatal_hernia"]
    assert low["status"] == none["status"] == "present" and none["reader_decided"]["score_yes"] is None and "score" not in none["basis"]


def test_v3_10_answers_are_identical_with_the_new_code_when_no_reader_runs():
    a = build_checklist(_ev36(0.0, 0.0), {}, V310)
    assert all("reader_decided" not in i for i in a["items"])
    assert {i["key"]: i["basis"] for i in a["items"]}["hiatal_hernia"].endswith("no reader answer")
