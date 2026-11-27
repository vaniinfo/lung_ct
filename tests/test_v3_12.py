"""checklist_v3_12: no 'suggested by segmentation'; two-source 'present' (NV-Reason-CT + classifier / fluid volume) for three conditions;
NV-Reason-CT alone no longer decides medical material, hiatal hernia, bronchiectasis."""
from pathlib import Path

import pytest

from lung_ct.agent.checklist import assessment_category, build_checklist, load_checklist_config
from test_v3_6 import _ev36
from test_v3_11 import _reader

ROOT = Path(__file__).resolve().parent.parent
V311 = load_checklist_config(ROOT / "configs" / "checklist_v3_11.json")
V312 = load_checklist_config(ROOT / "configs" / "checklist_v3_12.json")
R = "NV-Reason-CT"
TWO = {"pleural_effusion": 0.8197, "arterial_calcification": 0.0620, "coronary_calcification": 0.1346}


def _ev(pleural=(0.0, 0.0), scores=None):
    e = _ev36(0.0, 0.0, pleural=pleural)
    for i, (k, p) in enumerate((scores or {}).items()):
        e["rows"] = [r for r in e["rows"] if not (r.get("group") == "classifier" and r.get("ckey") == k)]
        e["rows"].append({"id": f"C{i}", "group": "classifier", "ckey": k, "value": p, "name": k})
    return e


def _items(ev, readers, ck=V312):
    return {i["key"]: i for i in build_checklist(ev, readers, ck)["items"]}


def test_config_changes_only_what_v3_12_says():
    a, b = {c["key"]: c for c in V311["conditions"]}, {c["key"]: c for c in V312["conditions"]}
    assert list(a) == list(b) and V312["two_source_rule"] is True and "trusted_reader" not in V312
    assert not any(c.get("reader_decides_present") for c in V312["conditions"])
    assert {k: c["two_source_present"]["classifier_above"] for k, c in b.items() if c.get("two_source_present")} == TWO
    assert b["pleural_effusion"]["two_source_present"]["measured_at_least"] == 20
    for k in a:
        strip = lambda d: {x: y for x, y in d.items() if x not in ("reader_decides_present", "two_source_present", "mask_wording", "note")}  # noqa: E731  (notes: wording only)
        assert strip(a[k]) == strip(b[k]), k


@pytest.mark.parametrize("key", ["medical_material", "hiatal_hernia", "bronchiectasis"])
def test_reader_alone_no_longer_decides(key):
    assert _items(_ev(), _reader({key}, 0.99), V311)[key]["status"] == "present"
    it = _items(_ev(), _reader({key}, 0.99))[key]
    assert it["status"] == "cannot tell" and "reader_decided" not in it


@pytest.mark.parametrize("key", ["arterial_calcification", "coronary_calcification", "pleural_effusion"])
def test_two_sources_needed_for_present(key):
    cut = TWO[key]
    above, below = {key: cut + 0.01}, {key: cut - 0.01}
    it = _items(_ev(scores=above), _reader({key}))[key]
    if it["status"] != "present":                       # a decisive measurement (e.g. aortic calcium 0 mL) is never overruled
        assert it["basis"].startswith("measured") and "two_source" not in it
        return
    assert it["two_source"]["reader"] == R and "classifier COLIPRI scores" in it["statement"] and it["display"] == "present"
    assert it["decision_basis"] == "two sources agree" and assessment_category(it) == "present" and it["reader_decided"]["with"]
    assert _items(_ev(scores=below), _reader({key}))[key]["status"] == "cannot tell"          # reader yes, classifier below the cut-off
    assert _items(_ev(scores=above), _reader(set()))[key]["status"] != "present"              # classifier high, reader no
    assert _items(_ev(scores=above), {})[key]["status"] != "present"                          # classifier alone


def test_pleural_mask_alone_is_cannot_tell_with_the_volume_listed_and_mask_plus_reader_is_present():
    ev = _ev(pleural=(40.0, 0.0), scores={"pleural_effusion": 0.1})
    it = _items(ev, {})["pleural_effusion"]
    assert it["status"] == "cannot tell" and it["display"] == "cannot tell" and "40 mL" in it["statement"] and "suggested" not in it["statement"].lower()
    assert "headline" not in it and assessment_category(it) == "cannot tell"
    it = _items(ev, _reader({"pleural_effusion"}))["pleural_effusion"]
    assert it["status"] == "present" and "outlined 40 mL" in it["statement"] and "headline" not in it
    assert _items(_ev(pleural=(10.0, 0.0), scores={"pleural_effusion": 0.1}), _reader({"pleural_effusion"}))["pleural_effusion"]["status"] == "cannot tell"


def test_no_suggested_by_segmentation_anywhere():
    for pleural in ((0.0, 0.0), (40.0, 30.0)):
        for readers in ({}, _reader(set(TWO))):
            for it in _items(_ev(pleural=pleural), readers).values():
                assert "suggested by segmentation" not in (it.get("display", "") + it.get("statement", "") + it.get("headline", "")).lower()
