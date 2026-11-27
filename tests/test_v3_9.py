"""checklist_v3_9: the atelectasis rule of v3_2 to v3_5 is restored (project-owner decision); nothing else changes."""
from pathlib import Path

from lung_ct.agent.checklist import build_checklist, load_checklist_config
from test_v3_6 import _ev36

ROOT = Path(__file__).resolve().parent.parent
V35 = load_checklist_config(ROOT / "configs" / "checklist_v3_5.json")
V38 = load_checklist_config(ROOT / "configs" / "checklist_v3_8.json")
V39 = load_checklist_config(ROOT / "configs" / "checklist_v3_9.json")


def _items(ck, cons_ml, e88, pleural=(0.0, 0.0)):
    return {i["key"]: i for i in build_checklist(_ev36(cons_ml, e88, pleural), {}, ck)["items"]}


def test_only_the_atelectasis_condition_and_its_candidate_paragraph_differ_from_v3_8():
    a, b = {c["key"]: c for c in V38["conditions"]}, {c["key"]: c for c in V39["conditions"]}
    assert list(a) == list(b)
    for k in a:
        if k != "atelectasis":
            assert a[k] == b[k], k
    m8, m9 = dict(V38["measurements"]), dict(V39["measurements"])
    i8, i9 = dict(m8.pop("dense_lung_inventory")), dict(m9.pop("dense_lung_inventory"))
    assert m8 == m9
    assert i8.pop("attach_to") == ["consolidation", "atelectasis", "lung_density"] and i9.pop("attach_to") == ["consolidation", "lung_density"]
    assert i8 == i9


def test_atelectasis_limit_is_the_one_used_up_to_v3_5():
    old = next(c for c in V35["conditions"] if c["key"] == "atelectasis")
    new = next(c for c in V39["conditions"] if c["key"] == "atelectasis")
    for f in ("measured_by", "present_at_least", "absent_below", "unit"):
        assert old[f] == new[f], f
    assert new["present_at_least"] == 30 and new["absent_below"] is None


def test_present_from_30_ml_and_never_absent():
    assert _items(V39, 0.0, 30.0, (200.0, 0.0))["atelectasis"]["status"] == "present"
    assert _items(V39, 0.0, 246.0, (200.0, 60.0))["atelectasis"]["status"] == "present"
    for e88 in (0.0, 5.0, 29.9):
        assert _items(V39, 0.0, e88, (200.0, 0.0))["atelectasis"]["status"] == "cannot tell", e88
    assert _items(V38, 0.0, 246.0, (200.0, 60.0))["atelectasis"]["status"] == "cannot tell"      # v3_8 is unchanged


def test_present_answer_does_not_carry_the_not_determined_paragraph():
    it = _items(V39, 0.0, 246.0, (200.0, 60.0))["atelectasis"]
    assert "not determined by the tool" not in it["statement"] and "dense_lung_candidates" not in it
    assert "E88" in it["evidence"]


def test_no_other_answer_moves_with_the_restored_rule():
    for cons, e88, pl in ((0.0, 0.0, (0.0, 0.0)), (50.0, 246.0, (200.0, 60.0)), (10.0, 29.0, (18.0, 0.0)), (134.0, 981.0, (900.0, 900.0))):
        x, y = _items(V38, cons, e88, pl), _items(V39, cons, e88, pl)
        for k in x:
            if k != "atelectasis":
                assert (x[k]["status"], x[k].get("display"), x[k].get("statement")) == (y[k]["status"], y[k].get("display"), y[k].get("statement")), k
