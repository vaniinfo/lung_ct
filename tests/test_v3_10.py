"""checklist_v3_10: lung nodule answer from the candidates of the lung_nodules segmentation task (suggested, never present / absent)."""
from pathlib import Path

from lung_ct.agent.checklist import assessment_category, build_checklist, export_row, load_checklist_config
from test_v3_6 import _ev36

ROOT = Path(__file__).resolve().parent.parent
V39 = load_checklist_config(ROOT / "configs" / "checklist_v3_9.json")
V310 = load_checklist_config(ROOT / "configs" / "checklist_v3_10.json")


def _ev(n, where="RUL, LUL"):
    e = _ev36(0.0, 0.0)
    if n is not None:
        e["rows"] += [{"id": "E97", "group": "nodule candidates", "name": "count", "value": n, "unit": "candidates"},
                      {"id": "E98", "group": "nodule candidates", "name": "list", "value": "x", "unit": None, "where": where if n else None}]
    return e


def _nod(ck, n, **k):
    return next(i for i in build_checklist(_ev(n, **k), {}, ck)["items"] if i["key"] == "lung_nodule")


def test_only_the_lung_nodule_condition_differs_from_v3_9():
    a, b = {c["key"]: c for c in V39["conditions"]}, {c["key"]: c for c in V310["conditions"]}
    assert list(a) == list(b) and V39["measurements"] == V310["measurements"]
    for k in a:
        if k != "lung_nodule":
            assert a[k] == b[k], k


def test_candidates_are_a_suggestion_never_a_confirmed_present():
    for n in (1, 6):
        it = _nod(V310, n)
        assert it["status"] == "present" and assessment_category(it) == "suggested"
        assert it["headline"] == "Lung nodule suggested by segmentation" and it["display"] == "suggested by segmentation"
        assert it["decision_basis"] == "predicted mask only" and "RUL, LUL" in it["statement"] and "Not confirmed" in it["statement"]
        assert "present >=" not in it["basis"] and "candidate outline" in it["basis"]
        assert export_row("c", "checklist_v3_10", it)["current_assessment"] == "suggested"


def test_no_candidate_is_cannot_tell_never_absent():
    it = _nod(V310, 0)
    assert it["status"] == "cannot tell" and it["display"] == "cannot tell (no candidate outlined)"
    assert "does not exclude a nodule" in it["statement"] and "cannot establish absence" in it["basis"]
    assert export_row("c", "checklist_v3_10", it)["current_assessment"] == "cannot tell"


def test_without_the_candidate_rows_the_answer_is_as_before():
    a, b = _nod(V39, None), _nod(V310, None)
    assert (a["status"], a["basis"], a["statement"]) == (b["status"], b["basis"], b["statement"]) and b["status"] == "cannot tell"


def test_rows_do_not_change_v3_9_or_any_other_v3_10_answer():
    assert _nod(V39, 6)["status"] == "cannot tell"          # the earlier policy ignores the candidate rows
    x = {i["key"]: i for i in build_checklist(_ev(6), {}, V39)["items"]}
    y = {i["key"]: i for i in build_checklist(_ev(6), {}, V310)["items"]}
    for k in x:
        if k != "lung_nodule":
            assert (x[k]["status"], x[k].get("display"), x[k].get("statement")) == (y[k]["status"], y[k].get("display"), y[k].get("statement")), k
