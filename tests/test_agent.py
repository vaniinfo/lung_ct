"""Cross-check and report-validation logic (no images needed)."""
import json
from pathlib import Path

from lung_ct.agent.crosscheck import crosscheck, description_field_conflicts
from lung_ct.agent.report import build_prompt, rule_based_report, validate_report

CFG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "agent_evidence_v1.json").read_text())


def _ev(r_share=93):
    rows = [{"id": "E10", "group": "dense lung", "name": "total", "value": 102, "unit": "mL", "reliability": "low", "source": "x"},
            {"id": "E11", "group": "dense lung", "name": "left", "value": 7, "unit": "mL", "reliability": "low", "source": "x"},
            {"id": "E12", "group": "dense lung", "name": "right", "value": 95, "unit": "mL", "reliability": "low", "source": "x"},
            {"id": "E13", "group": "distribution", "name": "right share", "value": r_share, "unit": "%", "reliability": "moderate", "source": "x"},
            {"id": "E14", "group": "distribution", "name": "zones", "value": {"lower": 60, "middle": 30, "upper": 10}, "unit": "%", "reliability": "moderate", "source": "x"},
            {"id": "E15", "group": "distribution", "name": "periph", "value": 72, "unit": "%", "reliability": "moderate", "source": "x"},
            {"id": "E25", "group": "severity", "name": "score", "value": 4, "unit": "points", "reliability": "low", "source": "x"},
            {"id": "E30", "group": "emphysema", "name": "laa", "value": 1.2, "unit": "%", "reliability": "moderate", "source": "x"}]
    return {"rows": rows, "derived_answers": {"distribution": {"laterality": "bilateral", "craniocaudal": "lower_predominant", "axial": "peripheral"}}}


def _reader(lat="bilateral", cc="lower_predominant", ax="peripheral", desc="Patchy opacities.", pres="present"):
    return {"opacity_presence": pres, "distribution": {"laterality": lat, "craniocaudal": cc, "axial": ax}, "description": desc}


def test_bilateral_flagged_when_one_side_dominates():
    out = crosscheck(_ev(93), {"m": _reader()}, CFG)
    lat = next(i for i in out["items"] if i["field"] == "laterality")
    assert lat["status"] == "conflict" and "right lung" in lat["detail"]
    assert out["decisions"]["laterality"]["note"].startswith("bilateral, strongly right")


def test_agreement_and_conflict_and_abstain():
    out = crosscheck(_ev(55), {"a": _reader(cc="upper_predominant", ax="indeterminate")}, CFG)
    s = {i["field"]: i["status"] for i in out["items"] if i["reader"] == "a"}
    assert s == {"laterality": "agree", "craniocaudal": "conflict", "axial": "reader_abstained"}


def test_absent_reader_and_reader_disagreement():
    out = crosscheck(_ev(55), {"a": _reader(), "b": _reader(pres="absent", lat="not_applicable")}, CFG)
    assert any(i["field"] == "opacity_presence" and i["status"] == "conflict" for i in out["items"])
    assert any(i["status"] == "readers_disagree" for i in out["items"])


def test_self_consistency():
    msgs = description_field_conflicts(_reader(cc="no_predominance", desc="Opacities, predominantly in the lower lobes."))
    assert msgs and "lower" in msgs[0]


def test_validate_report_flags():
    ev = _ev()
    good = {"impression": "x", "findings": [{"text": "Right lung 95 mL.", "evidence": ["E12"]}], "disagreements": [], "limitations": []}
    assert validate_report(good, ev) == {"valid": True, "errors": [], "flags": []}
    bad = {"impression": "x", "findings": [{"text": "Left lung 40 mL.", "evidence": ["E11", "E99"]},
                                          {"text": "Nodule seen.", "evidence": []}], "disagreements": [], "limitations": []}
    v = validate_report(bad, ev)
    kinds = " ".join(f["flag"] for f in v["flags"])
    assert "unknown ids" in kinds and "numbers not found" in kinds and "no evidence" in kinds


def test_rule_based_report_validates_and_prompt_builds():
    ev = _ev()
    cc = crosscheck(ev, {"m": _reader()}, CFG)
    rep = rule_based_report(ev, cc, {"m": _reader()})
    assert validate_report(rep, ev)["valid"]
    assert "[E12]" in build_prompt(ev, cc, {"m": _reader()})


def _ev_cl():
    ev = _ev()
    ev["classifier"] = {"model": "colipri@x", "high_threshold": 0.7, "low_threshold": 0.3}
    for i, (lab, p) in enumerate([("Consolidation", 0.1), ("Ground-glass opacity", 0.9), ("Pleural effusion", 0.85)]):
        ev["rows"].append({"id": f"E{50 + i}", "group": "classifier", "name": f"{lab} (classifier probability)",
                           "value": p, "unit": "p", "reliability": "uncalibrated", "source": "x"})
    return ev


def test_what_checks():
    ev = _ev_cl()
    out = crosscheck(ev, {"m": _reader(desc="Ground-glass opacities and consolidation. No pleural effusion.")}, CFG)
    f = {i["field"]: i["status"] for i in out["items"] if i["field"].startswith("finding")}
    assert f["finding: Consolidation"] == "conflict"
    assert f["finding: Ground-glass opacity"] == "agree"
    assert "finding: Pleural effusion" not in f or f["finding: Pleural effusion"] != "agree"
    rep = rule_based_report(ev, out, {"m": _reader()})
    assert validate_report(rep, ev)["valid"] and any("E51" in f["evidence"] for f in rep["findings"])
