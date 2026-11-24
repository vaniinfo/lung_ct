"""Finding-mask audit: statement parser, primary checks, run matching, sampling manifest, review schema."""
import pytest

from lung_ct.audit.checks import (THRESHOLDS, audit_finding, case_level_answer_gaps, check_contradictory_output,
                                  check_empty_mask, check_lobe, check_side, describe_attenuation)
from lung_ct.audit.matching import compare_answers, match_findings
from lung_ct.audit.parse import parse_statement
from lung_ct.audit.sampling import build_manifest
from lung_ct.audit.schema import finding_id, patient_id, review_status, validate_stage1


def shares(**kw):
    s = {"RUL": 0.0, "RML": 0.0, "RLL": 0.0, "LUL": 0.0, "LLL": 0.0}
    s.update(kw)
    assert abs(sum(s.values()) - 1.0) < 1e-9
    return s


# ---------------- parser: laterality and lobes ----------------
@pytest.mark.parametrize("text,side,lobes", [
    ("Consolidation in the right lower lobe", "right", ["RLL"]),
    ("Subcentimeter calcific nodule in the upper lobe of the right lung", "right", ["RUL"]),
    ("Scarring in the posterobasal level of the left lower lobe", "left", ["LLL"]),
    ("Ground glass density in the lower lobe of the left lung", "left", ["LLL"]),
    ("Nodule in the lingula", "left", ["LUL"]),
    ("Atelectasis in the middle lobe", "right", ["RML"]),
    ("Bilateral ground glass opacities within the lung parenchyma", "bilateral", []),
    ("Emphysema in both lungs", "bilateral", []),
    ("Consolidation in both lower lobes", "bilateral", ["RLL", "LLL"]),
    ("Nodules in the right upper lobe and left lower lobe", "bilateral", ["RUL", "LLL"]),
    ("Emphysema in the lower lobes of both lungs", "bilateral", ["RLL", "LLL"]),
    ("Pleural effusion on the left", "left", []),
    ("Areas of consolidation within the lung parenchyma", None, []),
])
def test_side_and_lobes(text, side, lobes):
    p = parse_statement(text)
    assert p["side"] == side
    assert p["lobes"] == lobes


def test_both_alone_is_not_bilateral():
    p = parse_statement("Nodules in both the upper and lower lobes of the right lung")
    assert p["side"] == "right"


def test_side_comparison_gives_no_single_side():
    p = parse_statement("More pronounced ground-glass opacities in the right lung compared to the left")
    assert p["side"] is None and p["side_comparison"] and not p["location_checkable"]


def test_lobe_without_side_is_marked():
    p = parse_statement("Nodule in the lower lobe")
    assert p["lobe_side_missing"] and p["lobes"] == [] and p["side"] is None


# ---------------- parser: negation, uncertainty, several findings ----------------
@pytest.mark.parametrize("text", [
    "No pleural effusion", "No evidence of consolidation in the right lower lobe", "Lungs without nodules",
    "Pleural effusion has resolved", "Absence of emphysema", "Consolidation is not seen in the left lung",
])
def test_negation(text):
    p = parse_statement(text)
    assert p["negated"] and not p["location_checkable"]


@pytest.mark.parametrize("text", ["Nonspecific subcentimeter nodules in the left lung", "Nodule in the right upper lobe"])
def test_not_negated(text):
    assert not parse_statement(text)["negated"]


@pytest.mark.parametrize("text", [
    "Possible consolidation in the right lower lobe", "Probable atelectasis in the left lower lobe",
    "Nodule in the right upper lobe, malignancy cannot be excluded", "Suspicious nodule in the left upper lobe",
])
def test_uncertainty_is_recorded_but_location_stays_checkable(text):
    p = parse_statement(text)
    assert p["uncertain"] and p["location_checkable"]


def test_qualifiers_are_not_uncertainty():
    assert not parse_statement("Nonspecific nodules measuring 5 mm in the right upper lobe")["uncertain"]


@pytest.mark.parametrize("text", [
    "Pleural effusion and adjacent atelectasis in the right lower lobe",
    "Consolidation in the right lower lobe; emphysema in the left upper lobe",
    "Ground glass opacities and nodules in the right upper lobe",
])
def test_several_findings_switch_location_checks_off(text):
    p = parse_statement(text)
    assert p["multiple_findings"] and not p["location_checkable"]
    assert "several findings in one statement" in p["why_not_checkable"]


@pytest.mark.parametrize("text", [
    "Subcentimeter calcific nodule in the upper lobe of the right lung",
    "Consolidation area containing air bronchograms in the lower lobe of the left lung",
])
def test_descriptor_words_are_not_a_second_finding(text):
    assert not parse_statement(text)["multiple_findings"]


def test_empty_statement():
    p = parse_statement(None)
    assert not p["location_checkable"] and p["types"] == []


# ---------------- checks ----------------
def test_empty_mask_flag():
    assert check_empty_mask(0)["status"] == "flag" and check_empty_mask(5)["status"] == "pass"


def test_side_flag_when_outline_on_other_side():
    p = parse_statement("Consolidation in the right lower lobe")
    assert check_side(p, 100, shares(LLL=0.9, RLL=0.1), 1.0)["status"] == "flag"
    assert check_side(p, 100, shares(RLL=0.9, LLL=0.1), 1.0)["status"] == "pass"


def test_bilateral_flag_when_one_sided():
    p = parse_statement("Bilateral ground glass opacities within the lung parenchyma")
    assert check_side(p, 100, shares(RUL=0.99, LUL=0.01), 1.0)["status"] == "flag"
    assert check_side(p, 100, shares(RUL=0.6, LUL=0.4), 1.0)["status"] == "pass"


def test_lobe_flag_and_pass():
    p = parse_statement("Subcentimeter calcific nodule in the upper lobe of the right lung")
    r = check_lobe(p, 100, shares(RLL=1.0), 1.0)
    assert r["status"] == "flag" and r["top_lobe"] == "RLL"
    assert check_lobe(p, 100, shares(RUL=0.8, RML=0.2), 1.0)["status"] == "pass"


def test_location_checks_not_applicable_cases():
    multi = parse_statement("Pleural effusion and adjacent atelectasis in the right lower lobe")
    assert check_side(multi, 100, shares(LLL=1.0), 1.0)["status"] == "not_applicable"
    one = parse_statement("Consolidation in the right lower lobe")
    assert check_lobe(one, 0, None, None)["status"] == "not_applicable"                       # empty outline
    assert check_side(one, 100, shares(LLL=1.0), THRESHOLDS["min_assigned_share"] - 0.1)["status"] == "not_applicable"
    nolobe = parse_statement("Pleural effusion on the left")
    assert check_lobe(nolobe, 100, shares(LLL=1.0), 1.0)["status"] == "not_applicable"
    assert check_side(nolobe, 100, shares(LLL=1.0), 1.0)["status"] == "pass"


def test_contradictory_output():
    nod = parse_statement("Nonspecific subcentimeter nodules in the left lung")
    assert check_contradictory_output(nod, 10, {"Lung nodule": "no"})["status"] == "flag"
    assert check_contradictory_output(nod, 10, {"Lung nodule": "yes"})["status"] == "pass"
    assert check_contradictory_output(nod, 10, None)["status"] == "not_applicable"
    con = parse_statement("Consolidation in the right lower lobe")       # needs No to BOTH mapped conditions
    assert check_contradictory_output(con, 10, {"Consolidation": "no", "Lung opacity": "yes"})["status"] == "pass"
    assert check_contradictory_output(con, 10, {"Consolidation": "no", "Lung opacity": "no"})["status"] == "flag"
    neg = parse_statement("No pleural effusion")
    assert check_contradictory_output(neg, 10, {"Pleural effusion": "no"})["status"] == "flag"   # negated, yet an outline
    assert check_contradictory_output(neg, 0, {"Pleural effusion": "no"})["status"] == "pass"


def test_audit_finding_pass_is_not_all_flags_and_keeps_versions():
    r = audit_finding("Consolidation in the right lower lobe", 100, shares(RLL=1.0), 1.0, {"Consolidation": "yes", "Lung opacity": "yes"})
    assert not r["any_primary_flag"] and set(r["checks"]) == {"empty_mask", "side", "lobe", "contradictory_output"}
    assert r["check_version"] and r["parsed"]["parser_version"]
    r = audit_finding("Nonspecific subcentimeter nodules", 0, None, None, {"Lung nodule": "no"})
    assert r["any_primary_flag"] and r["checks"]["empty_mask"]["status"] == "flag"


def test_attenuation_is_descriptive_only():
    d = describe_attenuation([-980, -960, -800, -500, -100, 50, 300])
    assert abs(sum(d["share_by_band"].values()) - 1.0) < 1e-3 and "status" not in d and "flag" not in str(d["note"]).split()[0]
    assert describe_attenuation([]) is None


def test_case_level_answer_gaps():
    parsed = [parse_statement("Consolidation in the right lower lobe")]
    gaps = case_level_answer_gaps({"Lung nodule": "yes", "Consolidation": "yes", "Cardiomegaly": "yes"}, parsed)
    assert gaps == ["Lung nodule"]            # cardiomegaly is outside the findings prompt's scope


# ---------------- matching across runs ----------------
def test_compare_answers():
    r = compare_answers({"A": "yes", "B": "no", "C": "unparsed"}, {"A": "yes", "B": "yes", "C": "no"})
    assert r["counts"] == {"same": 1, "flip": 1, "not_comparable": 1}


def F(i, voxels=100, types=("nodule",)):
    return {"id": i, "voxels": voxels, "types": list(types)}


def test_match_one_to_one_and_unmatched():
    r = match_findings([F("a1"), F("a2")], [F("b1"), F("b2")],
                       {("a1", "b1"): {"dice": 0.6, "centroid_mm": 3}, ("a1", "b2"): {"dice": 0.0, "centroid_mm": 150},
                        ("a2", "b1"): {"dice": 0.0, "centroid_mm": 150}, ("a2", "b2"): {"dice": 0.0, "centroid_mm": 90}})
    assert r["a"]["a1"]["outcome"] == "matched" and r["b"]["b1"]["outcome"] == "matched"
    assert r["a"]["a2"] == {"outcome": "unmatched", "kind": "disappeared", "links": []}
    assert r["b"]["b2"]["kind"] == "appeared"


def test_match_split_and_merge():
    split = match_findings([F("a1")], [F("b1"), F("b2")],
                           {("a1", "b1"): {"dice": 0.4, "centroid_mm": 10}, ("a1", "b2"): {"dice": 0.3, "centroid_mm": 12}})
    assert split["a"]["a1"]["outcome"] == "split" and split["b"]["b1"]["outcome"] == "split"
    merge = match_findings([F("a1"), F("a2")], [F("b1")],
                           {("a1", "b1"): {"dice": 0.4, "centroid_mm": 10}, ("a2", "b1"): {"dice": 0.3, "centroid_mm": 12}})
    assert merge["a"]["a1"]["outcome"] == "merge" and merge["b"]["b1"]["outcome"] == "merge"


def test_match_ambiguous_many_to_many_and_weak_link():
    mm = match_findings([F("a1"), F("a2")], [F("b1"), F("b2")],
                        {("a1", "b1"): {"dice": 0.4, "centroid_mm": 5}, ("a1", "b2"): {"dice": 0.3, "centroid_mm": 5},
                         ("a2", "b2"): {"dice": 0.3, "centroid_mm": 5}})
    assert mm["a"]["a1"]["outcome"] == "ambiguous"
    weak = match_findings([F("a1", types=("nodule",))], [F("b1", types=("consolidation",))],
                          {("a1", "b1"): {"dice": 0.0, "centroid_mm": 12}})       # near, no overlap, other type
    assert weak["a"]["a1"]["outcome"] == "ambiguous"
    near_same = match_findings([F("a1")], [F("b1")], {("a1", "b1"): {"dice": 0.0, "centroid_mm": 12}})
    assert near_same["a"]["a1"]["outcome"] == "matched"


def test_empty_outlines_never_link():
    r = match_findings([F("a1", voxels=0)], [F("b1", voxels=0)], {("a1", "b1"): {"dice": 0.0, "centroid_mm": None}})
    assert r["a"]["a1"]["kind"] == "disappeared" and r["b"]["b1"]["kind"] == "appeared"


# ---------------- sampling ----------------
def _inv(n_flag, n_un):
    return ([{"finding_id": f"f{i:03d}", "patient_id": f"p{i % 7}", "flagged": True} for i in range(n_flag)]
            + [{"finding_id": f"u{i:03d}", "patient_id": f"p{i % 7}", "flagged": False} for i in range(n_un)])


def test_manifest_probabilities_and_determinism():
    m = build_manifest(_inv(5, 40), n_unflagged=10, seed=7)
    assert m["n_findings"] == 45 and m["n_selected"] == 15
    fl = [r for r in m["rows"] if r["stratum"] == "flagged"]
    un = [r for r in m["rows"] if r["stratum"] == "unflagged"]
    assert all(r["selected"] and r["inclusion_probability"] == 1.0 for r in fl)
    assert all(r["inclusion_probability"] == 0.25 for r in un) and sum(r["selected"] for r in un) == 10
    assert abs(sum(r["weight"] for r in m["rows"] if r["selected"]) - 45) < 1e-9     # weights recover the inventory size
    assert build_manifest(list(reversed(_inv(5, 40))), 10, 7)["rows"] == m["rows"]   # order of input does not matter
    assert build_manifest(_inv(5, 40), 10, 8)["rows"] != m["rows"]


def test_manifest_edge_cases():
    m = build_manifest(_inv(3, 4), n_unflagged=99, seed=1)
    assert m["n_unflagged_selected"] == 4 and all(r["inclusion_probability"] == 1.0 for r in m["rows"])
    assert build_manifest(_inv(2, 0), 5, 1)["n_selected"] == 2
    with pytest.raises(ValueError):
        build_manifest(_inv(1, 1) + _inv(1, 0), 1, 1)


# ---------------- schema ----------------
def test_ids_and_review_rules():
    assert finding_id("valid_301_a_2", "P3_run", "findings_grounded", 2) == "valid_301_a_2|P3_run|findings_grounded|002"
    assert patient_id("valid_301_a_2") == "valid_301"
    with pytest.raises(ValueError):
        finding_id("a|b", "r", "s", 0)
    assert validate_stage1(["text_error", "finding_mask_error"]) == []
    assert validate_stage1(["no_identified_error"]) == []
    assert validate_stage1(["no_identified_error", "text_error"])
    assert validate_stage1([]) and validate_stage1(["anatomical_mask_error"])
    assert review_status(None) == "not_reviewed"
    assert review_status({"stage1": {"choices": ["text_error"]}}) == "initial_only"
    assert review_status({"stage1": {"choices": ["text_error"]}, "stage2": {"anatomy": "acceptable"}}) == "complete"


def test_review_v2_location_is_separate_and_v1_never_reads_as_assessed():
    import json
    from pathlib import Path
    from lung_ct.audit import schema as S
    ROOT = Path(__file__).resolve().parent.parent
    assert S.REVIEW_SCHEMA == "audit_review_v2" and "audit_review_v1" in S.REVIEW_SCHEMAS_READABLE
    ok = S.validate_stage1_location
    assert ok("present", {"status": "stated", "where": ["RUL", "RLL"]}) == []
    assert ok("cannot_tell", {"status": "indeterminate", "where": []}) == []
    assert ok("not_present", {"status": "not_assessed", "where": []}) == []
    assert ok("present", {"status": "not_in_a_lobe", "where": []}) == []
    assert ok(None, {"status": "not_assessed", "where": []}) == ["finding presence not answered"]
    assert ok("present", {"status": "stated", "where": []})            # 'stated' without a lobe
    assert ok("present", {"status": "stated", "where": ["S6"]})        # segments are not a location choice
    assert ok("present", {"status": "indeterminate", "where": ["RUL"]})
    assert ok("present", None)
    v1 = {"choices": ["no_identified_error"], "comment": "", "saved_utc": "x"}
    assert S.stage1_location(v1) == {"status": "not_recorded_form_v1", "where": []}
    assert S.stage1_location({**v1, "location": {"status": "stated", "where": ["LLL"]}}) == {"status": "stated", "where": ["LLL"]}
    assert S.stage1_location(None)["status"] is None
    assert S.review_status({"stage1": v1}) == "initial_only"          # a v1 review still counts as reviewed
    # the staged page asks for the reader's location in stage 1 and never offers geometric segments
    src = (ROOT / "scripts" / "build_blinded_viewer.py").read_text(encoding="utf-8")
    assert 'VIEWER_BUILD = "blinded_viewer_v2"' in src and "audit_review_protocol_v0_1_draft" in src
    assert "before any lobe outline is shown" in src and "r.draft.finding" in src
    proto = json.loads((ROOT / "configs" / "audit_review_protocol_v0_1_draft.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT / "configs" / "audit_review_protocol_v0_draft.json").read_text(encoding="utf-8"))
    assert proto["checks"]["primary"] == old["checks"]["primary"]                       # no check changed
    assert proto["checks"]["implementation"]["thresholds_provisional"] == old["checks"]["implementation"]["thresholds_provisional"]
    assert proto["sampling"] == old["sampling"] and proto["blinding"]["hidden_from_reader"] == old["blinding"]["hidden_from_reader"]
