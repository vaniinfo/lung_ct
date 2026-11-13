"""MedGemma prompt construction and strict response parsing."""
import json
from pathlib import Path

from lung_ct.vlm.prompting import build_messages, parse_response, prompt_sha256

CFG = json.loads((Path(__file__).resolve().parent.parent / "configs" /
                  "medgemma_ct_v1.json").read_text(encoding="utf-8"))
P = CFG["prompt"]

VALID = {
    "assessability": "assessable",
    "opacity_presence": "present",
    "distribution": {"laterality": "bilateral", "craniocaudal": "lower_predominant",
                     "axial": "peripheral"},
    "description": "Patchy opacities in both lungs.",
    "limitations": ["Sampled slices only."],
}


def test_message_layout_follows_notebook():
    msgs = build_messages(["img1", "img2"], ["SLICE 1", "SLICE 2"], P)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    c = msgs[0]["content"]
    assert [x["type"] for x in c] == ["text", "image", "text", "image", "text", "text"]
    assert c[0]["text"] == P["instruction"] and c[-1]["text"] == P["query"]
    assert c[1]["image"] == "img1" and c[2]["text"] == "SLICE 1"


def test_prompt_is_neutral_and_hash_stable():
    text = (P["instruction"] + P["query"]).lower()
    for leak in ("covid", "corona", "radiopaedia", "infection", "case_"):
        assert leak not in text
    assert prompt_sha256(P) == prompt_sha256(json.loads(json.dumps(P)))


def test_valid_whole_text_and_fenced():
    r = parse_response(json.dumps(VALID), truncated=False)
    assert r["status"] == "valid" and r["extraction_method"] == "whole_text"
    assert r["parsed"] == VALID and r["warnings"] == []
    r = parse_response("Here:\n```json\n" + json.dumps(VALID) + "\n```\n", truncated=False)
    assert r["status"] == "valid" and r["extraction_method"] == "fenced_block"


def test_invalid_json_and_empty_are_failures():
    assert parse_response("", False)["status"] == "empty_response"
    r = parse_response('{"assessability": "assessable",', True)
    assert r["status"] == "invalid_json" and r["truncated_at_max_new_tokens"] is True
    assert parse_response("No findings.", False)["status"] == "invalid_json"


def test_schema_errors_are_not_coerced():
    bad = {**VALID, "opacity_presence": "Present"}
    r = parse_response(json.dumps(bad), False)
    assert r["status"] == "schema_error" and "parsed" not in r
    assert r["unvalidated_object"]["opacity_presence"] == "Present"
    r = parse_response(json.dumps({**VALID, "confidence": 0.9}), False)
    assert r["status"] == "schema_error" and "unexpected keys" in r["errors"][0]
    missing = {k: v for k, v in VALID.items() if k != "limitations"}
    assert parse_response(json.dumps(missing), False)["status"] == "schema_error"
    bad_dist = {**VALID, "distribution": {"laterality": "bilateral"}}
    assert parse_response(json.dumps(bad_dist), False)["status"] == "schema_error"
    assert parse_response(json.dumps([VALID]), False)["status"] == "schema_error"


def test_consistency_warnings_flag_without_changing():
    obj = {**VALID, "opacity_presence": "absent"}
    r = parse_response(json.dumps(obj), False)
    assert r["status"] == "valid" and r["parsed"] == obj
    assert any("absent" in w for w in r["warnings"])
