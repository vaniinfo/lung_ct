"""Frozen prompt construction and strict response parsing for MedGemma CT inference.

Message layout follows the official notebook: instruction text, then for each slice an
image followed by a "SLICE n" text item, then the query text, as one user turn.
The prompt never contains case identifiers, filenames, or reference information.

Parsing never repairs content: the raw response is kept separately, JSON is only located
(whole text / fenced block / outermost braces, method recorded) and validated against a
fixed schema. Anything else is a recorded failure.
"""
from __future__ import annotations

import hashlib
import json
import re

ENUMS = {
    "assessability": {"assessable", "limited", "not_assessable"},
    "opacity_presence": {"present", "absent", "indeterminate"},
}
DISTRIBUTION_ENUMS = {
    "laterality": {"bilateral", "right", "left", "not_applicable", "indeterminate"},
    "craniocaudal": {"upper_predominant", "middle_predominant", "lower_predominant",
                     "no_predominance", "not_applicable", "indeterminate"},
    "axial": {"peripheral", "central", "mixed", "not_applicable", "indeterminate"},
}
TOP_KEYS = {"assessability", "opacity_presence", "distribution", "description", "limitations"}


def prompt_sha256(prompt_cfg: dict) -> str:
    blob = json.dumps(prompt_cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_messages(images: list, labels: list[str], prompt_cfg: dict) -> list[dict]:
    """images: objects accepted by the processor (PIL images); labels: 'SLICE n' texts."""
    if len(images) != len(labels):
        raise ValueError("images and labels differ in length")
    content = [{"type": "text", "text": prompt_cfg["instruction"]}]
    for im, label in zip(images, labels):
        content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": label})
    content.append({"type": "text", "text": prompt_cfg["query"]})
    return [{"role": "user", "content": content}]


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> tuple[object | None, str, str | None]:
    """Return (obj, method, error). Methods: whole_text, fenced_block, outer_braces."""
    if not text or not text.strip():
        return None, "none", "empty response"
    s = text.strip()
    try:
        return json.loads(s), "whole_text", None
    except json.JSONDecodeError:
        pass
    m = _FENCE.search(s)
    if m:
        try:
            return json.loads(m.group(1).strip()), "fenced_block", None
        except json.JSONDecodeError as e:
            return None, "fenced_block", f"fenced block is not valid JSON: {e}"
    a, b = s.find("{"), s.rfind("}")
    if 0 <= a < b:
        try:
            return json.loads(s[a:b + 1]), "outer_braces", None
        except json.JSONDecodeError as e:
            return None, "outer_braces", f"brace span is not valid JSON: {e}"
    return None, "none", "no JSON object found"


def validate(obj: object) -> list[str]:
    """Schema errors (empty list = valid). No coercion, no defaults."""
    if not isinstance(obj, dict):
        return [f"top level is {type(obj).__name__}, expected object"]
    errs = []
    keys = set(obj)
    if keys - TOP_KEYS:
        errs.append(f"unexpected keys: {sorted(keys - TOP_KEYS)}")
    if TOP_KEYS - keys:
        errs.append(f"missing keys: {sorted(TOP_KEYS - keys)}")
    for k, allowed in ENUMS.items():
        if k in obj and obj[k] not in allowed:
            errs.append(f"{k}={obj[k]!r} not in {sorted(allowed)}")
    d = obj.get("distribution")
    if "distribution" in obj:
        if not isinstance(d, dict):
            errs.append("distribution is not an object")
        else:
            dk = set(d)
            if dk != set(DISTRIBUTION_ENUMS):
                errs.append(f"distribution keys {sorted(dk)} != {sorted(DISTRIBUTION_ENUMS)}")
            for k, allowed in DISTRIBUTION_ENUMS.items():
                if k in d and d[k] not in allowed:
                    errs.append(f"distribution.{k}={d[k]!r} not in {sorted(allowed)}")
    if "description" in obj and not (isinstance(obj["description"], str)
                                     and obj["description"].strip()):
        errs.append("description must be a non-empty string")
    if "limitations" in obj:
        lim = obj["limitations"]
        if not (isinstance(lim, list) and all(isinstance(x, str) for x in lim)):
            errs.append("limitations must be a list of strings")
    return errs


def consistency_warnings(obj: dict) -> list[str]:
    """Internal contradictions in a schema-valid response (flagged, not corrected)."""
    warns = []
    d = obj["distribution"]
    if obj["opacity_presence"] == "absent":
        bad = [k for k, v in d.items() if v not in ("not_applicable", "indeterminate")]
        if bad:
            warns.append(f"opacities absent but distribution given for {bad}")
    if obj["opacity_presence"] == "present":
        na = [k for k, v in d.items() if v == "not_applicable"]
        if na:
            warns.append(f"opacities present but distribution not_applicable for {na}")
    if obj["assessability"] == "not_assessable" and obj["opacity_presence"] != "indeterminate":
        warns.append("not_assessable but opacity_presence is not indeterminate")
    return warns


def parse_response(text: str, truncated: bool) -> dict:
    """Classify a raw response. status: valid | empty_response | invalid_json | schema_error."""
    obj, method, err = extract_json(text)
    rec = {"extraction_method": method, "truncated_at_max_new_tokens": truncated,
           "errors": [], "warnings": []}
    if obj is None:
        rec["status"] = "empty_response" if err == "empty response" else "invalid_json"
        rec["errors"].append(err)
        return rec
    errs = validate(obj)
    if errs:
        rec["status"] = "schema_error"
        rec["errors"] = errs
        rec["unvalidated_object"] = obj
        return rec
    rec["status"] = "valid"
    rec["parsed"] = obj
    rec["warnings"] = consistency_warnings(obj)
    return rec
