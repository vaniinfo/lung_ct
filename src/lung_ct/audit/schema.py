"""Shared, versioned records for the finding-mask audit.

Three record kinds, kept in separate files on purpose:

  finding record   (audit_finding_v1)   what the model emitted: exact statement, mask reference, provenance.
                                        The ONLY record the blinded viewer loads.
  review record    (audit_review_v2)    what a reader entered in the blinded viewer (staged; initial judgment
                                        is kept unchanged after anatomy is revealed). v2 (2026-11-22) adds two
                                        stage-1 fields, saved and locked BEFORE the lobe outlines are revealed:
                                        whether the described finding is there, and where the reader sees it.
                                        They are separate fields. audit_review_v1 records stay valid; their
                                        location reads as 'not_recorded_form_v1', never as assessed.
  verdict record   (audit_verdict_v1)   what the automatic checks said. Never loaded by the blinded viewer.

Ids are stable: a finding id is built from the case, the model run, the output source and the position of the
finding in that output, so re-building an inventory from the same run gives the same ids.
"""
from __future__ import annotations

FINDING_SCHEMA = "audit_finding_v1"
REVIEW_SCHEMA = "audit_review_v2"
REVIEW_SCHEMAS_READABLE = ("audit_review_v1", "audit_review_v2")
VERDICT_SCHEMA = "audit_verdict_v1"
INVENTORY_SCHEMA = "audit_inventory_v1"

# ---- review form (protocol: configs/audit_review_protocol_*.json, section grading_form) ----
ERROR_TYPES_STAGE1 = ("text_error", "finding_mask_error")          # can be combined
EXCLUSIVE_STAGE1 = ("no_identified_error", "cannot_adjudicate")    # exclude every other choice
ANATOMY_CHOICES = ("acceptable", "anatomical_mask_error", "cannot_adjudicate")
# stage 1, v2: finding identity and location are graded separately, and separately from the error types
FINDING_PRESENCE = ("present", "not_present", "cannot_tell")
LOBE_NAMES = ("RUL", "RML", "RLL", "LUL", "LLL")
LOCATION_STATUS = ("stated", "not_in_a_lobe", "indeterminate", "not_assessed")   # 'stated' carries a list of lobes
LOCATION_NOT_RECORDED = "not_recorded_form_v1"
PRESENCE_LABELS = {"present": "Finding is there", "not_present": "Finding is not there", "cannot_tell": "Cannot tell"}
LOCATION_LABELS = {"not_in_a_lobe": "Not in a lung lobe", "indeterminate": "Indeterminate", "not_assessed": "Not assessed"}
CHOICE_LABELS = {
    "text_error": "Text error",
    "finding_mask_error": "Finding-mask error",
    "anatomical_mask_error": "Anatomical-mask error",
    "no_identified_error": "No identified error",
    "cannot_adjudicate": "Cannot adjudicate",
    "acceptable": "Lobe boundaries acceptable here",
}
CHOICE_HELP = {
    "text_error": "the statement is wrong for this scan (finding not there, wrong type, wrong side or place)",
    "finding_mask_error": "the outline does not cover what the statement describes",
    "no_identified_error": "statement and outline both look right",
    "cannot_adjudicate": "cannot be judged on this scan",
}


def finding_id(case_id: str, run_id: str, source: str, index: int) -> str:
    """Stable id, e.g. 'valid_301_a_2|P3_holoct_meeting4_20261120_182147|findings_grounded|002'."""
    for part in (case_id, run_id, source):
        if "|" in part:
            raise ValueError(f"'|' is not allowed in id parts: {part!r}")
    return f"{case_id}|{run_id}|{source}|{index:03d}"


def patient_id(case_id: str) -> str:
    """CT-RATE: valid_301_a_2 -> valid_301 (findings cluster within patients)."""
    parts = case_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else case_id


def validate_stage1(choices: list[str]) -> list[str]:
    """Return problems with a stage-1 selection (empty list = valid)."""
    problems = []
    allowed = set(ERROR_TYPES_STAGE1) | set(EXCLUSIVE_STAGE1)
    if not choices:
        problems.append("nothing selected")
    if any(c not in allowed for c in choices):
        problems.append("unknown choice")
    if len(set(choices)) != len(choices):
        problems.append("duplicate choice")
    if any(c in EXCLUSIVE_STAGE1 for c in choices) and len(choices) > 1:
        problems.append("'no identified error' and 'cannot adjudicate' exclude every other choice")
    return problems


def validate_stage1_location(finding: str | None, location: dict | None) -> list[str]:
    """Problems with the v2 stage-1 finding / location fields (empty list = valid)."""
    problems = []
    if finding not in FINDING_PRESENCE:
        problems.append("finding presence not answered")
    loc = location or {}
    st, where = loc.get("status"), loc.get("where") or []
    if st not in LOCATION_STATUS:
        problems.append("unknown location status")
    if st == "stated" and (not where or any(w not in LOBE_NAMES for w in where) or len(set(where)) != len(where)):
        problems.append("'stated' needs one or more distinct lobe names")
    if st != "stated" and where:
        problems.append("lobes given without status 'stated'")
    return problems


def stage1_location(stage1: dict | None) -> dict:
    """Location of a locked stage-1 record from either schema version. A v1 record never reads as assessed."""
    if not stage1:
        return {"status": None, "where": []}
    loc = stage1.get("location")
    return {"status": loc["status"], "where": list(loc.get("where") or [])} if loc else {"status": LOCATION_NOT_RECORDED, "where": []}


def review_status(review: dict | None) -> str:
    """not_reviewed / initial_only / complete. 'not_reviewed' must never be counted as correct."""
    if not review or not (review.get("stage1") or {}).get("choices"):
        return "not_reviewed"
    return "complete" if (review.get("stage2") or {}).get("anatomy") else "initial_only"
