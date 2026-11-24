"""Reading reviewer grade CSVs saved from the viewer pages (grades_v1, grades_v2, and policy-labelled packs such as M5).

Rules (2026-11-23):
  - A grade belongs to the POLICY (checklist version) and PACK of the page it was made on. Grades made on different
    policies or packs are never put under one key and never pooled in one table.
  - Files without the policy columns (everything before pack M5) are read exactly as before.
  - In a policy-labelled file the agent's answer is the column `agent_current_assessment` ('suggested' stays distinct
    from 'present'). The column `provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT` is carried along and is never
    read as an answer of the agent.
  - A policy-labelled file is checked against the pack it claims (PACK_INFO.json): checklist version, pack name, case
    and run must match, and the exported agreement must be the one the form's rules give. Any mismatch is reported.
  - Automated-test exports are recognised by their reviewer marker and are never mixed with reviewer files.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

POLICY_COLUMNS = ["checklist_version", "pack", "agent_current_assessment", "agent_headline_shown", "agent_stored_status", "agent_decision_basis",
                  "agent_mask_state", "provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT", "provisional_limits"]
THRESHOLD_COLUMN = "provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT"
TEST_MARKER = "automated test"
ASSESSMENTS = ("present", "suggested", "absent", "cannot tell", "cannot assess")
PACK_NOT_RECORDED = "(pack not recorded in this file)"


def read_grade_file(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [dict(r, source_file=path.name) for r in csv.DictReader(fh)]


def is_policy_row(r: dict) -> bool:
    """True for a row exported by a policy-labelled page (the policy columns exist in the file and name a policy or pack)."""
    return bool((r.get("checklist_version") or "").strip() or (r.get("pack") or "").strip())


def review_policy(r: dict) -> str:
    return (r.get("checklist_version") or "").strip() or (r.get("checklist_config") or "").strip() or "(policy not recorded)"


def review_pack(r: dict) -> str:
    return (r.get("pack") or "").strip() or PACK_NOT_RECORDED


def group_key(r: dict) -> tuple:
    """Key under which saves of one review are compared: case, reviewer, policy, pack. Never case and reviewer alone."""
    return (r["case_id"], r["reviewer"].strip().lower(), review_policy(r), review_pack(r))


def is_test_row(r: dict) -> bool:
    return TEST_MARKER in (r.get("reviewer") or "").lower()


def agent_answer_category(r: dict) -> str:
    """The agent's answer a grade is compared with. Never taken from the provisional threshold column."""
    if is_policy_row(r):
        return (r.get("agent_current_assessment") or "").strip()
    return (r.get("agent_status") or "").strip()


def expected_agreement(r: dict) -> str:
    """Agreement the grading form writes for a policy-labelled row (same rules as scripts/viewer_grading_v2.py)."""
    a, cat, stored = (r.get("doctor_answer") or "").strip(), agent_answer_category(r), (r.get("agent_stored_status") or "").strip()
    if not a:
        return ""
    if cat == "suggested":
        return "open" if a == "cannot_tell" else "suggested_reviewer_present" if a == "present" else "suggested_reviewer_absent"
    if a == "cannot_tell" or stored in ("cannot tell", "cannot assess"):
        return "open"
    if a == "present":
        return "agrees with agent" if stored == "present" else "agent missed it"
    return "agrees with agent" if stored == "absent" else "agent over-called"


def load_pack_registry(root: Path) -> dict:
    """Pack name -> PACK_INFO.json content, for every policy-labelled pack under runs/.

    Each page entry gets `expected`: per condition, what the page showed (current assessment, stored status, provisional
    threshold result), read from the checklist file the page was built from. If that file no longer has the hash recorded
    when the pack was built, `expected_problem` says so and rows of that page cannot be verified.
    """
    import hashlib
    from lung_ct.agent.checklist import assessment_category
    reg = {}
    for f in sorted((root / "runs").glob("*/PACK_INFO.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("pack") in reg:
            raise SystemExit(f"two packs are named {d['pack']}: {reg[d['pack']]['folder']} and {d.get('folder')}")
        for pg in d.get("pages", []):
            cf = root / pg.get("checklist_file", "")
            if not pg.get("checklist_file") or not cf.exists():
                pg["expected_problem"] = "the checklist file of this page is missing"
            elif hashlib.sha256(cf.read_bytes()).hexdigest() != pg.get("checklist_file_sha256"):
                pg["expected_problem"] = "the checklist file of this page changed after the pack was built"
            else:
                pg["expected"] = {i["key"]: {"category": assessment_category(i), "stored": i["status"],
                                             "threshold": (i.get("provisional_threshold_result") or {}).get("answer", "")}
                                  for i in json.loads(cf.read_text(encoding="utf-8"))["items"]}
        reg[d["pack"]] = d
    return reg


def check_policy_rows(rows: list[dict], registry: dict) -> list[str]:
    """Problems in policy-labelled rows. An empty list means every row matches the page it claims to grade."""
    bad = []
    for n, r in enumerate(rows, 1):
        if not is_policy_row(r):
            continue
        where = f"{r.get('source_file', '?')} row {n} ({r.get('case_id')}, {r.get('condition_key')})"
        ver, cfg, pack = (r.get("checklist_version") or "").strip(), (r.get("checklist_config") or "").strip(), (r.get("pack") or "").strip()
        if not ver or not pack:
            bad.append(f"{where}: checklist_version or pack is empty")
            continue
        if ver != cfg:
            bad.append(f"{where}: checklist_version '{ver}' differs from checklist_config '{cfg}'")
        info = registry.get(pack)
        if info is None:
            bad.append(f"{where}: pack '{pack}' is not a known policy-labelled pack")
            continue
        if info.get("policy") != ver:
            bad.append(f"{where}: pack {pack} shows {info.get('policy')}, the file says {ver}")
        page = next((p for p in info.get("pages", []) if p["case_id"] == r.get("case_id")), None)
        if page is None:
            bad.append(f"{where}: pack {pack} has no page for this case")
        elif page["run"] != (r.get("run") or "").strip():
            bad.append(f"{where}: run '{r.get('run')}' is not the run shown on the {pack} page ({page['run']})")
        if r.get("condition_key") == "_overall":
            continue
        cat, st = (r.get("agent_current_assessment") or "").strip(), (r.get("agent_status") or "").strip()
        if cat not in ASSESSMENTS:
            bad.append(f"{where}: agent_current_assessment '{cat}' is not one of {ASSESSMENTS}")
        if st != cat:
            bad.append(f"{where}: agent_status '{st}' differs from agent_current_assessment '{cat}'")
        if cat == "suggested" and (r.get("agent_stored_status") or "").strip() != "present":
            bad.append(f"{where}: 'suggested' needs the stored status 'present'")
        if page is not None and "expected_problem" in page:
            bad.append(f"{where}: cannot be verified against the pack: {page['expected_problem']}")
        elif page is not None and "expected" in page:
            exp = page["expected"].get(r.get("condition_key"))
            if exp is None:
                bad.append(f"{where}: the {pack} page has no such condition")
            elif (cat, (r.get("agent_stored_status") or "").strip(), (r.get(THRESHOLD_COLUMN) or "").strip()) != (exp["category"], exp["stored"], exp["threshold"]):
                bad.append(f"{where}: the file says the page showed '{cat}' (stored '{r.get('agent_stored_status')}', threshold column '{r.get(THRESHOLD_COLUMN)}'); "
                           f"the {pack} page showed '{exp['category']}' (stored '{exp['stored']}', threshold '{exp['threshold']}')")
        if (r.get("agent_mask_state") or "").strip() and cat == "absent":
            bad.append(f"{where}: a predicted-mask item is exported as 'absent'; a mask below a provisional limit is never an absence")
        if (r.get("doctor_answer") or "").strip() and (r.get("agreement") or "").strip() != expected_agreement(r):
            bad.append(f"{where}: agreement '{r.get('agreement')}' is not what the form's rules give ('{expected_agreement(r)}')")
    return bad


def check_no_test_mix(rows: list[dict]) -> list[str]:
    t = {r.get("source_file") for r in rows if is_test_row(r)}
    o = {r.get("source_file") for r in rows if not is_test_row(r)}
    return [f"automated-test exports {sorted(t)} are mixed with reviewer files {sorted(o)}"] if t and o else []


def keep_latest_save(rows: list[dict]) -> list[dict]:
    """Newest save per (case, reviewer, policy, pack). A grade made on one policy never replaces a grade made on another."""
    latest: dict = {}
    for r in rows:
        k = group_key(r)
        if k not in latest or r["saved_utc"] > latest[k]:
            latest[k] = r["saved_utc"]
    return [r for r in rows if latest[group_key(r)] == r["saved_utc"]]


def policy_groups(rows: list[dict]) -> list[tuple]:
    return sorted({(review_policy(r), review_pack(r)) for r in rows})
