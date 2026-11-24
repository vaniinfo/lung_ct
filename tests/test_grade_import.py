"""Grade CSV import: grades stay with the policy and pack of the page they were made on (pack M5 and later)."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from lung_ct.audit import grade_import as gi

ROOT = Path(__file__).resolve().parent.parent
OLD_COLS = ["case_id", "reviewer", "saved_utc", "condition_key", "condition", "agent_answer", "agent_status", "doctor_answer", "severity", "confidence",
            "agreement", "comment", "checklist_config", "run"]


def old_row(**k):
    r = dict(case_id="valid_1_a_2", reviewer="Dr X", saved_utc="2026-11-20T10:00:00.000Z", condition_key="pleural_effusion", condition="Pleural effusion",
             agent_answer="present", agent_status="present", doctor_answer="present", severity="", confidence="", agreement="agrees with agent", comment="",
             checklist_config="checklist_v3_4", run="A1_run_1")
    r.update(k)
    return r


def new_row(**k):
    r = old_row(saved_utc="2026-11-24T10:00:00.000Z", agent_answer="suggested by segmentation", agent_status="suggested", agreement="suggested_reviewer_present",
                checklist_config="checklist_v3_8", checklist_version="checklist_v3_8", pack="M5", agent_current_assessment="suggested",
                agent_headline_shown="Pleural effusion suggested by segmentation", agent_stored_status="present", agent_decision_basis="predicted mask only",
                agent_mask_state="volume_at_or_above_present_limit", provisional_limits="present >= 20, none < 5 mL (provisional, not validated)")
    r[gi.THRESHOLD_COLUMN] = "present"
    r.update(k)
    return r


REG = {"M5": {"pack": "M5", "policy": "checklist_v3_8", "pages": [
    {"case_id": "valid_1_a_2", "run": "A1_run_1", "expected": {"pleural_effusion": {"category": "suggested", "stored": "present", "threshold": "present"},
                                                                 "consolidation": {"category": "cannot tell", "stored": "cannot tell", "threshold": ""}}}]}}


def test_old_files_are_not_policy_rows_and_are_never_checked_or_changed():
    r = old_row()
    assert not gi.is_policy_row(r) and gi.check_policy_rows([r], REG) == []
    assert gi.review_policy(r) == "checklist_v3_4" and gi.review_pack(r) == gi.PACK_NOT_RECORDED
    assert gi.agent_answer_category(r) == "present"


def test_a_grade_made_on_one_policy_never_replaces_a_grade_made_on_another():
    old, new = old_row(), new_row()
    assert gi.group_key(old) != gi.group_key(new)                       # same case, same reviewer, different policy and pack
    kept = gi.keep_latest_save([old, new])
    assert old in kept and new in kept                                  # the earlier grade is not dropped as an "older save"
    newer = new_row(saved_utc="2026-11-25T09:00:00.000Z", doctor_answer="absent", agreement="suggested_reviewer_absent")
    kept = gi.keep_latest_save([old, new, newer])
    assert old in kept and newer in kept and new not in kept            # within one policy and pack the newest save wins
    assert gi.policy_groups(kept) == [("checklist_v3_4", gi.PACK_NOT_RECORDED), ("checklist_v3_8", "M5")]


def test_suggested_stays_distinct_and_the_threshold_column_is_never_an_answer():
    r = new_row()
    assert gi.agent_answer_category(r) == "suggested"
    for v in ("present", "none", "undecided", ""):
        assert gi.agent_answer_category(dict(r, **{gi.THRESHOLD_COLUMN: v})) == "suggested"
    low = new_row(agent_status="cannot tell", agent_current_assessment="cannot tell", agent_stored_status="cannot tell", agent_mask_state="empty_mask",
                  doctor_answer="absent", agreement="open", **{gi.THRESHOLD_COLUMN: "none"})
    assert gi.agent_answer_category(low) == "cannot tell" and gi.expected_agreement(low) == "open"     # a historical 'none' is not an 'absent'
    assert gi.expected_agreement(new_row(doctor_answer="present")) == "suggested_reviewer_present"
    assert gi.expected_agreement(new_row(doctor_answer="absent")) == "suggested_reviewer_absent"
    assert gi.expected_agreement(new_row(doctor_answer="cannot_tell")) == "open"
    conf = dict(agent_status="present", agent_current_assessment="present", agent_stored_status="present")
    assert gi.expected_agreement(new_row(doctor_answer="present", **conf)) == "agrees with agent"
    assert gi.expected_agreement(new_row(doctor_answer="absent", **conf)) == "agent over-called"


def test_a_file_that_does_not_match_the_page_it_claims_is_flagged():
    assert gi.check_policy_rows([new_row()], REG) == []
    for change, word in ((dict(pack="M4"), "not a known policy-labelled pack"), (dict(checklist_version="checklist_v3_7"), "differs from checklist_config"),
                         (dict(checklist_version="checklist_v3_4", checklist_config="checklist_v3_4"), "pack M5 shows checklist_v3_8"),
                         (dict(run="A1_other"), "is not the run shown"), (dict(case_id="valid_2_a_2"), "no page for this case"),
                         (dict(agent_status="present"), "differs from agent_current_assessment"),
                         (dict(agent_status="present", agent_current_assessment="present", agreement="agrees with agent"), "the M5 page showed 'suggested'"),
                         (dict(agreement="agrees with agent"), "not what the form's rules give"),
                         ({gi.THRESHOLD_COLUMN: "none"}, "the M5 page showed 'suggested'"),
                         (dict(pack=""), "checklist_version or pack is empty")):
        bad = gi.check_policy_rows([new_row(**change)], REG)
        assert bad and any(word in b for b in bad), (change, bad)
    absent = new_row(agent_status="absent", agent_current_assessment="absent", agent_stored_status="absent", agent_mask_state="empty_mask", doctor_answer="absent",
                     agreement="agrees with agent", **{gi.THRESHOLD_COLUMN: "none"})
    assert any("never an absence" in b for b in gi.check_policy_rows([absent], REG))


def test_automated_test_exports_are_never_mixed_with_reviewer_files():
    t = new_row(reviewer="AUTOMATED TEST - not a reviewer", source_file="a.csv")
    assert gi.is_test_row(t) and gi.check_no_test_mix([t]) == [] and gi.check_no_test_mix([dict(old_row(), source_file="b.csv")]) == []
    assert gi.check_no_test_mix([t, dict(old_row(), source_file="b.csv")])


def test_pack_registry_reads_what_the_page_showed_and_notices_a_changed_checklist_file(tmp_path):
    chk = {"items": [{"key": "pleural_effusion", "status": "present", "headline": "Pleural effusion suggested by segmentation",
                      "provisional_threshold_result": {"answer": "present"}},
                     {"key": "consolidation", "status": "cannot tell"}]}
    cf = tmp_path / "runs" / "A1_run_1" / "checklist_checklist_v3_8.json"
    cf.parent.mkdir(parents=True)
    cf.write_text(json.dumps(chk), encoding="utf-8")
    info = {"pack": "M5", "policy": "checklist_v3_8", "folder": "runs/M5_x", "pages": [
        {"case_id": "valid_1_a_2", "run": "A1_run_1", "checklist_file": "runs/A1_run_1/checklist_checklist_v3_8.json",
         "checklist_file_sha256": hashlib.sha256(cf.read_bytes()).hexdigest()}]}
    (tmp_path / "runs" / "M5_x").mkdir()
    (tmp_path / "runs" / "M5_x" / "PACK_INFO.json").write_text(json.dumps(info), encoding="utf-8")
    reg = gi.load_pack_registry(tmp_path)
    assert reg["M5"]["pages"][0]["expected"] == {"pleural_effusion": {"category": "suggested", "stored": "present", "threshold": "present"},
                                                  "consolidation": {"category": "cannot tell", "stored": "cannot tell", "threshold": ""}}
    assert gi.check_policy_rows([new_row()], reg) == []
    cf.write_text(json.dumps(chk) + "\n", encoding="utf-8")            # the checklist file changed after the pack was built
    bad = gi.check_policy_rows([new_row()], gi.load_pack_registry(tmp_path))
    assert any("changed after the pack was built" in b for b in bad)


def test_collector_reads_a_folder_of_earlier_files_the_old_way(tmp_path):
    f = tmp_path / "grades_valid_1_a_2_Dr_X_2026-11-20.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OLD_COLS)
        w.writeheader()
        w.writerow(old_row(saved_utc="2026-11-19T10:00:00.000Z", doctor_answer="absent", agreement="agent over-called"))
        w.writerow(old_row())
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_grades.py"), "--folder", str(tmp_path), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(out / "all_grades.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["doctor_answer"] == "present"            # newest save per case and reviewer, as before
    assert "review_policy" not in rows[0] and "## Policy " not in (out / "summary.md").read_text(encoding="utf-8")
