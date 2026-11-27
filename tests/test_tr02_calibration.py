"""TR02 calibration: label-blind selection, halves, confirmation logic (no model, no network)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ctrate_fetch_calibration as cal  # noqa: E402
import trust_rules_confirm as trc  # noqa: E402


def _names(n):
    return [f"valid_{i}_a_1.nii.gz" for i in range(1, n + 1)] + [f"valid_{i}_a_2.nii.gz" for i in range(1, n + 1, 3)] + [f"valid_{i}_b_1.nii.gz" for i in range(1, 6)]


def test_selection_is_one_scan_per_new_patient_repeatable_and_split_in_halves():
    used = {"valid_3", "valid_7"}
    a = cal.select_patients(_names(50), used, 20, 20261126)
    assert a == cal.select_patients(list(reversed(_names(50))), used, 20, 20261126)      # input order does not matter
    pats = [s["patient"] for s in a]
    assert len(pats) == len(set(pats)) == 20 and not used & set(pats)
    assert [s["part"] for s in a] == ["fit"] * 10 + ["confirm"] * 10
    assert cal.select_patients(_names(50), used, 20, 1) != a


def test_selection_never_reads_labels():
    src = (ROOT / "scripts" / "ctrate_fetch_calibration.py").read_text(encoding="utf-8")
    body = src.split("def build_selection")[1].split("def main")[0]
    assert 'names = [r["VolumeName"] for r in labels]' in body                        # the label table is read for the scan names only
    assert "lab_by" not in body and "pos_of" not in body and "abn" not in body


def test_the_frozen_selection_exists_only_if_made_with_the_plan_seed():
    plan = (ROOT / "docs" / "prompt_tests" / "TR02_calibration_400_PLAN.md").read_text(encoding="utf-8")
    assert "frozen 2026-11-26" in plan and "Changes after freezing: none" in plan and str(cal.SEED) in plan
    p = Path(cal.DEFAULT_ROOT) / "selection.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))
        assert s["seed"] == cal.SEED and s["requested"] == 400 and len(s["selection"]) == 400


def test_confirmed_needs_both_halves():
    yes = {"present_may_decide": True, "absent_may_decide": False}
    no = {"present_may_decide": False, "absent_may_decide": False}
    fit = {"A": {"s": yes}, "B": {"s": yes}}
    con = {"A": {"s": yes}, "B": {"s": no}, "C": {"s": {"present_may_decide": False, "absent_may_decide": True}}}
    r = trc.confirmed_entries(fit, con)
    assert r["confirmed"] == [("A", "s", "present")] and r["fit_half_only"] == [("B", "s", "present")] and r["confirm_half_only"] == [("C", "s", "absent")]


def test_confirm_half_uses_the_frozen_cutoff_and_never_refits():
    rule = {**trc.RULE, "resamples": 50}
    gt = {f"valid_{i}_a_1": {"labels": ["X"] if i % 3 == 0 else []} for i in range(60)}
    scans = sorted(gt)
    co = {"X": {s: (0.9 if "X" in gt[s]["labels"] else 0.1) for s in scans}}
    _, fit = trc.half_result({}, co, scans, gt, None, rule)
    assert 0.1 < fit["X"]["cutoff"] < 0.9 and fit["X"]["balanced_accuracy"] == 1.0
    _, frozen = trc.half_result({}, co, scans, gt, {"X": 0.95}, rule)                 # a bad frozen cut-off stays bad
    assert frozen["X"]["cutoff"] == 0.95 and frozen["X"]["found"] == 0 and frozen["X"]["balanced_accuracy"] == 0.5
    _, none = trc.half_result({}, co, scans, gt, {}, rule)                           # no cut-off: every answer is "neither"
    assert none["X"]["unreadable"] == 60


def test_scan_list_options_exist_in_the_model_scripts():
    nv = (ROOT / "scripts" / "prompt_test_nvreason.py").read_text(encoding="utf-8")
    co = (ROOT / "scripts" / "prompt_test_colipri.py").read_text(encoding="utf-8")
    an = (ROOT / "scripts" / "anatomy_contrast_check.py").read_text(encoding="utf-8")
    assert "--scan-list" in nv and "ct_path(scan)" in nv
    assert "--scores-only" in co and 'gt = {} if args.scores_only' in co
    assert "--write-scan-list" in an and "--seg-root" in an


def test_scans_outside_the_repository_are_read_from_the_scan_list():
    import prompt_test_nvreason as pn
    pn.CT_PATHS.clear()
    pn.CT_PATHS["valid_9_a_1"] = "D:/datasets/x/valid_9_a_1.nii.gz"
    assert pn.ct_path("valid_9_a_1") == Path("D:/datasets/x/valid_9_a_1.nii.gz")
    assert pn.ct_path("valid_5_a_1") == ROOT / "data/raw/ct_rate/volumes" / "valid_5_a_1.nii.gz"
    pn.CT_PATHS.clear()


def test_review_sample_cells_and_draw():
    import review_sample_model_vs_report as rs
    assert rs.cell_of("present", False) == "model_only" and rs.cell_of("absent", True) == "report_only" and rs.cell_of(None, True) is None
    frame = {("x", "model_only"): [f"s{i}" for i in range(10)], ("x", "agree_absent"): ["a", "b"]}
    a = rs.draw(frame, {"model_only": 4, "agree_absent": 5}, 1)
    assert a == rs.draw(frame, {"model_only": 4, "agree_absent": 5}, 1)
    assert sum(i["cell"] == "model_only" for i in a) == 4 and sum(i["cell"] == "agree_absent" for i in a) == 2
    assert {i["probability"] for i in a if i["cell"] == "model_only"} == {0.4}
