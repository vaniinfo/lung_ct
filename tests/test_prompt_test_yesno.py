"""Yes / no prompt test (PT02) and trust rules (TR01): reading answers, counting, the pre-stated rules, the frozen configs (no model)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import prompt_test_nvreason as pn  # noqa: E402
import prompt_test_yesno as py  # noqa: E402
import trust_rules as tr  # noqa: E402

CFG = json.loads((ROOT / "configs" / "nvreason_prompt_sets_v1.json").read_text(encoding="utf-8"))
BASE = json.loads((ROOT / "configs" / "nvreason_ct_v1.json").read_text(encoding="utf-8"))
AN = {**CFG["analysis"], "bootstrap": {**CFG["analysis"]["bootstrap"], "resamples": 200}}


def test_baseline_set_is_exactly_the_current_question_and_conditions():
    assert CFG["sets"]["A"]["template"] == BASE["questions"]["presence_template"] and CFG["sets"]["A"]["thinking"] is BASE["questions"]["enable_thinking"]
    assert [(c["key"], c["phrase"], c["ct_rate_label"]) for c in CFG["conditions"]] == [(c["key"], c["phrase"], c["model_label"]) for c in BASE["questions"]["conditions"]]
    assert CFG["compared_with_baseline"] == ["V", "B", "C", "D", "E"] and CFG["derived_sets"]["V"]["from"] == ["A", "A2", "A3"]
    assert all(c["definition"] and c["definition"] in pn.question(CFG["sets"]["C"], c) for c in CFG["conditions"])
    assert len({pn.question(CFG["sets"][s], CFG["conditions"][0]) for s in ("A", "A2", "A3")}) == 3
    assert (ROOT / CFG["plan"]).exists() and CFG["fixed"]["do_sample"] is False


def test_reading_answers_never_guesses():
    assert pn.read_yes_no("Yes") == "present" and pn.read_yes_no("No.") == "absent" and pn.read_yes_no("I think so") is None and pn.read_yes_no("") is None
    assert pn.read_yes_no("<think>it says yes somewhere</think>\n\nNo") == "absent"          # only the text after the reasoning counts
    assert pn.read_location("None") == "absent" and pn.read_location("No pleural effusion.") == "absent" and pn.read_location("Not present") == "absent"
    assert pn.read_location("Right lower lobe") == "present" and pn.read_location("Bilateral, more on the left") == "present" and pn.read_location("  ") is None
    assert pn.common_prefix_len([[1, 2, 3, 9], [1, 2, 4], [1, 2, 3]]) == 2 and pn.common_prefix_len([[1, 2], [1, 2]]) == 2


def test_counts_treat_unreadable_as_wrong():
    y = np.array([1, 1, 1, 0, 0, 0])
    pred = np.array([1, 0, -1, 0, 1, -1])
    assert py.counts(pred, y) == {"found": 1, "missed": 2, "extra": 2, "correctly_absent": 1, "unreadable": 2}
    assert py.balanced_accuracy(pred, y) == pytest.approx(0.5 * (1 / 3 + 1 / 3))
    assert py.balanced_accuracy(np.zeros(6, int), y) == pytest.approx(0.5)                    # "absent" to everything scores as a coin flip
    assert py.balanced_accuracy(np.array([1, 1]), np.array([1, 1])) is None


def test_vote_is_a_majority_and_unreadable_stays_unreadable():
    a, b, c = np.array([1, 1, 0, 0, -1]), np.array([1, 0, 0, 1, 1]), np.array([0, 1, 0, 1, 1])
    assert py.vote([a, b, c]).tolist() == [1, 1, 0, 1, -1]


def _data(gain: float, n=120, seed=3):
    rng = np.random.default_rng(seed)
    patients = [f"valid_{i // 2}" for i in range(n)]
    labels = {k: (rng.random(n) < 0.3).astype(int) for k in ("x", "y")}

    def answers(p_right):
        return {k: np.where(rng.random(n) < p_right, labels[k], 1 - labels[k]) for k in labels}
    return {"A": answers(0.7), "B": answers(0.7 + gain)}, labels, patients


def test_rule_met_only_for_a_clear_gain():
    preds, labels, patients = _data(0.28)
    r = py.analyse(preds, labels, patients, AN, ["B"])
    assert r["sets"]["B"]["rule_met"] and r["rule_met_by"] == ["B"] and r["sets"]["B"]["interval_95"][0] > 0 and r["patients"] == 60
    preds, labels, patients = _data(0.0)
    r = py.analyse(preds, labels, patients, AN, ["B"])
    assert not r["sets"]["B"]["rule_met"] and r["decision"].startswith("set A stays")


def test_unreadable_answers_block_the_rule_and_a_drop_in_one_condition_blocks_it():
    preds, labels, patients = _data(0.28)
    b = labels["x"].copy()
    b[np.flatnonzero(labels["x"] == 0)[:6]] = -1                              # perfect apart from 6 unreadable answers
    preds["B"]["x"] = b
    r = py.analyse(preds, labels, patients, AN, ["B"])
    assert r["sets"]["B"]["unreadable_share"] > 0.01 and not r["sets"]["B"]["rule_met"]
    preds, labels, patients = _data(0.28)
    preds["B"]["y"] = 1 - preds["A"]["y"]
    r = py.analyse(preds, labels, patients, AN, ["B"])
    assert r["sets"]["B"]["conditions_worse_by_more_than_0_05"] == ["y"] and not r["sets"]["B"]["rule_met"]


def test_conditions_with_few_positives_do_not_enter_the_main_measure():
    preds, labels, patients = _data(0.0)
    labels["rare"] = np.zeros(120, int)
    labels["rare"][:4] = 1
    for s in preds:
        preds[s]["rare"] = np.zeros(120, int)
    r = py.analyse(preds, labels, patients, AN, ["B"])
    assert "rare" not in r["conditions_interpretable"] and r["per_condition"]["rare"]["interpretable"] is False and r["per_condition"]["rare"]["sets"]["A"]["missed"] == 4


def test_error_split_needs_twice_the_error_rate_and_twenty_flagged():
    wrong = np.array([1] * 10 + [0] * 20 + [1] * 5 + [0] * 95, bool)
    flagged = np.array([1] * 30 + [0] * 100, bool)
    e = py.error_split(wrong, flagged)
    assert e["flagged"] == 30 and e["flagged_wrong"] == 10 and e["not_flagged_wrong"] == 5 and e["share_of_all_errors_flagged"] == pytest.approx(10 / 15) and e["useful_flag"]
    assert not py.error_split(wrong, np.array([1] * 12 + [0] * 118, bool))["useful_flag"]       # fewer than 20 flagged
    assert not py.error_split(np.array([1, 0] * 65, bool), flagged)["useful_flag"]              # same error rate in and out


def test_cutoff_maximises_found_plus_correctly_absent_and_takes_the_lowest_on_a_tie():
    s, y = np.array([0.1, 0.2, 0.3, 0.8, 0.9]), np.array([0, 0, 0, 1, 1])
    assert tr.fit_cutoff(s, y) == pytest.approx(0.55)
    assert tr.fit_cutoff(np.array([0.1, 0.4, 0.6, 0.9]), np.array([0, 1, 0, 1])) == pytest.approx(0.25)      # 0.25 and 0.75 tie: the lowest
    assert tr.fit_cutoff(s, np.zeros(5, int)) is None and tr.fit_cutoff(np.full(5, 0.5), y) is None


def test_left_out_answers_never_fit_on_the_answered_or_the_graded_patient():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.05, 0.95])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])                                   # the last two (graded patient) are labelled against the trend
    patients = ["p1", "p2", "p3", "p4", "p5", "p6", "g", "g"]
    pred, cuts, final = tr.left_out_answers(scores, y, patients, {"g"})
    assert final == pytest.approx(0.5) and pred.tolist() == [0, 0, 0, 1, 1, 1, 0, 1]        # 'g' is answered, but its labels never moved a cut-off
    assert all(0.25 <= c <= 0.75 for c in cuts) and len(cuts) == 7


def _judge(n_pos_found, n_pos_missed, n_extra, n_neg=60):
    y = np.array([1] * (n_pos_found + n_pos_missed) + [0] * n_neg)
    pred = np.array([1] * n_pos_found + [0] * n_pos_missed + [1] * n_extra + [0] * (n_neg - n_extra))
    return tr.judge(pred, y, [f"p{i}" for i in range(len(y))], {**tr.RULE, "resamples": 300})


def test_present_may_decide_needs_ten_calls_ninety_percent_right_and_a_lower_bound():
    assert _judge(30, 5, 0)["present_may_decide"]
    assert not _judge(8, 0, 0)["present_may_decide"]                          # right every time, but only 8 calls
    assert not _judge(14, 0, 6)["present_may_decide"]                         # 70 % right
    j = _judge(10, 0, 1)                                                      # 91 % right on 11 calls: the lower end decides
    assert j["share_of_present_calls_right"] == pytest.approx(10 / 11) and j["present_may_decide"] == (j["share_of_present_calls_right_lower"] > 0.75)


def test_absent_may_decide_cannot_be_passed_by_answering_absent_to_everything():
    always_absent = _judge(0, 4, 0, n_neg=96)                                 # 96 % of its "absent" answers are right, yet it found nothing
    assert always_absent["share_of_absent_calls_right"] == pytest.approx(0.96) and not always_absent["absent_may_decide"]
    assert not _judge(0, 20, 0)["absent_may_decide"]
    assert _judge(29, 1, 10)["absent_may_decide"] and not _judge(29, 1, 10)["present_may_decide"]
    assert _judge(4, 0, 0)["too_few_to_judge"] and not _judge(4, 0, 0)["absent_may_decide"]


def test_combine_only_listens_to_sources_that_may_decide():
    t = {"a": {"present_may_decide": True, "absent_may_decide": False}, "b": {"present_may_decide": False, "absent_may_decide": True}}
    assert tr.combine({"a": 1, "b": 1}, t) == "present" and tr.combine({"a": 0, "b": 0}, t) == "absent"
    assert tr.combine({"a": 1, "b": 0}, t) == "cannot tell" and tr.combine({"a": 0, "b": 1}, t) is None and tr.combine({"a": -1, "b": -1}, t) is None


def test_plans_exist_and_state_they_were_frozen_first():
    for name in ("PT02_nvreason_prompts_PLAN.md", "TR01_cutoffs_and_trust_rules_PLAN.md"):
        text = (ROOT / "docs" / "prompt_tests" / name).read_text(encoding="utf-8")
        assert "frozen 2026-11-26" in text and "Changes after freezing: none" in text
