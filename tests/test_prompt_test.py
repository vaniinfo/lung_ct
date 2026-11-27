"""Prompt test tooling (scripts/prompt_test_colipri.py): ranking measures, prompt averaging, the pre-stated rule, the frozen prompt sets."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import prompt_test_colipri as pt  # noqa: E402

CFG = json.loads((ROOT / "configs" / "colipri_prompt_sets_v1.json").read_text(encoding="utf-8"))
BASE = json.loads((ROOT / "configs" / "colipri_ct_v1.json").read_text(encoding="utf-8"))
AN = {**CFG["analysis"], "bootstrap": {**CFG["analysis"]["bootstrap"], "resamples": 200}}


def test_auroc_known_values_and_ties():
    y = np.array([1, 1, 0, 0])
    assert pt.auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 1.0
    assert pt.auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 0.0
    assert pt.auroc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5                 # all tied: a coin flip
    assert pt.auroc(np.array([0.9, 0.3, 0.3, 0.1]), y) == pytest.approx(0.875)  # one tie between a positive and a negative counts half
    assert pt.auroc(np.array([0.9, 0.8]), np.array([1, 1])) is None           # no negative scan: not defined


def test_average_precision_known_value():
    assert pt.average_precision(np.array([0.9, 0.8, 0.7, 0.1]), np.array([1, 0, 1, 0])) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert pt.average_precision(np.array([0.9, 0.1]), np.array([0, 0])) is None


def test_patient_of_groups_reconstructions():
    assert pt.patient_of("valid_1057_a_1") == pt.patient_of("valid_1057_a_2") == "valid_1057"


def test_side_embedding_normalises_each_sentence_before_averaging():
    e = pt.side_embedding(np.array([[10.0, 0.0], [0.0, 1.0]]))               # a long vector must not dominate
    assert e == pytest.approx(np.array([1.0, 1.0]) / np.sqrt(2))
    assert pt.side_embedding(np.array([[3.0, 4.0]])) == pytest.approx([0.6, 0.8])


def test_score_is_the_softmax_over_the_pair():
    img = pt.unit(np.array([[1.0, 2.0, 0.5], [0.2, -1.0, 0.3]]))
    neg, pos, t = pt.unit(np.array([0.3, 0.1, 0.9])), pt.unit(np.array([0.5, 0.8, 0.1])), 0.07
    logits = np.stack([img @ neg, img @ pos], axis=1) / t
    want = np.exp(logits[:, 1]) / np.exp(logits).sum(axis=1)
    assert pt.p_present(img, neg, pos, t) == pytest.approx(want)


def _data(better_gap=0.0):
    rng = np.random.default_rng(1)
    n = 80
    patients = [f"valid_{i // 2}" for i in range(n)]                          # two scans per patient
    labels = {"x": (np.arange(n) % 4 < 2).astype(int), "y": (np.arange(n) % 5 < 2).astype(int)}
    noise = {k: rng.normal(0, 1, n) for k in labels}
    a = {k: labels[k] * 0.5 + noise[k] for k in labels}
    b = {k: labels[k] * (0.5 + better_gap) + noise[k] for k in labels}
    return {"A": a, "B": b}, labels, patients


def test_rule_not_met_when_sets_are_equal_and_baseline_stays():
    scores, labels, patients = _data(0.0)
    r = pt.analyse(scores, labels, patients, AN)
    assert r["sets"]["B"]["difference_to_baseline"] == pytest.approx(0.0) and not r["sets"]["B"]["rule_met"]
    assert r["rule_met_by"] == [] and r["decision"].startswith("set A stays")
    assert r["scans"] == 80 and r["patients"] == 40


def test_rule_met_when_a_set_is_clearly_better():
    scores, labels, patients = _data(3.0)
    r = pt.analyse(scores, labels, patients, AN)
    b = r["sets"]["B"]
    assert b["difference_to_baseline"] > 0.1 and b["interval_95"][0] > 0 and b["rule_met"] and r["rule_met_by"] == ["B"]
    assert "confirmation on untouched scans" in r["decision"]


def test_a_large_drop_in_one_condition_blocks_the_rule():
    scores, labels, patients = _data(3.0)
    scores["B"]["y"] = -scores["A"]["y"]                                      # ranking reversed for one condition
    r = pt.analyse(scores, labels, patients, AN)
    assert r["sets"]["B"]["conditions_worse_by_more_than_0_05"] == ["y"] and not r["sets"]["B"]["rule_met"]


def test_conditions_with_few_positives_are_marked():
    scores, labels, patients = _data(0.0)
    labels["rare"] = np.zeros(80, int)
    labels["rare"][:4] = 1
    for s in scores:
        scores[s]["rare"] = np.linspace(1, 0, 80)
    r = pt.analyse(scores, labels, patients, AN)
    assert r["per_condition"]["rare"]["interpretable"] is False and r["per_condition"]["x"]["interpretable"] is True


def test_baseline_set_is_exactly_the_current_prompts():
    base = {f["key"]: f for f in BASE["findings"]}
    assert [f["key"] for f in CFG["findings"]] == [f["key"] for f in BASE["findings"]]
    for f in CFG["findings"]:
        assert f["A"] == {"positive": [base[f["key"]]["positive"]], "negative": [base[f["key"]]["negative"]]}
        assert f["B"]["positive"][0] == f["A"]["positive"][0] and f["C"]["positive"][0] == f["A"]["positive"][0]
        assert len(f["B"]["positive"]) == len(f["B"]["negative"]) == 4 and 2 <= len(f["C"]["positive"]) == len(f["C"]["negative"]) <= 4
        assert all(s.lower().startswith(("no ", "there is no", "there are no")) for s in f["B"]["negative"] + f["C"]["negative"])


def test_plan_settings_are_in_the_frozen_config():
    assert CFG["analysis"]["min_positive_scans_to_interpret"] == 10 and CFG["analysis"]["bootstrap"]["unit"] == "patient"
    assert CFG["native"]["exclude"] == "every patient of the test scans" and CFG["native"]["n_positive"] == CFG["native"]["n_negative"] == 50
    assert (ROOT / CFG["plan"]).exists()
