"""Metric handling fixed before the model comparison: empty predictions, missing references, verified-empty references."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from pleural_compare_metrics import case_metrics, mask_state  # noqa: E402


def _vol(n):
    a = np.zeros((10, 10, 10), bool)
    a.ravel()[:n] = True
    return a


def test_empty_prediction_on_nonempty_reference_is_dice_zero_and_stays_in():
    r = case_metrics(_vol(0), {"first": _vol(100)}, 0.001)
    assert r["reference_status"] == "expert outline" and r["first_dice"] == 0 and r["first_share_of_reference_outlined"] == 0
    assert r["predicted_voxels"] == 0 and r["mask_state_provisional_limits"] == "empty" and r["first_predicted_minus_reference_ml"] == -0.1


def test_missing_reference_gets_no_segmentation_score_and_is_not_a_negative():
    r = case_metrics(_vol(50), {}, 0.001)
    assert "not a confirmed negative" in r["reference_status"] and not any(k.endswith("_dice") for k in r)
    assert r["predicted_voxels"] == 50
    r0 = case_metrics(_vol(50), {"first": _vol(0)}, 0.001)          # an empty file without verification is not a verified negative
    assert "treated as missing" in r0["reference_status"] and not any(k.endswith("_dice") for k in r0)


def test_verified_empty_reference_has_no_dice_and_reports_the_prediction():
    r = case_metrics(_vol(0), {}, 0.001, verified_empty=True)
    assert r["reference_status"].startswith("verified empty") and not any(k.endswith("_dice") for k in r)       # never Dice 1 for empty on empty
    r2 = case_metrics(_vol(30), {}, 0.001, verified_empty=True)
    assert r2["predicted_ml_on_verified_empty_reference"] == 0.03 and not any(k.endswith("_dice") for k in r2)


def test_reviewer_files_stay_separate_and_volumes_are_unrounded():
    a, b = _vol(100), _vol(60)
    r = case_metrics(_vol(80), {"first": a, "second": b}, 0.0028610229)
    assert r["first_dice"] == 2 * 80 / 180 and r["second_dice"] == 2 * 60 / 140
    assert r["predicted_ml"] == 80 * 0.0028610229 and (mask_state(4.99), mask_state(5.0), mask_state(20.0), mask_state(0)) == ("under_5_mL", "5_to_20_mL", "20_mL_or_more", "empty")
