"""Patient-level splits and descriptive group comparison for airway measurements. No classifier, no threshold.

Rules built into this module
  * the unit of splitting is the PATIENT: every scan and reconstruction of a patient is in one split only
  * a patient with conflicting labels across scans is excluded and reported, never silently assigned
  * splits are deterministic (seed + patient id) and are written once; an existing split file is never changed
  * group comparison is a description of measurements in two labelled groups. It is not classification performance:
    no threshold is chosen, no sensitivity / specificity is computed here
  * the test split is not read unless the caller passes a frozen analysis plan (see scripts/airway_group_compare.py)
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

SPLITS = ("train", "validation", "test")
_CTRATE = re.compile(r"^(train|valid)_(\d+)_[a-z]_\d+$")


def patient_id_from_case(case_id: str) -> str:
    """CT-RATE ids are <split>_<patient>_<scan>_<reconstruction>: the patient is <split>_<patient>. Other ids: the case id."""
    m = _CTRATE.match(case_id)
    return f"{m.group(1)}_{m.group(2)}" if m else case_id


def patient_labels(cases: list[dict]) -> tuple[dict, list[str]]:
    """cases: [{case_id, label (0/1), patient_id (optional)}] -> ({patient: label}, [patients excluded for conflicting labels])."""
    seen: dict[str, set] = {}
    for c in cases:
        seen.setdefault(c.get("patient_id") or patient_id_from_case(c["case_id"]), set()).add(int(c["label"]))
    conflict = sorted(p for p, s in seen.items() if len(s) > 1)
    return {p: next(iter(s)) for p, s in seen.items() if len(s) == 1}, conflict


def assign_splits(labels: dict, seed: int, fractions=(0.6, 0.2, 0.2)) -> dict:
    """{patient: label} -> {patient: split}; stratified by label, deterministic in (seed, patient id)."""
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) < 0:
        raise ValueError("fractions must be three non-negative numbers that sum to 1")
    out = {}
    for lab in sorted(set(labels.values())):
        pts = sorted((p for p, v in labels.items() if v == lab), key=lambda p: hashlib.sha256(f"{seed}|{p}".encode()).hexdigest())
        n = len(pts)
        a = int(round(fractions[0] * n))
        b = a + int(round(fractions[1] * n))
        for i, p in enumerate(pts):
            out[p] = SPLITS[0] if i < a else SPLITS[1] if i < b else SPLITS[2]
    return out


def check_no_leakage(case_split: dict, case_patient: dict) -> None:
    """Raises if any patient has cases in more than one split."""
    by = {}
    for c, s in case_split.items():
        by.setdefault(case_patient[c], set()).add(s)
    bad = sorted(p for p, s in by.items() if len(s) > 1)
    if bad:
        raise ValueError(f"patient-level leakage: {bad[:5]} appear in more than one split")


def compare_groups(cases_vals: list[float], controls_vals: list[float]) -> dict:
    """Descriptive comparison of one measurement between two groups of PATIENTS.

    rank_biserial = 2 * U / (n1 * n2) - 1 in [-1, 1]: +1 means every labelled case is above every control. It describes
    separation of the measurement in this sample; it is not a classification metric and implies no threshold.
    p is the two-sided Mann-Whitney p value, uncorrected for the number of features compared.
    """
    a = np.asarray([v for v in cases_vals if v is not None and np.isfinite(v)], float)
    b = np.asarray([v for v in controls_vals if v is not None and np.isfinite(v)], float)
    r = {"n_cases": int(a.size), "n_controls": int(b.size), "median_cases": None, "median_controls": None, "median_difference": None,
         "rank_biserial": None, "p_mann_whitney_uncorrected": None}
    if a.size:
        r["median_cases"] = round(float(np.median(a)), 4)
    if b.size:
        r["median_controls"] = round(float(np.median(b)), 4)
    if a.size and b.size:
        r["median_difference"] = round(float(np.median(a) - np.median(b)), 4)
    if a.size >= 2 and b.size >= 2:
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(a, b, alternative="two-sided")
        r["rank_biserial"] = round(2.0 * float(u.statistic) / (a.size * b.size) - 1.0, 4)
        r["p_mann_whitney_uncorrected"] = float(u.pvalue)
    return r
