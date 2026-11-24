"""Primary automatic checks on one generated finding-mask pair (pure functions on numbers; I/O is in scripts/).

Primary checks (each returns 'flag', 'pass' or 'not_applicable' with a reason):
  empty_mask            the model named a finding and drew nothing
  side                  stated side vs where the outline lies (left / right lung, nearest-lobe assignment)
  lobe                  stated lobe(s) vs share of the outline in those lobes
  contradictory_output  the model's own outputs disagree: a negated statement with an outline, or a 'No' to the
                        fixed-condition question while its findings list states that condition
Attenuation is DESCRIPTIVE ONLY here (median, percentiles, share of voxels per band). No density rule is active.

'pass' means no tested inconsistency was found. It does not mean the finding is correct.
All thresholds are PROVISIONAL until the protocol is frozen.
"""
from __future__ import annotations

from lung_ct.audit.parse import parse_statement

CHECK_VERSION = "audit_checks_v0_draft"
PRIMARY = ("empty_mask", "side", "lobe", "contradictory_output")

THRESHOLDS = {  # provisional
    "min_assigned_share": 0.5,     # at least this share of the outline must be assignable to a lobe for side / lobe checks
    "side_min_share": 0.5,         # one-sided statement: flag if less than this share of the assigned outline is on that side
    "bilateral_min_share": 0.05,   # bilateral statement: flag if the smaller side holds less than this share
    "lobe_min_share": 0.5,         # flag if less than this share of the assigned outline is in the stated lobe(s)
    "nearest_lobe_max_mm": 20.0,   # voxels outside the lobes are assigned to the nearest lobe within this distance
}
# finding type (parser) -> fixed-condition question (CT-RATE label) it should agree with
TYPE_TO_CONDITION = {
    "nodule": ["Lung nodule"],
    "consolidation": ["Consolidation", "Lung opacity"],
    "ground_glass": ["Lung opacity"],
    "pleural_effusion": ["Pleural effusion"],
    "emphysema": ["Emphysema"],
    "fibrotic_scarring": ["Pulmonary fibrotic sequela"],
    "atelectasis": ["Atelectasis"],
    "bronchiectasis": ["Bronchiectasis"],
    "septal_thickening": ["Interlobular septal thickening"],
    "peribronchial_thickening": ["Peribronchial thickening"],
    "mosaic_attenuation": ["Mosaic attenuation pattern"],
}
HU_BANDS = (("below_-950", -1e9, -950.0), ("-950_to_-700", -950.0, -700.0), ("-700_to_-300", -700.0, -300.0),
            ("-300_to_100", -300.0, 100.0), ("above_100", 100.0, 1e9))


def _r(status: str, detail: str, **extra) -> dict:
    return {"status": status, "detail": detail, **extra}


def check_empty_mask(voxels: int) -> dict:
    return _r("flag", "outline has no voxels") if voxels == 0 else _r("pass", f"{voxels} voxels")


def _location_ready(parsed: dict, voxels: int, assigned_share: float | None) -> str | None:
    if voxels == 0:
        return "empty outline"
    if not parsed["location_checkable"]:
        return "; ".join(parsed["why_not_checkable"])
    if assigned_share is None or assigned_share < THRESHOLDS["min_assigned_share"]:
        return (f"only {0 if assigned_share is None else assigned_share:.0%} of the outline lies in or within "
                f"{THRESHOLDS['nearest_lobe_max_mm']:g} mm of a lung lobe")
    return None


def check_side(parsed: dict, voxels: int, lobe_share: dict | None, assigned_share: float | None) -> dict:
    """lobe_share: share of the ASSIGNED outline per lobe (RUL, RML, RLL, LUL, LLL), summing to 1."""
    why = _location_ready(parsed, voxels, assigned_share)
    if why:
        return _r("not_applicable", why)
    if parsed["side"] is None:
        return _r("not_applicable", "no side stated")
    right = sum(v for k, v in lobe_share.items() if k.startswith("R"))
    left = 1.0 - right
    if parsed["side"] == "bilateral":
        small = min(right, left)
        ok = small >= THRESHOLDS["bilateral_min_share"]
        return _r("pass" if ok else "flag", f"stated bilateral; outline {right:.0%} right, {left:.0%} left", right_share=round(right, 4))
    share = right if parsed["side"] == "right" else left
    ok = share >= THRESHOLDS["side_min_share"]
    return _r("pass" if ok else "flag", f"stated {parsed['side']}; {share:.0%} of the outline on that side", right_share=round(right, 4))


def check_lobe(parsed: dict, voxels: int, lobe_share: dict | None, assigned_share: float | None) -> dict:
    why = _location_ready(parsed, voxels, assigned_share)
    if why:
        return _r("not_applicable", why)
    if not parsed["lobes"]:
        return _r("not_applicable", "no lobe stated")
    if parsed["lobe_side_missing"]:
        return _r("not_applicable", "a lobe is named without its side")
    share = sum(lobe_share.get(k, 0.0) for k in parsed["lobes"])
    top = max(lobe_share, key=lobe_share.get)
    ok = share >= THRESHOLDS["lobe_min_share"]
    return _r("pass" if ok else "flag",
              f"stated {'+'.join(parsed['lobes'])}; {share:.0%} of the outline there (largest share: {top} {lobe_share[top]:.0%})",
              stated_share=round(share, 4), top_lobe=top)


def check_contradictory_output(parsed: dict, voxels: int, answers: dict | None) -> dict:
    """answers: the model's own fixed-condition answers for this scan, {condition: 'yes' | 'no' | 'unparsed'}."""
    reasons = []
    if parsed["negated"] and voxels > 0:
        reasons.append("negated statement with a non-empty outline")
    if answers is not None and not parsed["negated"]:
        for t in parsed["types"]:
            conds = TYPE_TO_CONDITION.get(t, [])
            if conds and all(answers.get(c) == "no" for c in conds):
                reasons.append(f"findings list states {t.replace('_', ' ')}, but the model answered No to: {', '.join(conds)}")
    if reasons:
        return _r("flag", "; ".join(reasons))
    if answers is None:
        return _r("not_applicable", "no fixed-condition answers for this scan")
    if not parsed["types"]:
        return _r("not_applicable", "finding type not recognised by the parser")
    return _r("pass", "no contradiction with the fixed-condition answers")


def describe_attenuation(hu_values) -> dict | None:
    """Descriptive statistics of the CT values inside the outline. No rule is applied to them."""
    import numpy as np
    v = np.asarray(hu_values, dtype=np.float32)
    if v.size == 0:
        return None
    p10, p50, p90 = (float(x) for x in np.percentile(v, [10, 50, 90]))
    return {"n_voxels": int(v.size), "median_hu": round(p50, 1), "p10_hu": round(p10, 1), "p90_hu": round(p90, 1),
            "share_by_band": {name: round(float(((v >= lo) & (v < hi)).mean()), 4) for name, lo, hi in HU_BANDS},
            "note": "descriptive only; no attenuation rule is active"}


def audit_finding(statement: str | None, voxels: int, lobe_share: dict | None, assigned_share: float | None,
                  answers: dict | None) -> dict:
    """All primary checks for one finding. Returns {'parsed', 'checks', 'any_primary_flag'}."""
    parsed = parse_statement(statement)
    checks = {
        "empty_mask": check_empty_mask(voxels),
        "side": check_side(parsed, voxels, lobe_share, assigned_share),
        "lobe": check_lobe(parsed, voxels, lobe_share, assigned_share),
        "contradictory_output": check_contradictory_output(parsed, voxels, answers),
    }
    return {"parsed": parsed, "checks": checks, "any_primary_flag": any(c["status"] == "flag" for c in checks.values()),
            "check_version": CHECK_VERSION, "thresholds": dict(THRESHOLDS)}


def case_level_answer_gaps(answers: dict, statements_parsed: list[dict]) -> list[str]:
    """Conditions the model answered Yes to that none of its generated findings states (reported per scan,
    separately; not a finding-level flag, because the findings prompt covers lung, pleura and airways only)."""
    stated = {c for p in statements_parsed if not p["negated"] for t in p["types"] for c in TYPE_TO_CONDITION.get(t, [])}
    in_scope = {c for cs in TYPE_TO_CONDITION.values() for c in cs}
    return sorted(c for c, a in answers.items() if a == "yes" and c in in_scope and c not in stated)
