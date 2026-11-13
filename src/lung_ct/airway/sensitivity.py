"""Parameter sensitivity of airway tree metrics (one-at-a-time around the defaults).

A metric is rated per case from its values across all settings:
  * count-type proximal structure (branches per generation 0-3): "stable" only if every
    setting gives the same value, otherwise "unstable"
  * tortuosity metrics are rated on the excess over 1 (tortuosity - 1), the part that
    carries information: a 1.02 -> 1.05 change is small relative to 1.02 but 2.5x in excess
  * continuous / large-count metrics: relative range = (max - min) / |default|
      <= STABLE_REL  -> "stable"
      <= MODERATE_REL -> "moderate"
      otherwise       -> "unstable"
Thresholds are reporting conventions chosen before looking at results, not statistics.
"unstable" means the value depends materially on a processing choice and must not be
compared between patients without that caveat.
"""
from __future__ import annotations

import numpy as np

STABLE_REL = 0.05
MODERATE_REL = 0.15

# one-at-a-time grid; None = module default (smoothing default = coarsest input spacing)
GRID = {
    "prune_radius_factor": [1.0, 2.0],
    "prune_min_mm": [1.0, 4.0],
    "contract_radius_factor": [0.5, 1.5],
    "loop_perimeter_factor": [3.1416, 9.4248],
    "smoothing_x_coarsest_spacing": [0.5, 2.0],
}
EXACT_METRICS = ("gen0_branches", "gen1_branches", "gen2_branches", "gen3_branches")
EXCESS_METRICS = ("median_tortuosity_internal", "median_tortuosity_gen2_4")


def settings(coarsest_spacing_mm: float) -> list[tuple[str, object, dict]]:
    """[(parameter, value, params-dict)], first entry = defaults."""
    out = [("default", None, {})]
    for name, values in GRID.items():
        for v in values:
            if name == "smoothing_x_coarsest_spacing":
                out.append((name, v, {"path_smoothing_sigma_mm": v * coarsest_spacing_mm}))
            else:
                out.append((name, v, {name: v}))
    return out


def metrics_from(res) -> dict:
    s = res.summary
    per_gen = s["branches_per_generation"]
    g1 = {r["side"]: r for r in res.branches if r["generation"] == 1}
    root = [r for r in res.branches if r["kind"] == "root"]
    mid = [r for r in res.branches if r["in_tree"] and r["generation"] is not None
           and 2 <= r["generation"] <= 4]
    return {
        **{f"gen{g}_branches": per_gen.get(g, 0) for g in range(4)},
        "n_tree_branches": s["n_tree_branches"],
        "n_terminal_branches": s["n_terminal_branches"],
        "max_generation": s["max_generation"],
        "n_cycles": s["n_cycles"],
        "trachea_length_mm": root[0]["length_mm"] if root else None,
        "gen1_right_length_mm": g1.get("right", {}).get("length_mm"),
        "gen1_left_length_mm": g1.get("left", {}).get("length_mm"),
        "median_length_gen2_4_mm": float(np.median([r["length_mm"] for r in mid])) if mid else None,
        "total_centreline_mm": s["total_centreline_length_mm"],
        "median_tortuosity_internal": s["median_tortuosity_internal"],
        "median_tortuosity_gen2_4": float(np.median([r["tortuosity"] for r in mid
                                                     if r["tortuosity"]])) if mid else None,
    }


def rate(metric: str, values: list, default) -> tuple[str, float | None]:
    """-> (rating, relative_range). Missing values anywhere -> 'unstable'."""
    if default is None or any(v is None for v in values):
        return "unstable", None
    if metric in EXACT_METRICS:
        return ("stable" if len(set(values)) == 1 else "unstable"), None
    if metric in EXCESS_METRICS:
        values, default = [v - 1.0 for v in values], default - 1.0
    rel = (max(values) - min(values)) / abs(default) if default else float("inf")
    if rel <= STABLE_REL:
        return "stable", rel
    if rel <= MODERATE_REL:
        return "moderate", rel
    return "unstable", rel


# ---- reporting helpers (single source for per-case and aggregate statements) ----------

CENTRAL_METRICS = ("gen0_branches", "gen1_branches", "trachea_length_mm",
                   "gen1_right_length_mm", "gen1_left_length_mm")
CENTRAL_STATEMENT = ("Central-airway measurements were generally stable across tested "
                     "settings; anatomical accuracy remains unvalidated.")
PARAMETER_VS_ANATOMY = ("Parameter stability means a value changed little when processing "
                        "settings changed. It says nothing about anatomical accuracy, which "
                        "is unvalidated (no airway reference in this dataset).")
TORTUOSITY_STATEMENT = ("The current tortuosity implementation is too sensitive to "
                        "centreline smoothing for patient comparisons.")
SMOOTHING_PARAM = "smoothing_x_coarsest_spacing"
TORTUOSITY_METRICS = ("median_tortuosity_internal", "median_tortuosity_gen2_4")


def aggregate_ratings(stab_rows: list[dict]) -> dict:
    """{metric: {"stable": n, "moderate": n, "unstable": n, "unstable_cases": [...]}} from
    the per-case rows of stability_by_case.csv - the only source for aggregate counts."""
    agg = {}
    for r in stab_rows:
        a = agg.setdefault(r["metric"], {"stable": 0, "moderate": 0, "unstable": 0,
                                         "unstable_cases": []})
        a[r["rating"]] += 1
        if r["rating"] == "unstable":
            a["unstable_cases"].append(r["case_id"])
    return agg


def central_exceptions(stab_rows: list[dict]) -> list[dict]:
    """Per-case central-airway metrics rated unstable (the exceptions to CENTRAL_STATEMENT)."""
    return [r for r in stab_rows if r["metric"] in CENTRAL_METRICS and r["rating"] == "unstable"]


def central_summary(stab_rows: list[dict]) -> str:
    """Data-driven central-airway sentence; lists every exception with its range."""
    exc = central_exceptions(stab_rows)
    n_cases = len({r["case_id"] for r in stab_rows})
    txt = CENTRAL_STATEMENT
    if not exc:
        return txt + f" No central-airway metric was rated unstable in {n_cases} cases."
    parts = [f"{r['case_id']} {r['metric']} {float(r['min']):.0f}-{float(r['max']):.0f} mm "
             f"(default {float(r['default']):.1f}; most influential: {r['most_influential']})"
             if r["metric"].endswith("_mm") else f"{r['case_id']} {r['metric']}"
             for r in exc]
    return txt + " Exception(s): " + "; ".join(parts) + "."


def tortuosity_by_smoothing(long_rows: list[dict]) -> dict:
    """{metric: {setting_label: {"abs": (min, max), "excess": (min, max), "n": n}}} across
    cases, for the default and each smoothing setting (from sensitivity_long.csv rows)."""
    out = {}
    for m in TORTUOSITY_METRICS:
        per = {}
        for r in long_rows:
            if r["metric"] != m or r["result"] in ("", None):
                continue
            if r["param"] == "default":
                label = "default (1.0 x coarsest spacing)"
            elif r["param"] == SMOOTHING_PARAM:
                label = f"{float(r['value']):g} x coarsest spacing"
            else:
                continue
            per.setdefault(label, []).append(float(r["result"]))
        def mult(label):
            return 1.0 if label.startswith("default") else float(label.split()[0])
        out[m] = {k: {"abs": (min(v), max(v)), "excess": (min(v) - 1, max(v) - 1), "n": len(v)}
                  for k, v in sorted(per.items(), key=lambda kv: mult(kv[0]))}
    return out
