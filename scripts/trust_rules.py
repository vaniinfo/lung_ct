"""COLIPRI cut-offs and trust rules across three answer sources (core env; no model is loaded).

Applies the frozen plan docs/prompt_tests/TR01_cutoffs_and_trust_rules_PLAN.md to answers that already exist:

  NV-Reason-CT   Yes / No of set A in a PT02 run (scripts/prompt_test_nvreason.py)
  COLIPRI        the stored score (set A of a PT01 run), turned into yes / no by a per-condition cut-off (part 1)
  Tool           present / absent / cannot tell of checklist_checklist_v3_9.json in the latest agent run of each scan

Part 1: cut-off per condition = the score that maximises (share found + share correctly absent - 1). Every patient is
answered with a cut-off fitted WITHOUT that patient, and the clinician-graded patients are never used for fitting.
Part 2: per source and condition, may its "present" decide, may its "absent" decide (bars in the plan).
Reference: CT-RATE report labels. Nothing is applied to any checklist; the output is a proposal.

Usage:
  .venv/Scripts/python.exe scripts/trust_rules.py --pt02-run runs/PT02_nvreason_prompts_<time> --colipri-run runs/PT01_colipri_prompts_20261126_073924
Writes runs/TR01_trust_rules_<time>/{cutoffs.csv,source_table.csv,summary.json,summary.md}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from prompt_test_yesno import UNREADABLE, balanced_accuracy, clinician_grades, counts, load_answers, patient_of, to_int  # noqa: E402

RULE = {"min_present_calls": 10, "min_positive_scans": 10, "share": 0.90, "lower_bound": 0.75, "resamples": 2000, "seed": 20261126}
TOOL_FILE = "checklist_checklist_v3_9.json"


def fit_cutoff(scores: np.ndarray, y: np.ndarray) -> float | None:
    """Score above which the answer is 'present': maximises share found + share correctly absent - 1 over the midpoints
    between neighbouring score values; the lowest such midpoint on a tie. None without both classes or two score values."""
    u = np.unique(scores)
    if not (y == 1).any() or not (y == 0).any() or len(u) < 2:
        return None
    cand = (u[:-1] + u[1:]) / 2.0
    pos, neg = scores[y == 1], scores[y == 0]
    j = np.array([(pos > c).mean() + (neg <= c).mean() - 1.0 for c in cand])
    return float(cand[int(np.argmax(j))])                 # argmax returns the first (lowest) maximum


def left_out_answers(scores: np.ndarray, y: np.ndarray, patients: list[str], never_fit: set[str]) -> tuple[np.ndarray, list[float], float | None]:
    """(answer per scan from a cut-off fitted without that scan's patient, the cut-offs used, the cut-off fitted on all allowed patients)."""
    pats = np.array(patients)
    allowed = ~np.isin(pats, list(never_fit))
    out, cuts = np.full(len(y), UNREADABLE), []
    for p in sorted(set(patients)):
        fit = allowed & (pats != p)
        c = fit_cutoff(scores[fit], y[fit])
        if c is not None:
            out[pats == p] = (scores[pats == p] > c).astype(int)
            cuts.append(c)
    return out, cuts, fit_cutoff(scores[allowed], y[allowed])


def _boot(stat, patients: list[str], rule: dict) -> float | None:
    """Lower end (2.5th percentile) of a statistic under resampling of patients; resamples where it is not defined are skipped."""
    uniq = sorted(set(patients))
    by = {p: np.array([i for i, q in enumerate(patients) if q == p]) for p in uniq}
    rng = np.random.default_rng(rule["seed"])
    vals = []
    for _ in range(rule["resamples"]):
        v = stat(np.concatenate([by[p] for p in rng.choice(uniq, len(uniq))]))
        if v is not None:
            vals.append(v)
    return float(np.percentile(vals, 2.5)) if vals else None


def judge(pred: np.ndarray, y: np.ndarray, patients: list[str], rule: dict = RULE) -> dict:
    """One source, one condition, on the scans that source answered. pred: 1 present, 0 absent, -1 neither (cannot tell / unreadable)."""
    def share_right(idx):          # of its "present" answers, the share on labelled scans
        m = pred[idx] == 1
        return float(y[idx][m].mean()) if m.any() else None

    def share_found(idx):          # of the labelled scans, the share it answered "present"
        m = y[idx] == 1
        return float((pred[idx][m] == 1).mean()) if m.any() else None

    full = np.arange(len(y))
    calls, pos, absent = int((pred == 1).sum()), int(y.sum()), pred == 0
    sr, sf = share_right(full), share_found(full)
    sr_lo, sf_lo = _boot(share_right, patients, rule), _boot(share_found, patients, rule)
    absent_right = float((y[absent] == 0).mean()) if absent.any() else None
    c = counts(pred, y)
    return {"scans": int(len(y)), "positive_scans": pos, "present_calls": calls, "absent_calls": int(absent.sum()), "neither": int((pred == UNREADABLE).sum()),
            "found": c["found"], "missed": pos - c["found"], "extra": int(((pred == 1) & (y == 0)).sum()),
            "share_of_present_calls_right": sr, "share_of_present_calls_right_lower": sr_lo, "share_found": sf, "share_found_lower": sf_lo,
            "share_of_absent_calls_right": absent_right,
            "present_may_decide": bool(calls >= rule["min_present_calls"] and sr is not None and sr >= rule["share"] and sr_lo is not None and sr_lo > rule["lower_bound"]),
            "absent_may_decide": bool(pos >= rule["min_positive_scans"] and sf is not None and sf >= rule["share"] and sf_lo is not None and sf_lo > rule["lower_bound"]
                                      and absent_right is not None and absent_right >= rule["share"]),
            "too_few_to_judge": bool(pos < rule["min_positive_scans"])}


def combine(answers: dict, table: dict) -> str | None:
    """Answers of the sources for one scan-condition -> 'present' / 'absent' / 'cannot tell' (trusted sources disagree) / None (no trusted answer)."""
    votes = set()
    for src, a in answers.items():
        if a == 1 and table[src]["present_may_decide"]:
            votes.add("present")
        if a == 0 and table[src]["absent_may_decide"]:
            votes.add("absent")
    return None if not votes else votes.pop() if len(votes) == 1 else "cannot tell"


def tool_answers(scans: list[str], keys: list[str]) -> dict:
    """scan -> key -> 1 / 0 / -1 from checklist_v3_9 of the latest agent run; scans without that file are absent from the result."""
    out = {}
    for s in scans:
        runs = sorted(p for p in (ROOT / "runs").glob(f"A1_agent_{s}_*") if (p / TOOL_FILE).exists())
        if runs:
            items = {i["key"]: i["status"] for i in json.loads((runs[-1] / TOOL_FILE).read_text(encoding="utf-8"))["items"]}
            out[s] = {k: to_int(items.get(k)) for k in keys}
    return out


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.0f} %"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pt02-run", required=True)
    ap.add_argument("--colipri-run", required=True)
    args = ap.parse_args()
    out = ROOT / "runs" / f"TR01_trust_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    grades = clinician_grades()
    graded_patients = {patient_of(s) for s in grades}

    # ---- sources, keyed by CT-RATE label ----
    pman = json.loads((ROOT / args.pt02_run / "run_manifest.json").read_text(encoding="utf-8"))
    pcfg = json.loads((ROOT / pman["prompt_sets"]).read_text(encoding="utf-8"))
    nv_key = {c["ct_rate_label"]: c["key"] for c in pcfg["conditions"] if c["ct_rate_label"]}
    ans, _, _ = load_answers(ROOT / args.pt02_run)
    nv_scans = sorted(s for s in pman["scans"] if all(s in ans["A"].get(k, {}) for k in nv_key.values()))
    ccfg = json.loads((ROOT / "configs/colipri_prompt_sets_v1.json").read_text(encoding="utf-8"))
    co_label = {f["key"]: f["label"] for f in ccfg["findings"] if f.get("ct_rate_label")}
    co = {}
    with open(ROOT / args.colipri_run / "scores.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["condition"] in co_label and r["A"] != "":
                co.setdefault(co_label[r["condition"]], {})[r["scan"]] = float(r["A"])
    tool = tool_answers(sorted(gt), list(nv_key.values()))
    labels_all = sorted(set(nv_key) | set(co))

    # ---- part 1: COLIPRI cut-offs, every patient answered without being fitted on ----
    cut_rows, co_pred = [], {}
    for lab in sorted(co):
        scans = sorted(co[lab])
        sc, y = np.array([co[lab][s] for s in scans]), np.array([int(lab in gt[s]["labels"]) for s in scans])
        pred, cuts, final = left_out_answers(sc, y, [patient_of(s) for s in scans], graded_patients)
        co_pred[lab] = dict(zip(scans, pred.tolist()))
        c = counts(pred, y)
        cut_rows.append({"condition": lab, "scans": len(scans), "positive_scans": int(y.sum()), "proposed_cutoff": None if final is None else round(final, 4),
                         "cutoff_lowest_fold": round(min(cuts), 4) if cuts else None, "cutoff_highest_fold": round(max(cuts), 4) if cuts else None,
                         "found": c["found"], "missed": c["missed"], "extra": c["extra"], "correctly_absent": c["correctly_absent"],
                         "balanced_accuracy_left_out": None if balanced_accuracy(pred, y) is None else round(balanced_accuracy(pred, y), 4),
                         "too_few_to_judge": bool(y.sum() < RULE["min_positive_scans"])})
    with open(out / "cutoffs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cut_rows[0]))
        w.writeheader()
        w.writerows(cut_rows)

    # ---- part 2: the rule, per source and condition ----
    sources = {"NV-Reason-CT": {lab: {s: to_int(ans["A"][k][s]) for s in nv_scans} for lab, k in nv_key.items()},
               "COLIPRI (cut-off)": co_pred,
               "Tool (checklist_v3_9)": {lab: {s: tool[s][k] for s in tool} for lab, k in nv_key.items()}}
    table, rows = {}, []
    for lab in labels_all:
        table[lab] = {}
        for src, d in sources.items():
            if lab not in d or not d[lab]:
                continue
            scans = sorted(d[lab])
            pred, y = np.array([d[lab][s] for s in scans]), np.array([int(lab in gt[s]["labels"]) for s in scans])
            table[lab][src] = judge(pred, y, [patient_of(s) for s in scans])
            rows.append({"condition": lab, "source": src, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in table[lab][src].items()}})
    with open(out / "source_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    proposal = {lab: {"present_decided_by": [s for s, j in t.items() if j["present_may_decide"]], "absent_decided_by": [s for s, j in t.items() if j["absent_may_decide"]],
                      "too_few_to_judge": all(j["too_few_to_judge"] for j in t.values())} for lab, t in table.items()}

    # ---- what the proposal would do to today's tool answers (description of the proposal, not a measure of its quality) ----
    change = {"scan_conditions_with_a_tool_answer": 0, "changed": 0, "changed_to_present": 0, "changed_to_absent": 0, "changed_to_cannot_tell": 0,
              "changed_and_now_equal_to_report_label": 0, "changed_and_now_different_from_report_label": 0, "conflicts_between_trusted_sources": 0}
    for lab, k in nv_key.items():
        for s in tool:
            a = {src: d[lab].get(s) for src, d in sources.items() if lab in d and s in d[lab]}
            new = combine(a, table[lab])
            old = {1: "present", 0: "absent"}.get(tool[s][k], "cannot tell")
            change["scan_conditions_with_a_tool_answer"] += 1
            change["conflicts_between_trusted_sources"] += new == "cannot tell"
            if new is not None and new != old:
                change["changed"] += 1
                change["changed_to_" + new.replace(" ", "_")] += 1
                truth = "present" if lab in gt[s]["labels"] else "absent"
                if new in ("present", "absent"):
                    change["changed_and_now_equal_to_report_label" if new == truth else "changed_and_now_different_from_report_label"] += 1

    # ---- clinician-graded scans, descriptive ----
    clin = {}
    for src, d in sources.items():
        c = {"graded": 0, "same": 0, "present_found": 0, "present_missed": 0, "absent_called_present": 0, "neither": 0}
        for scan, g in grades.items():
            for lab, k in nv_key.items():
                if k in g and lab in d and scan in d[lab]:
                    p, v = d[lab][scan], g[k]
                    c["graded"] += 1
                    c["neither"] += p == UNREADABLE
                    c["same"] += p != UNREADABLE and (p == 1) == (v == "present")
                    c["present_found"] += v == "present" and p == 1
                    c["present_missed"] += v == "present" and p != 1
                    c["absent_called_present"] += v == "absent" and p == 1
        clin[src] = {k: int(v) for k, v in c.items()}
    v311 = {lab: table[lab].get("NV-Reason-CT") for lab in ("Medical material", "Hiatal hernia", "Bronchiectasis") if lab in table}
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "plan": "docs/prompt_tests/TR01_cutoffs_and_trust_rules_PLAN.md", "rule": RULE,
               "pt02_run": args.pt02_run, "colipri_run": args.colipri_run, "tool_file": TOOL_FILE,
               "scans": {"NV-Reason-CT": len(nv_scans), "COLIPRI (cut-off)": len(next(iter(co.values()))), "Tool (checklist_v3_9)": len(tool)},
               "never_used_to_fit_cutoffs": sorted(graded_patients), "cutoffs": cut_rows, "table": table, "proposal": proposal, "effect_on_todays_tool_answers": change,
               "v3_11_reader_conditions_against_the_rule": v311, "clinician_graded_scans": clin,
               "not_judged": "MedGemma and Hulu-Med answered 20 scans only; below the numbers the rule asks for"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    md = ["# TR01: COLIPRI cut-offs and trust rules - summary", "", "Development scans; reference = CT-RATE report labels. A proposal, not a validated table. Nothing is applied.", "",
          "## Part 1: COLIPRI as a yes / no answer (each patient answered with a cut-off fitted without that patient)", "",
          "| Condition | Positive scans | Proposed cut-off | Cut-off range over folds | Found | Missed | Extra | Balanced accuracy | Note |", "|---|---|---|---|---|---|---|---|---|"]
    for r in cut_rows:
        md.append(f"| {r['condition']} | {r['positive_scans']} | {r['proposed_cutoff']} | {r['cutoff_lowest_fold']} to {r['cutoff_highest_fold']} | {r['found']} | {r['missed']} | {r['extra']} | "
                  f"{r['balanced_accuracy_left_out']} | {'too few to judge' if r['too_few_to_judge'] else ''} |")
    md += ["", "## Part 2: may a source decide?", "", f"Bars: at least {RULE['min_present_calls']} present calls, {RULE['share'] * 100:.0f} % right, lower end above {RULE['lower_bound'] * 100:.0f} % "
           f"(present); at least {RULE['min_positive_scans']} positive scans, {RULE['share'] * 100:.0f} % found, lower end above {RULE['lower_bound'] * 100:.0f} %, and {RULE['share'] * 100:.0f} % of absent answers right (absent).", "",
           "| Condition | Source | Scans | Positive | Says present | Of those right (lower end) | Found (lower end) | Neither | Present may decide | Absent may decide |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lab in labels_all:
        for src, j in table[lab].items():
            md.append(f"| {lab} | {src} | {j['scans']} | {j['positive_scans']} | {j['present_calls']} | {pct(j['share_of_present_calls_right'])} ({pct(j['share_of_present_calls_right_lower'])}) | "
                      f"{pct(j['share_found'])} ({pct(j['share_found_lower'])}) | {j['neither']} | {'YES' if j['present_may_decide'] else 'no'} | {'YES' if j['absent_may_decide'] else 'no'} |")
    md += ["", "## Proposed trust table", "", "| Condition | 'Present' may be decided by | 'Absent' may be decided by |", "|---|---|---|"]
    md += [f"| {lab} | {', '.join(p['present_decided_by']) or 'nobody'} | {', '.join(p['absent_decided_by']) or 'nobody'}{' (too few positive scans to judge)' if p['too_few_to_judge'] else ''} |" for lab, p in proposal.items()]
    md += ["", "Effect on today's tool answers (a description of the proposal, chosen on these same scans): " + json.dumps(change), "",
           "## Against the clinician reviewer's grades (four scans, descriptive only)", "", "| Source | Graded answers | Same | Present found | Present missed | Absent called present | Neither |", "|---|---|---|---|---|---|---|"]
    md += [f"| {s} | {c['graded']} | {c['same']} | {c['present_found']} | {c['present_missed']} | {c['absent_called_present']} | {c['neither']} |" for s, c in clin.items()]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
