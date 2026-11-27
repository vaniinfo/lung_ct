"""TR02 (core env): cut-offs and trust rules fitted on one half of the calibration scans and confirmed on the other half.

Plan (frozen first): docs/prompt_tests/TR02_calibration_400_PLAN.md. Bars, cut-off rule and intervals are those of TR01
(scripts/trust_rules.py), unchanged. Sources: NV-Reason-CT set A answers and COLIPRI set A scores on the calibration scans.

  fit half      cut-off per condition from all fit scans; COLIPRI judged with answers from cut-offs fitted without the
                answered patient; every source and condition judged -> proposed trust table
  confirm half  looked at after the fit half is written to disk: frozen cut-offs, same bars
  confirmed     an entry that meets the bars in both halves
  TR01          its cut-offs and its one entry (COLIPRI "absent", consolidation) on all new scans
  pooled        the same figures on all scans: descriptive only

Usage:
  .venv/Scripts/python.exe scripts/trust_rules_confirm.py --nvreason-run runs/TR02_nvreason_calib400_<time> --colipri-run runs/TR02_colipri_calib400_<time>
Writes runs/TR02_calibration_<time>/{fit_half.json,table.csv,summary.json,summary.md}
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
from prompt_test_yesno import balanced_accuracy, counts, load_answers, patient_of, to_int  # noqa: E402
from trust_rules import RULE, fit_cutoff, judge, left_out_answers  # noqa: E402

V3_11 = ["Medical material", "Hiatal hernia", "Bronchiectasis"]
NV, CO = "NV-Reason-CT", "COLIPRI (cut-off)"


def arrays(d: dict, scans: list[str], gt: dict, lab: str):
    """(values, labels, patients) for the scans of this list that the source answered."""
    use = [s for s in scans if s in d]
    return np.array([d[s] for s in use]), np.array([int(lab in gt[s]["labels"]) for s in use]), [patient_of(s) for s in use]


def half_result(nv: dict, co: dict, scans: list[str], gt: dict, cutoffs: dict | None, rule: dict) -> tuple[dict, dict]:
    """One half. cutoffs None: fit them here (COLIPRI answers left-out); else apply the given ones. Returns (table lab -> source -> judge, cut-off info)."""
    table, cuts = {}, {}
    for lab in sorted(set(nv) | set(co)):
        table[lab] = {}
        if lab in nv:
            pred, y, pats = arrays(nv[lab], scans, gt, lab)
            if len(y):
                table[lab][NV] = judge(pred, y, pats, rule)
        if lab in co:
            sc, y, pats = arrays(co[lab], scans, gt, lab)
            if not len(y):
                continue
            if cutoffs is None:
                pred, fold_cuts, final = left_out_answers(sc, y, pats, set())
                cuts[lab] = {"cutoff": final, "lowest_fold": min(fold_cuts) if fold_cuts else None, "highest_fold": max(fold_cuts) if fold_cuts else None}
            else:
                c = cutoffs.get(lab)
                pred = (sc > c).astype(int) if c is not None else np.full(len(y), -1)
                cuts[lab] = {"cutoff": c}
            cuts[lab].update(positive_scans=int(y.sum()), **counts(pred, y), balanced_accuracy=balanced_accuracy(pred, y))
            table[lab][CO] = judge(pred, y, pats, rule)
    return table, cuts


def entries(table: dict) -> set:
    return {(lab, src, kind) for lab, t in table.items() for src, j in t.items() for kind in ("present", "absent") if j[f"{kind}_may_decide"]}


def confirmed_entries(fit: dict, confirm: dict) -> dict:
    a, b = entries(fit), entries(confirm)
    return {"confirmed": sorted(a & b), "fit_half_only": sorted(a - b), "confirm_half_only": sorted(b - a)}


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f} %"


def f2(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nvreason-run", required=True)
    ap.add_argument("--colipri-run", required=True)
    ap.add_argument("--cases", default="D:/datasets/ct_rate/calib400/cases.json")
    ap.add_argument("--tr01-run", default="runs/TR01_trust_rules_20261126_190105")
    ap.add_argument("--resamples", type=int, default=RULE["resamples"])
    args = ap.parse_args()
    rule = {**RULE, "resamples": args.resamples}
    gt = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    pman = json.loads((ROOT / args.nvreason_run / "run_manifest.json").read_text(encoding="utf-8"))
    pcfg = json.loads((ROOT / pman["prompt_sets"]).read_text(encoding="utf-8"))
    nv_key = {c["ct_rate_label"]: c["key"] for c in pcfg["conditions"] if c["ct_rate_label"]}
    ans, _, _ = load_answers(ROOT / args.nvreason_run)
    nv = {lab: {s: to_int(a) for s, a in ans["A"].get(k, {}).items() if s in gt} for lab, k in nv_key.items()}
    ccfg = json.loads((ROOT / "configs/colipri_prompt_sets_v1.json").read_text(encoding="utf-8"))
    co_label = {f["key"]: f["label"] for f in ccfg["findings"] if f.get("ct_rate_label")}
    co = {}
    with open(ROOT / args.colipri_run / "scores.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["condition"] in co_label and r["A"] != "" and r["scan"] in gt:
                co.setdefault(co_label[r["condition"]], {})[r["scan"]] = float(r["A"])
    answered = sorted({s for d in list(nv.values()) + list(co.values()) for s in d})
    part = {p: [s for s in answered if gt[s]["part"] == p] for p in ("fit", "confirm")}
    if {patient_of(s) for s in part["fit"]} & {patient_of(s) for s in part["confirm"]}:
        raise SystemExit("a patient is in both halves")
    out = ROOT / "runs" / f"TR02_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)

    # ---- fit half first, written to disk before the confirm half is read ----
    fit_table, fit_cuts = half_result(nv, co, part["fit"], gt, None, rule)
    frozen = {lab: c["cutoff"] for lab, c in fit_cuts.items()}
    (out / "fit_half.json").write_text(json.dumps({"scans": len(part["fit"]), "cutoffs": fit_cuts, "table": fit_table, "entries": sorted(entries(fit_table)),
                                                    "written_utc": datetime.now(timezone.utc).isoformat()}, indent=1), encoding="utf-8")
    con_table, con_cuts = half_result(nv, co, part["confirm"], gt, frozen, rule)
    conf = confirmed_entries(fit_table, con_table)

    # ---- TR01's proposals on all new scans ----
    tr01 = {}
    with open(ROOT / args.tr01_run / "cutoffs.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            tr01[r["condition"]] = float(r["proposed_cutoff"]) if r["proposed_cutoff"] not in ("", "None") else None
    tr01_table, tr01_cuts = half_result({}, co, answered, gt, tr01, rule)
    # ---- pooled (descriptive) and the three checklist_v3_11 conditions ----
    pool_table, pool_cuts = half_result(nv, co, answered, gt, None, rule)

    rows = []
    for name, table in (("fit", fit_table), ("confirm", con_table), ("all, TR01 cut-offs", tr01_table), ("all, pooled (descriptive)", pool_table)):
        for lab, t in table.items():
            for src, j in t.items():
                rows.append({"scans_used": name, "condition": lab, "source": src, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in j.items()}})
    with open(out / "table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = {"created_utc": datetime.now(timezone.utc).isoformat(), "plan": "docs/prompt_tests/TR02_calibration_400_PLAN.md", "rule": rule,
           "nvreason_run": args.nvreason_run, "colipri_run": args.colipri_run, "cases": args.cases, "tr01_run": args.tr01_run,
           "scans": {"in_cases_file": len(gt), "answered": len(answered), "fit": len(part["fit"]), "confirm": len(part["confirm"]),
                     "not_answered": sorted(set(gt) - set(answered))},
           "cutoffs": {"fit": fit_cuts, "confirm_with_frozen": con_cuts, "tr01_on_all": tr01_cuts, "pooled": pool_cuts},
           "table": {"fit": fit_table, "confirm": con_table, "tr01_on_all": tr01_table, "pooled": pool_table},
           "entries": {k: [list(e) for e in v] for k, v in conf.items()},
           "tr01_entry_colipri_absent_consolidation": tr01_table.get("Consolidation", {}).get(CO),
           "v3_11_reader_conditions": {lab: pool_table.get(lab, {}).get(NV) for lab in V3_11}}
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    md = ["# TR02: cut-offs and trust rules, fitted on one half and confirmed on the other - summary", "",
          f"{len(answered)} scans answered ({len(part['fit'])} fit, {len(part['confirm'])} confirm) of {len(gt)} selected. Reference = CT-RATE report labels. Nothing is applied.", "",
          "## Entries that meet the bars", "",
          f"Bars: at least {rule['min_present_calls']} present calls, {pct(rule['share'])} right, lower end above {pct(rule['lower_bound'])} (present); "
          f"at least {rule['min_positive_scans']} positive scans, {pct(rule['share'])} found, lower end above {pct(rule['lower_bound'])}, and {pct(rule['share'])} of absent answers right (absent).", ""]
    for title, key in (("Confirmed (both halves)", "confirmed"), ("Fit half only (not confirmed)", "fit_half_only"), ("Confirm half only (not proposed)", "confirm_half_only")):
        md.append(f"- **{title}:** " + ("; ".join(f"{src} '{kind}' for {lab}" for lab, src, kind in conf[key]) or "none"))
    md += ["", "## COLIPRI cut-offs", "",
           "| Condition | Positive fit / confirm | Cut-off (fit half) | Range over folds | Balanced accuracy fit (left-out) | Balanced accuracy confirm (frozen) | Found / missed / extra, confirm | TR01 cut-off | Balanced accuracy with TR01 cut-off, all scans |",
           "|---|---|---|---|---|---|---|---|---|"]
    for lab in sorted(fit_cuts):
        a, b, t = fit_cuts[lab], con_cuts.get(lab, {}), tr01_cuts.get(lab, {})
        md.append(f"| {lab} | {a['positive_scans']} / {b.get('positive_scans', 'n/a')} | {f2(a['cutoff']) if a['cutoff'] is None else format(a['cutoff'], '.4f')} | "
                  f"{'n/a' if a.get('lowest_fold') is None else format(a['lowest_fold'], '.4f') + ' to ' + format(a['highest_fold'], '.4f')} | {f2(a['balanced_accuracy'])} | {f2(b.get('balanced_accuracy'))} | "
                  f"{b.get('found', 'n/a')} / {b.get('missed', 'n/a')} / {b.get('extra', 'n/a')} | {'n/a' if t.get('cutoff') is None else format(t['cutoff'], '.4f')} | {f2(t.get('balanced_accuracy'))} |")
    md += ["", "## Sources against the bars, per half", "",
           "| Condition | Source | Half | Positive | Says present | Of those right (lower end) | Found (lower end) | Absent answers right | Present may decide | Absent may decide |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lab in sorted(fit_table):
        for src in (NV, CO):
            for name, table in (("fit", fit_table), ("confirm", con_table)):
                j = table.get(lab, {}).get(src)
                if j:
                    md.append(f"| {lab} | {src} | {name} | {j['positive_scans']} | {j['present_calls']} | {pct(j['share_of_present_calls_right'])} ({pct(j['share_of_present_calls_right_lower'])}) | "
                              f"{pct(j['share_found'])} ({pct(j['share_found_lower'])}) | {pct(j['share_of_absent_calls_right'])} | {'YES' if j['present_may_decide'] else 'no'} | "
                              f"{'YES' if j['absent_may_decide'] else 'too few' if j['too_few_to_judge'] else 'no'} |")
    j = res["tr01_entry_colipri_absent_consolidation"]
    md += ["", "## TR01's proposals on scans it never saw", ""]
    if j:
        md.append(f"COLIPRI 'absent' for consolidation with TR01's cut-off, all new scans: {j['positive_scans']} positive, found {pct(j['share_found'])} (lower end {pct(j['share_found_lower'])}), "
                  f"absent answers right {pct(j['share_of_absent_calls_right'])}: **{'meets' if j['absent_may_decide'] else 'does not meet'} the bar**.")
    md += ["", "## The three conditions where checklist_v3_11 lets NV-Reason-CT decide 'present' (all new scans)", "",
           "| Condition | Positive scans | Present calls | Labelled present | Lower end | Meets the bar |", "|---|---|---|---|---|---|"]
    for lab, j in res["v3_11_reader_conditions"].items():
        if j:
            md.append(f"| {lab} | {j['positive_scans']} | {j['present_calls']} | {j['found']} ({pct(j['share_of_present_calls_right'])}) | {pct(j['share_of_present_calls_right_lower'])} | {'YES' if j['present_may_decide'] else 'no'} |")
    md += ["", f"Scans selected but not answered by either source (failed download, failed check or failed run): {len(res['scans']['not_answered'])}.",
           "Pooled figures on all scans are in table.csv and summary.json, marked descriptive."]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
