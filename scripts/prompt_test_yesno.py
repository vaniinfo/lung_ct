"""Score a prompt test of a model that answers present / absent in words (core env; no model is loaded).

Reads the answers collected by scripts/prompt_test_nvreason.py and the CT-RATE report labels, and applies the frozen
plan in the prompt-set config (docs/prompt_tests/PT02_nvreason_prompts_PLAN.md):

  * per set and condition: found / missed / extra / correctly absent, balanced accuracy
  * macro balanced accuracy over conditions with enough positive scans, with a paired patient bootstrap against the baseline
  * the pre-stated rule for "better"
  * wording stability: do the reworded questions agree, and does disagreement (or a middling Yes-score) mark wrong answers
  * descriptive only: answers against the clinician reviewer's grades on the graded scans

Usage:
  .venv/Scripts/python.exe scripts/prompt_test_yesno.py --run runs/PT02_nvreason_prompts_<time>
Writes into the run folder: summary.json, summary.md, per_condition.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
UNREADABLE = -1


def patient_of(scan: str) -> str:
    return scan.rsplit("_", 2)[0]                     # valid_301_a_2 -> valid_301


def auroc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """Share of (positive, negative) pairs ranked the right way; a tie counts half. None without both classes."""
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return None
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def counts(pred: np.ndarray, y: np.ndarray) -> dict:
    """pred: 1 present, 0 absent, -1 unreadable. An unreadable answer is wrong: missed if labelled, extra if not."""
    return {"found": int(((pred == 1) & (y == 1)).sum()), "missed": int(((pred != 1) & (y == 1)).sum()),
            "extra": int(((pred != 0) & (y == 0)).sum()), "correctly_absent": int(((pred == 0) & (y == 0)).sum()),
            "unreadable": int((pred == UNREADABLE).sum())}


def balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> float | None:
    c = counts(pred, y)
    pos, neg = c["found"] + c["missed"], c["extra"] + c["correctly_absent"]
    return None if not pos or not neg else 0.5 * (c["found"] / pos + c["correctly_absent"] / neg)


def macro_ba(preds: dict, labels: dict, keys: list[str], idx: np.ndarray) -> float | None:
    vals = [balanced_accuracy(preds[k][idx], labels[k][idx]) for k in keys]
    vals = [v for v in vals if v is not None]          # a condition without both classes in a resample is left out of that resample
    return float(np.mean(vals)) if vals else None


def vote(parts: list[np.ndarray]) -> np.ndarray:
    """Majority of the given answer arrays; unreadable in any part -> unreadable."""
    a = np.stack(parts)
    out = (a.sum(axis=0) * 2 > len(parts)).astype(int)
    out[(a == UNREADABLE).any(axis=0)] = UNREADABLE
    return out


def patient_groups(patients: list[str]) -> tuple[list[str], dict]:
    uniq = sorted(set(patients))
    return uniq, {p: np.array([i for i, q in enumerate(patients) if q == p]) for p in uniq}


def analyse(preds: dict, labels: dict, patients: list[str], an: dict, compared: list[str], baseline: str = "A") -> dict:
    """preds: set -> condition -> [n] in {1, 0, -1}; labels: condition -> [n] 0/1."""
    n = len(patients)
    full = np.arange(n)
    keys = [k for k in labels if int(labels[k].sum()) >= an["min_positive_scans_to_interpret"]]
    per = {k: {"positive_scans": int(labels[k].sum()), "interpretable": k in keys,
               "sets": {s: {**counts(preds[s][k], labels[k]), "balanced_accuracy": balanced_accuracy(preds[s][k], labels[k])} for s in preds}} for k in labels}
    uniq, by = patient_groups(patients)
    rng = np.random.default_rng(an["bootstrap"]["seed"])
    diffs = {s: [] for s in compared}
    for _ in range(an["bootstrap"]["resamples"]):
        idx = np.concatenate([by[p] for p in rng.choice(uniq, len(uniq))])
        b = macro_ba(preds[baseline], labels, keys, idx)
        for s in compared:
            v = macro_ba(preds[s], labels, keys, idx)
            if v is not None and b is not None:
                diffs[s].append(v - b)
    base = macro_ba(preds[baseline], labels, keys, full)
    total = n * len(labels)
    sets = {}
    for s in preds:
        m = macro_ba(preds[s], labels, keys, full)
        unread = sum(per[k]["sets"][s]["unreadable"] for k in labels) / total
        rec = {"macro_balanced_accuracy": m, "unreadable_share": unread,
               "found": sum(per[k]["sets"][s]["found"] for k in labels), "missed": sum(per[k]["sets"][s]["missed"] for k in labels),
               "extra": sum(per[k]["sets"][s]["extra"] for k in labels)}
        if s in compared:
            lo, hi = np.percentile(diffs[s], an["bootstrap"]["interval"])
            worse = [k for k in keys if per[k]["sets"][s]["balanced_accuracy"] < per[k]["sets"][baseline]["balanced_accuracy"] - 0.05]
            rec.update(difference_to_baseline=m - base, interval_95=[float(lo), float(hi)], conditions_worse_by_more_than_0_05=worse,
                       rule_met=bool(lo > 0 and not worse and unread <= an["max_unreadable_share"]))
        sets[s] = rec
    met = sorted([s for s in compared if sets[s]["rule_met"]], key=lambda s: -sets[s]["macro_balanced_accuracy"])
    return {"scans": n, "patients": len(uniq), "conditions_scored": len(labels), "conditions_interpretable": keys, "baseline": baseline,
            "per_condition": per, "sets": sets, "rule_met_by": met,
            "decision": (f"set {met[0]} is the proposed way of asking (confirmation on untouched scans still needed)" if met else f"set {baseline} stays: no set met the pre-stated rule")}


def error_split(wrong: np.ndarray, flagged: np.ndarray) -> dict:
    """Error rate inside and outside a flagged group of scan-conditions."""
    f, u = int(flagged.sum()), int((~flagged).sum())
    ef, eu = (float(wrong[flagged].mean()) if f else None), (float(wrong[~flagged].mean()) if u else None)
    return {"flagged": f, "flagged_wrong": int(wrong[flagged].sum()), "flagged_error_rate": ef, "not_flagged": u, "not_flagged_wrong": int(wrong[~flagged].sum()),
            "not_flagged_error_rate": eu, "share_of_all_errors_flagged": float(wrong[flagged].sum() / wrong.sum()) if wrong.sum() else None,
            "useful_flag": bool(f >= 20 and ef is not None and eu is not None and ef >= 2 * eu and ef > 0)}


def stability(preds: dict, scores: dict, labels: dict, parts: list[str], baseline: str = "A") -> dict:
    """Do the reworded questions agree, and are the answers they disagree on more often wrong?"""
    a = np.concatenate([preds[baseline][k] for k in labels])
    y = np.concatenate([labels[k] for k in labels])
    stack = np.stack([np.concatenate([preds[s][k] for k in labels]) for s in parts])
    split = ~(stack == stack[0]).all(axis=0)
    wrong = a != y
    sc = np.concatenate([scores[baseline][k] for k in labels])
    middling = (sc >= 0.05) & (sc <= 0.95)
    per = {k: int((~(np.stack([preds[s][k] for s in parts]) == preds[parts[0]][k]).all(axis=0)).sum()) for k in labels}
    return {"scan_conditions": int(len(a)), "baseline_wrong": int(wrong.sum()), "same_answer_in_all_wordings_share": float((~split).mean()),
            "split_per_condition": per, "by_disagreement": error_split(wrong, split), "by_middling_yes_score_0_05_to_0_95": error_split(wrong, middling)}


def load_answers(run: Path) -> tuple[dict, dict, list[str]]:
    """answers.jsonl -> (set -> key -> scan -> answer), (set -> key -> scan -> score), scans seen."""
    ans, sco, scans = {}, {}, []
    for line in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["key"] == "_report":
            continue
        ans.setdefault(r["set"], {}).setdefault(r["key"], {})[r["scan"]] = r["answer"]
        if r.get("score_yes") is not None:
            sco.setdefault(r["set"], {}).setdefault(r["key"], {})[r["scan"]] = r["score_yes"]
        if r["scan"] not in scans:
            scans.append(r["scan"])
    return ans, sco, scans


def to_int(a: str | None) -> int:
    return {"present": 1, "absent": 0}.get(a, UNREADABLE)


def clinician_grades() -> dict:
    """scan -> condition key -> present / absent, from the authenticated grade files (other answers are left out)."""
    out = {}
    for p in sorted((ROOT / "runs" / "grades_received").glob("grades_*.csv")):
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r["doctor_answer"] in ("present", "absent"):
                    out.setdefault(r["case_id"], {})[r["condition_key"]] = r["doctor_answer"]
    return out


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = ROOT / args.run
    man = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / man["prompt_sets"]).read_text(encoding="utf-8"))
    an, conds = cfg["analysis"], cfg["conditions"]
    ans, sco, seen = load_answers(run)
    asked = [s for s in cfg["sets"] if s in ans]
    scans = sorted(s for s in man["scans"] if all(s in ans[x].get(c["key"], {}) for x in asked for c in conds))
    not_scored = [s for s in man["scans"] if s not in scans]
    gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    labelled = [c for c in conds if c["ct_rate_label"]]
    labels = {c["key"]: np.array([int(c["ct_rate_label"] in gt[s]["labels"]) for s in scans]) for c in labelled}
    preds = {x: {c["key"]: np.array([to_int(ans[x][c["key"]][s]) for s in scans]) for c in conds} for x in asked}
    scores = {x: {c["key"]: np.array([sco[x][c["key"]].get(s, np.nan) for s in scans], float) for c in conds} for x in asked if x in sco}
    for name, d in cfg["derived_sets"].items():
        if all(x in preds for x in d["from"]):
            preds[name] = {c["key"]: vote([preds[x][c["key"]] for x in d["from"]]) for c in conds}
    compared = [s for s in cfg["compared_with_baseline"] if s in preds]
    lab_preds = {s: {k: preds[s][k] for k in labels} for s in preds}
    res = analyse(lab_preds, labels, [patient_of(s) for s in scans], an, compared)
    res["auroc_of_yes_score"] = {x: {k: auroc(scores[x][k], labels[k]) for k in labels} for x in scores}
    parts = cfg["derived_sets"]["V"]["from"]
    if all(x in preds for x in parts):
        res["stability"] = stability(lab_preds, scores, labels, parts)
    grades = clinician_grades()
    clin = {}
    for s in preds:
        c = {"same": 0, "graded": 0, "present_found": 0, "present_missed": 0, "absent_called_present": 0, "unreadable": 0}
        for scan, g in grades.items():
            if scan not in scans:
                continue
            i = scans.index(scan)
            for k, v in g.items():
                if k not in preds[s]:
                    continue
                p = int(preds[s][k][i])
                c["graded"] += 1
                c["unreadable"] += p == UNREADABLE
                c["same"] += (p == 1) == (v == "present") and p != UNREADABLE
                c["present_found"] += v == "present" and p == 1
                c["present_missed"] += v == "present" and p != 1
                c["absent_called_present"] += v == "absent" and p == 1
        clin[s] = {k: int(v) for k, v in c.items()}
    res["clinician_graded_scans"] = {"scans": sorted(s for s in grades if s in scans), "note": "descriptive only; four development scans", "sets": clin}
    res.update(run=args.run, plan=cfg["plan"], prompt_sets_sha256=man["prompt_sets_sha256"], scans_not_scored=not_scored,
               left_out_before_run=man["left_out"], not_scored_condition="ground_glass (no CT-RATE label)")
    (run / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with open(run / "per_condition.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "positive_scans", "interpretable", "set", "found", "missed", "extra", "correctly_absent", "unreadable", "balanced_accuracy", "auroc_of_yes_score"])
        for k, d in res["per_condition"].items():
            for s, c in d["sets"].items():
                w.writerow([k, d["positive_scans"], d["interpretable"], s, c["found"], c["missed"], c["extra"], c["correctly_absent"], c["unreadable"], f3(c["balanced_accuracy"]),
                            f3(res["auroc_of_yes_score"].get(s, {}).get(k))])
    order = [s for s in ["A", "A2", "A3", "V", "B", "C", "D", "E"] if s in res["sets"]]
    idea = {**{k: v["idea"] for k, v in cfg["sets"].items()}, **{k: v["idea"] for k, v in cfg["derived_sets"].items()}}
    md = [f"# {run.name}: summary", "", f"{res['scans']} scans, {res['patients']} patients, {res['conditions_scored']} conditions with a CT-RATE label, "
          f"{len(res['conditions_interpretable'])} with at least {an['min_positive_scans_to_interpret']} positive scans.",
          "Development scans; reference = CT-RATE report labels. A choice of method, not a performance figure.", "",
          "| Set | Idea | Macro balanced accuracy | Difference to A | 95 % interval | Found | Missed | Extra | Unreadable | Rule met |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in order:
        r = res["sets"][s]
        d = f"{r['difference_to_baseline']:+.3f}" if "difference_to_baseline" in r else ""
        iv = f"{r['interval_95'][0]:+.3f} to {r['interval_95'][1]:+.3f}" if "interval_95" in r else ""
        md.append(f"| {s} | {idea[s]} | {f3(r['macro_balanced_accuracy'])} | {d} | {iv} | {r['found']} | {r['missed']} | {r['extra']} | {r['unreadable_share'] * 100:.1f} % | "
                  f"{'yes' if r.get('rule_met') else 'no' if 'rule_met' in r else ''} |")
    md += ["", f"**Decision: {res['decision']}.**", "", "Balanced accuracy per condition (found / missed / extra in brackets for A):", "",
           "| Condition | Positive scans | " + " | ".join(order) + " | Note |", "|---|---|" + "---|" * (len(order) + 1)]
    for k, d in res["per_condition"].items():
        a = d["sets"]["A"]
        md.append(f"| {k} | {d['positive_scans']} | " + " | ".join(f3(d["sets"][s]["balanced_accuracy"]) + (f" ({a['found']} / {a['missed']} / {a['extra']})" if s == "A" else "") for s in order)
                  + f" | {'' if d['interpretable'] else 'too few to interpret'} |")
    if "stability" in res:
        st = res["stability"]
        md += ["", "## Wording stability", "", f"Same answer in all three wordings: {st['same_answer_in_all_wordings_share'] * 100:.1f} % of {st['scan_conditions']} scan-conditions. "
               f"Set A is wrong (against the report label) on {st['baseline_wrong']}.", "",
               "| Flag | Flagged | Wrong among flagged | Not flagged | Wrong among not flagged | Share of all errors flagged | Useful by the pre-stated test |", "|---|---|---|---|---|---|---|"]
        for name, key in (("The three wordings disagree", "by_disagreement"), ("Yes-score of A between 0.05 and 0.95", "by_middling_yes_score_0_05_to_0_95")):
            e = st[key]
            pct = lambda x: "n/a" if x is None else f"{x * 100:.0f} %"  # noqa: E731
            md.append(f"| {name} | {e['flagged']} | {e['flagged_wrong']} ({pct(e['flagged_error_rate'])}) | {e['not_flagged']} | {e['not_flagged_wrong']} ({pct(e['not_flagged_error_rate'])}) | "
                      f"{pct(e['share_of_all_errors_flagged'])} | {'yes' if e['useful_flag'] else 'no'} |")
    cg = res["clinician_graded_scans"]
    md += ["", f"## Against the clinician reviewer's grades ({len(cg['scans'])} scans, descriptive only)", "",
           "| Set | Graded answers | Same | Present found | Present missed | Absent called present |", "|---|---|---|---|---|---|"]
    md += [f"| {s} | {c['graded']} | {c['same']} | {c['present_found']} | {c['present_missed']} | {c['absent_called_present']} |" for s, c in ((s, cg["sets"][s]) for s in order)]
    md += ["", "Scans not scored: " + (", ".join(not_scored) or "none") + ". Left out before the run: " + (", ".join(x["scan"] for x in man["left_out"]) or "none") + "."]
    (run / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
