"""Check that the location-first update (checklist_v3_4) changed no measurement and no decision (core env).

Three comparisons over the finished agent runs listed in the pages files:
  A. Code change: the v3_3 checklist built by the CURRENT code is identical, field for field, to the v3_3 checklist
     file written before the update (snapshot folder).
  B. Config change: v3_4 vs v3_3 (both built by the current code from the same evidence): status, display, measured
     value, basis, flags, classifier score and the older 'location' string must be equal for every condition.
     Statements and evidence lists may differ; every difference is listed.
  C. Evidence: the row-by-row recomputation written by scripts/add_lobe_distribution.py (verification.json) must
     show zero differing rows and no failed run, and every run must hold the per-lobe rows it can have.
Also re-scores both versions against the CT-RATE report labels (found / missed / extra / cannot tell).

Usage: python scripts/verify_location_update.py --pages-file runs/test_logs/pages_test20b_v3.txt ... \
           --snapshot runs/test_logs/snapshot_before_v3_4 --lobe-verification runs/E12_lobe_distribution_<time> [more]
Writes runs/E13_location_update_verification_<time>/{verification.json,verification.md,statements_changed.csv}.
Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.checklist import build_checklist, load_checklist_config  # noqa: E402

DECISION_FIELDS = ("status", "display", "measured_value", "basis", "flags", "classifier_p", "location", "readers")
NEW_ROWS = ("E83", "E89", "E91")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pages-file", nargs="+", required=True)
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--lobe-verification", nargs="+", required=True, type=Path)
    args = ap.parse_args()
    v33 = load_checklist_config(ROOT / "configs" / "checklist_v3_3.json")
    v34 = load_checklist_config(ROOT / "configs" / "checklist_v3_4.json")
    gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    runs = []
    for pf in args.pages_file:
        runs += [ROOT / Path(x.strip().replace("\\", "/")).parent for x in (ROOT / pf).read_text(encoding="utf-8").splitlines() if x.strip()]
    problems, changed, score = [], [], {"v3_3": Counter(), "v3_4": Counter()}
    n_items = n_a_equal = n_b_equal = 0
    rows_new = Counter()
    loc_added = Counter()
    for run in runs:
        ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
        cid = ev["case_id"]
        readers = (json.loads((run / "readers.json").read_text(encoding="utf-8")) if (run / "readers.json").exists() else {}).get("parsed", {})
        cl = (ev.get("classifier") or {})
        hi, lo = cl.get("high_threshold", 0.7), cl.get("low_threshold", 0.3)
        a = build_checklist(ev, readers, v33, hi, lo)
        b = build_checklist(ev, readers, v34, hi, lo)
        # A. current code vs the file written before the update
        old = json.loads(((ROOT / args.snapshot) / f"{run.name}__v3_3.json").read_text(encoding="utf-8"))
        if old["items"] != a["items"] or old["counts"] != a["counts"]:
            diff = [i["key"] for i, j in zip(old["items"], a["items"]) if i != j]
            problems.append(f"A {cid}: v3_3 checklist differs from the pre-update file in {diff}")
        else:
            n_a_equal += 1
        # B. v3_4 vs v3_3
        ok = True
        for x, y in zip(a["items"], b["items"]):
            n_items += 1
            for f in DECISION_FIELDS:
                if x.get(f) != y.get(f):
                    ok = False
                    problems.append(f"B {cid} {x['key']}: {f} differs ({x.get(f)!r} -> {y.get(f)!r})")
            extra_ev = [e for e in y["evidence"] if e not in x["evidence"]]
            if [e for e in x["evidence"] if e not in y["evidence"]] or any(e not in NEW_ROWS for e in extra_ev):
                ok = False
                problems.append(f"B {cid} {x['key']}: evidence list changed beyond the new rows")
            if x.get("statement") != y.get("statement"):
                kind = "location" if y.get("location_text") else "wording (denominator)"
                changed.append({"case_id": cid, "condition": x["key"], "status": x["status"], "kind": kind,
                                "v3_3": x.get("statement"), "v3_4": y.get("statement"),
                                "lobe_assignment_uncertain": (y.get("location_report") or {}).get("lobe_assignment_uncertain", "")})
            if y.get("location_text"):
                loc_added[(x["key"], x["status"])] += 1
        if a["counts"] != b["counts"]:
            ok = False
            problems.append(f"B {cid}: status counts differ")
        n_b_equal += ok
        # scores against report labels (text-derived; weak reference) - must be equal by construction
        labels = set(gt[cid]["labels"]) if cid in gt else None
        for name, chk in (("v3_3", a), ("v3_4", b)):
            for it in chk["items"]:
                if labels is None or not it.get("ct_rate_label"):
                    continue
                pos = it["ct_rate_label"] in labels
                s = it["status"]
                score[name]["found" if s == "present" and pos else "extra" if s == "present" else
                            "cannot tell / cannot assess" if s in ("cannot tell", "cannot assess") else
                            "missed" if pos else "correct absent"] += 1
        ids = {r["id"] for r in ev["rows"]}
        for nid in NEW_ROWS:
            rows_new[nid] += nid in ids
        if "E83" not in ids or "E89" not in ids:
            problems.append(f"C {cid}: per-lobe rows E83 / E89 missing")
    if score["v3_3"] != score["v3_4"]:
        problems.append("scores against report labels differ between v3_3 and v3_4")
    # C. evidence recomputation records (the newest record per case wins)
    rec = {}
    for d in args.lobe_verification:
        v = json.loads(((ROOT / d) / "verification.json").read_text(encoding="utf-8"))
        for r in v["records"]:
            rec[r["case_id"]] = r
    compared = sum(r.get("rows_compared", 0) for r in rec.values())
    different = sum(r.get("rows_different", 0) for r in rec.values())
    failed = [c for c, r in rec.items() if r["status"] != "ok"]
    missing = [json.loads((r / "evidence.json").read_text(encoding="utf-8"))["case_id"] for r in runs]
    missing = [c for c in missing if c not in rec]
    if different or failed or missing:
        problems.append(f"C evidence: {different} rows differ, failed {failed}, not verified {missing}")
    out = ROOT / "runs" / f"E13_location_update_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    kinds = Counter(c["kind"] for c in changed)
    res = {"runs": len(runs), "conditions_compared": n_items, "A_v3_3_identical_to_pre_update_file": n_a_equal,
           "B_runs_with_identical_decisions": n_b_equal, "decision_fields": DECISION_FIELDS,
           "C_evidence_rows_recomputed": compared, "C_evidence_rows_different": different, "C_failed": failed,
           "new_rows_present": dict(rows_new), "statements_changed": len(changed), "statements_changed_by_kind": dict(kinds),
           "location_statements_by_condition_and_status": {f"{k[0]} / {k[1]}": n for k, n in sorted(loc_added.items())},
           "lobe_assignment_flagged_uncertain": sum(1 for c in changed if c["lobe_assignment_uncertain"] is True),
           "scores_vs_report_labels": {k: dict(v) for k, v in score.items()}, "problems": problems, "passed": not problems}
    (out / "verification.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    if changed:
        with open(out / "statements_changed.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(changed[0].keys()))
            w.writeheader()
            w.writerows(changed)
    s3 = score["v3_3"]
    md = [f"# Location-first update (checklist_v3_4): verification - {'PASSED' if not problems else 'FAILED'}", "",
          f"{len(runs)} finished runs, {n_items} condition answers.", "",
          "| check | result |", "|---|---|",
          f"| A. v3_3 checklist from the current code = file written before the update | {n_a_equal} of {len(runs)} runs identical |",
          f"| B. v3_4 vs v3_3: status, display, measured value, basis, flags, classifier score, old location string | {n_b_equal} of {len(runs)} runs identical |",
          f"| C. existing evidence rows recomputed from the CT and compared | {compared} rows, {different} different, {len(failed)} runs failed |",
          f"| New per-lobe rows present (E83 / E89 / E91) | {rows_new['E83']} / {rows_new['E89']} / {rows_new['E91']} of {len(runs)} runs (E91 only where pleural fluid was outlined) |",
          f"| Scores vs report labels, v3_3 and v3_4 | found {s3['found']}, missed {s3['missed']}, extra {s3['extra']}, correct absent {s3['correct absent']}, "
          f"cannot tell / assess {s3['cannot tell / cannot assess']}: {'identical' if score['v3_3'] == score['v3_4'] else 'DIFFERENT'} |",
          "", f"Statements whose text changed: {len(changed)} ({', '.join(f'{k}: {n}' for k, n in kinds.items())}).",
          "Location statements generated, by condition and answer: " + "; ".join(f"{k[0]} / {k[1]}: {n}" for k, n in sorted(loc_added.items())) + ".",
          f"Of these, flagged 'lobe assignment uncertain': {res['lobe_assignment_flagged_uncertain']}.", ""]
    md += [f"- PROBLEM {p}" for p in problems]
    (out / "verification.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
