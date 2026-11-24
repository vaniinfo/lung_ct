"""Collect the reviewers' grade CSVs (saved from the viewer pages) into one table and summary (core env).

Usage: python scripts/collect_grades.py --folder path/to/returned_grades
Writes runs/E7_reviewer_grades_<time>/{all_grades.csv, summary.md}. If a reviewer saved the same case
more than once, only the newest save is used.
Files from grading form v1 and v2 (location graded separately, optional staged step 1) can be mixed.
Agreement is with the agent's checklist; the doctor's answer is also compared with the CT-RATE report
labels where a condition has one.

Policy-labelled packs (M5 and later; CSV columns checklist_version, pack, agent_current_assessment, ...):
  - a grade belongs to the policy and pack of the page it was made on; saves are compared per
    (case, reviewer, policy, pack), and the summary has one table per policy and pack, never a pooled one;
  - 'suggested' (a positive answer from a predicted mask only) is counted apart from the agent's confirmed answers;
  - the column provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT is copied and never read as an answer;
  - a file whose checklist version, pack, case or run does not match the pack it claims is rejected;
  - automated-test exports are never mixed with reviewer files.
Folders without such files are collected exactly as before.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.audit import grade_import as gi  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", required=True, type=Path)
    ap.add_argument("--checklist-config", default="configs/checklist_v3_2.json")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default runs/E7_reviewer_grades_<time>)")
    args = ap.parse_args()
    rows = []
    mf = args.folder / "AUTHENTICATED_EXPORTS.json"   # when present, only authenticated exports are read
    listed = {x["file"]: x["sha256"] for x in json.loads(mf.read_text(encoding="utf-8"))["files"]} if mf.exists() else None
    for f in sorted(args.folder.glob("grades_*.csv")):
        if listed is not None:
            import hashlib
            if listed.get(f.name) != hashlib.sha256(f.read_bytes()).hexdigest():
                raise SystemExit(f"{f.name} is not an authenticated clinician export (see {mf.name})")
        with open(f, newline="", encoding="utf-8-sig") as fh:
            rows += [dict(r, source_file=f.name) for r in csv.DictReader(fh)]
    if not rows:
        raise SystemExit(f"no grades_*.csv files in {args.folder}")
    policy_mode = any(gi.is_policy_row(r) for r in rows)
    if policy_mode:      # files from policy-labelled packs: checked against the pack they claim, never pooled across policies
        problems = gi.check_no_test_mix(rows) + gi.check_policy_rows(rows, gi.load_pack_registry(ROOT))
        if problems:
            raise SystemExit("grade files rejected:\n  " + "\n  ".join(problems[:40]) + (f"\n  ... {len(problems) - 40} more" if len(problems) > 40 else ""))
        rows = gi.keep_latest_save(rows)
        for r in rows:
            r["review_policy"], r["review_pack"] = gi.review_policy(r), gi.review_pack(r)
    else:
        latest = {}
        for r in rows:
            k = (r["case_id"], r["reviewer"].strip().lower())
            if k not in latest or r["saved_utc"] > latest[k]:
                latest[k] = r["saved_utc"]
        rows = [r for r in rows if latest[(r["case_id"], r["reviewer"].strip().lower())] == r["saved_utc"]]
    ck = json.loads((ROOT / args.checklist_config).read_text(encoding="utf-8"))
    ct_label = {c["key"]: c.get("ct_rate_label") for c in ck["conditions"]}
    cj = ROOT / "data" / "raw" / "ct_rate" / "cases.json"
    gt = json.loads(cj.read_text(encoding="utf-8"))["cases"] if cj.exists() else {}

    per = defaultdict(lambda: defaultdict(int))
    vs_rep = defaultdict(int)
    graded = [r for r in rows if r["condition_key"] != "_overall" and r["doctor_answer"]]
    for r in graded:
        per[r["condition"]][r["agreement"] or "?"] += 1
        lab = ct_label.get(r["condition_key"])
        if lab and r["case_id"] in gt and r["doctor_answer"] in ("present", "absent"):
            pos = lab in gt[r["case_id"]]["labels"]
            vs_rep["agree" if (r["doctor_answer"] == "present") == pos else "differ"] += 1
    out = (args.out if args.out.is_absolute() else ROOT / args.out) if args.out else ROOT / "runs" / f"E7_reviewer_grades_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=bool(args.out))
    with open(out / "all_grades.csv", "w", newline="", encoding="utf-8") as fh:
        fields = []                      # grades_v1 and grades_v2 files can be mixed: v1 rows leave the v2 columns empty
        for r in rows:
            fields += [k for k in r if k not in fields]
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    cases = sorted({r["case_id"] for r in rows})
    revs = sorted({r["reviewer"] for r in rows})
    tot = defaultdict(int)
    for v in per.values():
        for k, n in v.items():
            tot[k] += n
    md = [f"# Reviewer grades: {len(cases)} cases, {len(revs)} reviewer(s) ({', '.join(revs)})", "",
          "| condition | agrees with agent | agent missed it | agent over-called | open |", "|---|---|---|---|---|"]
    for name, v in per.items():
        md.append(f"| {name} | {v['agrees with agent']} | {v['agent missed it']} | {v['agent over-called']} | {v['open']} |")
    md.append(f"| **all** | {tot['agrees with agent']} | {tot['agent missed it']} | {tot['agent over-called']} | {tot['open']} |")
    if policy_mode:      # one table per policy and pack; the pooled table above is replaced
        test = any(gi.is_test_row(r) for r in rows)
        md = [f"# Reviewer grades: {len(cases)} cases, {len(revs)} reviewer(s) ({', '.join(revs)})", ""]
        if test:
            md += ["**AUTOMATED TEST DATA. Not from a reviewer. Not evidence about any case.**", ""]
        md += ["Grades are grouped by the policy (checklist version) and pack of the page they were made on. Groups are never pooled.",
               "'Suggested' = the tool's positive answer rested on a predicted mask only; it is counted apart from the tool's confirmed answers.",
               "The provisional threshold result is not an answer of the tool and is not counted anywhere.", ""]
        for pol, pack in gi.policy_groups(rows):
            sub = [r for r in graded if gi.review_policy(r) == pol and gi.review_pack(r) == pack]
            pg = defaultdict(lambda: defaultdict(int))
            for r in sub:
                pg[r["condition"]][r["agreement"] or "?"] += 1
            cols = ["agrees with agent", "agent missed it", "agent over-called", "open", "suggested_reviewer_present", "suggested_reviewer_absent"]
            other = sorted({k for v in pg.values() for k in v} - set(cols))
            md += [f"## Policy {pol}, pack {pack}: {len({r['case_id'] for r in sub})} case(s), {len(sub)} graded condition(s)", "",
                   "| condition | agrees with agent | agent missed it | agent over-called | open | suggested, reviewer present | suggested, reviewer absent |"
                   + "".join(f" {o} |" for o in other), "|---|---|---|---|---|---|---|" + "---|" * len(other)]
            for name, v in pg.items():
                md.append(f"| {name} | " + " | ".join(str(v[c]) for c in cols + other) + " |")
            md.append("| **all** | " + " | ".join(str(sum(v[c] for v in pg.values())) for c in cols + other) + " |")
            md.append("")
    md += ["", f"Doctor vs CT-RATE report labels (present/absent answers on labelled conditions): "
               f"{vs_rep['agree']} agree, {vs_rep['differ']} differ.",
           "", "## Location (form grades_v2 only; finding and location are graded separately)",
           "A form-v1 grade has no location: it is listed as not recorded and is never counted as assessed.", "",
           "| case | condition | reviewer answer | reviewer location | tool's named lobes | tool flags lobe uncertain |", "|---|---|---|---|---|---|"]
    n_v1 = 0
    for r in graded:
        if (r.get("form_version") or "grades_v1") == "grades_v1":
            n_v1 += r["doctor_answer"] in ("present", "cannot_tell")
            continue
        if r.get("location_kind") and r.get("location_status") not in ("", "not_applicable", None):
            where = r["location_where"] if r["location_status"] == "stated" else r["location_status"].replace("_", " ")
            md.append(f"| {r['case_id']} | {r['condition']} | {r['doctor_answer']} | {where} | {r.get('agent_lobes_named') or '-'} | "
                      f"{r.get('agent_lobe_assignment_uncertain') or '-'} |")
    md += ["", f"Form-v1 grades with a present / can't tell answer and no recorded location: {n_v1}.", "", "## Comments"]
    for r in rows:
        if r["comment"].strip():
            md.append(f"- {r['case_id']} / {r['condition']} ({r['reviewer']}): {r['comment'].strip()}")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:len(per) + 6]))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
