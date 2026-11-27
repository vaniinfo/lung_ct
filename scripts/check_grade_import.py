"""Export -> re-import check for grade files of a policy-labelled pack (core env). Automated-test files only.

Takes grade CSVs exported by the pack's pages with the automated-test reviewer marker, runs the project's collector
(scripts/collect_grades.py) on them, and then on deliberately altered copies that the collector must refuse:
  wrong pack, wrong checklist version, checklist_version differing from checklist_config, wrong run, the provisional
  threshold result pushed in as the agent's answer, 'suggested' relabelled as 'present', an altered agreement, and a
  folder that mixes automated-test files with reviewer files.
Writes <out>/import_check.json. Nothing is written outside <out>.

Usage: python scripts/check_grade_import.py --exports <folder with grades_*.csv> [--exports-2 <folder>] --reviewer-file runs/grades_received/<one file> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.audit import grade_import as gi  # noqa: E402


def collect(folder: Path, out: Path) -> tuple[int, str]:
    if out.exists():
        shutil.rmtree(out)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_grades.py"), "--folder", str(folder), "--checklist-config", "configs/checklist_v3_8.json", "--out", str(out)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout + r.stderr).strip()


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "source_file"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exports", required=True)
    ap.add_argument("--exports-2", default=None, help="a second set of test exports (e.g. with the reviewer answering 'absent' on a suggested item)")
    ap.add_argument("--reviewer-file", required=True, help="one real reviewer file, copied (never changed) into the 'mixed' test folder")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    res = []

    def ok(what, cond, detail=""):
        res.append({"check": what, "pass": bool(cond), "detail": str(detail)[:500]})
        print(("PASS " if cond else "FAIL ") + what + (f"  [{str(detail)[:220]}]" if detail != "" else ""), flush=True)

    sets = [("set1", ROOT / args.exports)] + ([("set2", ROOT / args.exports_2)] if args.exports_2 else [])
    for name, src in sets:
        d = out / f"accepted_{name}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        files = sorted(src.glob("*.csv"))
        for f in files:
            shutil.copy(f, d / (f.name if f.name.startswith("grades_") else "grades_" + f.name))
        code, txt = collect(d, out / f"collected_{name}")
        ok(f"{name}: the collector imports the {len(files)} exported files", code == 0 and len(files) > 0, txt.splitlines()[-1] if txt else "")
        if code:
            continue
        allr = list(csv.DictReader(open(out / f"collected_{name}" / "all_grades.csv", encoding="utf-8")))
        src_rows = [r for f in sorted(d.glob("grades_*.csv")) for r in gi.read_grade_file(f)]
        ok(f"{name}: every exported row is in the collected table, with every new column unchanged",
           len(allr) == len(src_rows) and all(all(a.get(c, "") == b.get(c, "") for c in gi.POLICY_COLUMNS + ["doctor_answer", "agreement", "agent_status", "agent_answer"]) for a, b in zip(allr, src_rows)), f"{len(allr)} rows")
        ok(f"{name}: every collected row carries its policy and pack", all(r["review_policy"] == r["checklist_version"] and r["review_pack"] == r["pack"] and r["review_pack"] for r in allr))
        pe = [r for r in allr if r["condition_key"] == "pleural_effusion"]
        ok(f"{name}: pleural effusion: the agent's answer is 'suggested' or 'cannot tell', never 'present' / 'absent', whatever the threshold column says",
           all(r["agent_status"] in ("suggested", "cannot tell") for r in pe) and {r[gi.THRESHOLD_COLUMN] for r in pe} <= {"present", "undecided", "none"},
           sorted({(r["agent_status"], r[gi.THRESHOLD_COLUMN], r["doctor_answer"], r["agreement"]) for r in pe}))
        sg = [r for r in pe if r["agent_status"] == "suggested" and r["doctor_answer"] in ("present", "absent")]
        ok(f"{name}: a reviewer answer on a 'suggested' item is counted as suggested_reviewer_*, not as agreement with / over-call of a confirmed answer",
           bool(sg) and all(r["agreement"] == "suggested_reviewer_" + r["doctor_answer"] for r in sg), [(r["case_id"], r["doctor_answer"], r["agreement"]) for r in sg])
        md = (out / f"collected_{name}" / "summary.md").read_text(encoding="utf-8")
        line = next((x for x in md.splitlines() if x.startswith("| Pleural effusion |")), "")
        cells = [c.strip() for c in line.strip("|").split("|")]
        n_sp, n_sa = sum(r["agreement"] == "suggested_reviewer_present" for r in pe), sum(r["agreement"] == "suggested_reviewer_absent" for r in pe)
        ok(f"{name}: summary has one table for this policy and pack, with the suggested counts in their own columns and 0 'agrees' / 'over-called' for pleural effusion",
           md.count("## Policy ") == 1 and "AUTOMATED TEST DATA" in md and len(cells) >= 7 and cells[1] == "0" and cells[3] == "0" and cells[5] == str(n_sp) and cells[6] == str(n_sa), line)
        # the threshold column is never read as an answer: changing it changes nothing but that column
        alt = out / f"threshold_changed_{name}"
        if alt.exists():
            shutil.rmtree(alt)
        for f in sorted(d.glob("grades_*.csv")):
            rows = gi.read_grade_file(f)
            for r in rows:
                if r[gi.THRESHOLD_COLUMN]:
                    r[gi.THRESHOLD_COLUMN] = {"present": "none", "none": "present", "undecided": "none"}[r[gi.THRESHOLD_COLUMN]]
            write(rows, alt / f.name)
        code2, _ = collect(alt, out / f"collected_threshold_changed_{name}")
        md2 = (out / f"collected_threshold_changed_{name}" / "summary.md").read_text(encoding="utf-8") if code2 == 0 else ""
        ok(f"{name}: a file with altered provisional threshold values is refused (they no longer match the page), so the column cannot be used to change a result",
           code2 != 0 and md2 == "")

    base = sorted((out / "accepted_set1").glob("grades_*.csv"))
    first = next((f for f in base if any(r["agent_status"] == "suggested" for r in gi.read_grade_file(f))), base[0])

    def altered(tag: str, change) -> None:
        rows = gi.read_grade_file(first)
        change(rows)
        d = out / f"rejected_{tag}"
        if d.exists():
            shutil.rmtree(d)
        write(rows, d / "grades_altered_AUTOMATED_TEST_NOT_from_a_reviewer.csv")
        code, txt = collect(d, d / "_out")
        ok(f"rejected: {tag}", code != 0 and "grade files rejected" in txt and not (d / "_out" / "all_grades.csv").exists(), txt.splitlines()[1].strip() if len(txt.splitlines()) > 1 else txt)

    def setall(col, val):
        return lambda rows: [r.__setitem__(col, val) for r in rows]

    def on_pe(col, val):
        def f(rows):
            for r in rows:
                if r["condition_key"] == "pleural_effusion":
                    r[col] = val
        return f
    altered("pack changed to M4 (a pack that is not policy-labelled)", setall("pack", "M4"))
    altered("pack name unknown", setall("pack", "M9"))
    altered("checklist_version changed to checklist_v3_4 in both version columns", lambda rows: [r.update(checklist_version="checklist_v3_4", checklist_config="checklist_v3_4") for r in rows])
    altered("checklist_version differs from checklist_config", setall("checklist_version", "checklist_v3_7"))
    altered("run is not the run shown on the pack page", setall("run", "A1_agent_some_other_run"))
    altered("'suggested' relabelled as 'present' in agent_status", on_pe("agent_status", "present"))
    altered("'suggested' relabelled as 'present' in both assessment columns", lambda rows: [r.update(agent_status="present", agent_current_assessment="present") for r in rows if r["condition_key"] == "pleural_effusion"])
    altered("agreement on the suggested item rewritten to 'agrees with agent'", on_pe("agreement", "agrees with agent"))
    altered("'suggested' relabelled as 'present' consistently (both columns and the agreement)",
            lambda rows: [r.update(agent_status="present", agent_current_assessment="present",
                                   agreement=gi.expected_agreement(dict(r, agent_current_assessment="present")))
                          for r in rows if r["condition_key"] == "pleural_effusion"])
    altered("a mask-state item exported as 'absent' (threshold result pushed in as the answer)", lambda rows: [r.update(agent_status="absent", agent_current_assessment="absent") for r in rows if r["condition_key"] == "pleural_effusion"])
    d = out / "rejected_mixed_with_reviewer_file"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copy(first, d / first.name)
    shutil.copy(ROOT / args.reviewer_file, d / Path(args.reviewer_file).name)
    code, txt = collect(d, d / "_out")
    ok("rejected: automated-test export in one folder with a reviewer file", code != 0 and "mixed with reviewer files" in txt, txt.splitlines()[-1] if txt else "")
    n_fail = sum(not r["pass"] for r in res)
    (out / "import_check.json").write_text(json.dumps({"checks": len(res), "failed": n_fail, "results": res}, indent=1), encoding="utf-8")
    print(f"{len(res) - n_fail} of {len(res)} checks passed -> {out / 'import_check.json'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
