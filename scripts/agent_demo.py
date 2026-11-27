"""Grounded agent demo: run all pipeline steps for a few cases, each in its own environment.

  step 2  scripts/agent_classify.py      environments/colipri  (GPU)  [skip with --no-classify]
  steps 0,1,3,4  scripts/agent_run_case.py   core env (this one)
  step 5  scripts/agent_write_report.py  environments/vlm      (GPU)  [skip with --no-report]
  page    scripts/agent_render_review.py core env

GPU stages run one after another in separate processes (project rule). Readers (step 3) are
read from existing runs. Usage (Windows, Git Bash or cmd):
  .venv/Scripts/python.exe scripts/agent_demo.py --cases coronacases_006 coronacases_001
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def env_python(name: str) -> str:
    for p in (ROOT / "environments" / name / "Scripts" / "python.exe", ROOT / "environments" / name / "bin" / "python"):
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"environment '{name}' not found under environments/")


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode}): {' '.join(cmd[1:])}")


def latest(pattern: str) -> str:
    hits = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise SystemExit(f"no run matching {pattern}")
    return str(hits[-1].relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", nargs="+", help="case ids; omit with --all-from-qc")
    ap.add_argument("--all-from-qc", action="store_true",
                    help="every case in --qc-run that did not fail (e.g. passed the anatomy check)")
    ap.add_argument("--qc-run", default="runs/P1_qc_20cases_20261107_223515")
    ap.add_argument("--reader", action="append",
                    default=None, metavar="NAME=RUNDIR")
    ap.add_argument("--no-classify", action="store_true")
    ap.add_argument("--classifier-run", default=None, help="reuse an existing COLIPRI run instead of running step 2")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--no-checklist", action="store_true", help="old opacity-only output")
    ap.add_argument("--checklist-config", default=None, help="e.g. configs/checklist_v2.json (default: v3)")
    ap.add_argument("--no-readers", action="store_true", help="run without AI reader outputs")
    ap.add_argument("--pages-out", default="runs/test_logs/agent_demo_pages.txt",
                    help="where to list the review pages (one per line)")
    args = ap.parse_args()
    if args.all_from_qc:
        import json
        qc = json.loads((ROOT / args.qc_run / "run_manifest.json").read_text(encoding="utf-8"))
        args.cases = [c["case_id"] for c in qc["cases"] if c.get("status") != "failed"]
        print(f"{len(args.cases)} cases from {args.qc_run}")
    if not args.cases:
        raise SystemExit("no cases: give --cases or --all-from-qc")
    readers = [] if args.no_readers else (args.reader or ["medgemma=runs/P3_vlm_10cases_20261108_083032",
                                                           "hulumed=runs/P3_hulumed_10cases_20261115_065454"])
    me = sys.executable

    classifier = args.classifier_run
    if classifier is None and not args.no_classify:
        run([env_python("colipri"), "scripts/agent_classify.py", "--qc-run", args.qc_run, "--cases", *args.cases])
        classifier = latest(f"runs/P3_colipri_{len(args.cases)}cases_*")

    pages = []
    for cid in args.cases:
        cmd = [me, "scripts/agent_run_case.py", "--case", cid, "--qc-run", args.qc_run]
        for r in readers:
            cmd += ["--reader", r]
        if classifier:
            cmd += ["--classifier", classifier]
        if args.no_checklist:
            cmd += ["--no-checklist"]
        elif args.checklist_config:
            cmd += ["--checklist-config", args.checklist_config]
        run(cmd)
        agent_run = latest(f"runs/A1_agent_{cid}_*")
        if not args.no_report:
            run([env_python("vlm"), "scripts/agent_write_report.py", "--run", agent_run])
        if (ROOT / agent_run / "checklist.json").exists():
            run([me, "scripts/agent_render_checklist.py", "--run", agent_run])
            pages.append(str(Path(agent_run) / "review_checklist.html"))
        else:
            run([me, "scripts/agent_render_review.py", "--run", agent_run])
            pages.append(str(Path(agent_run) / "review.html"))  # native separators for "start"
    print("\nReview pages:\n  " + "\n  ".join(pages))
    (ROOT / "runs" / "test_logs").mkdir(parents=True, exist_ok=True)
    po = ROOT / args.pages_out
    po.parent.mkdir(parents=True, exist_ok=True)
    po.write_text("\n".join(pages), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
