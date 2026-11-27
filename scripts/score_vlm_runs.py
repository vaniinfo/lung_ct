"""Score one or more VLM runs against reference answers from the expert masks.

Model-agnostic: any run directory laid out like scripts/run_medgemma.py output
(<run>/<case_id>/vlm_parsed.json when the response was schema-valid) can be scored, so
MedGemma, Qwen3-VL and grounded/ungrounded variants are compared on the same footing.

Usage (core env):
  python scripts/score_vlm_runs.py --run medgemma=runs/P3_vlm_10cases_20261108_083032 \
                                   --run qwen=runs/<qwen run>
  python scripts/score_vlm_runs.py --reference-only --cases coronacases_001 coronacases_002

By default only cases present in EVERY run are scored (a fair comparison); excluded cases
are listed in the manifest. A case present but without a valid response counts as
no_valid_output, never dropped.

Writes runs/E5_vlm_scores_<timestamp>/:
  reference_answers.json  per-case reference answers + the measurements behind them
  scores_per_case.csv     run, case, field, model value, reference value, outcome
  summary.csv             per run and field: counts and agreement rates
  run_manifest.json       provenance (git commit, rules hash, inputs)
Rules: configs/reference_answer_rules_v1.json (PROVISIONAL, exploratory).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lung_ct.evaluation.reference_answers import load_case_reference, load_rules  # noqa: E402
from lung_ct.evaluation.score_vlm import FIELDS, OUTCOMES, score_case, summarise  # noqa: E402


def git_commit() -> str:
    try:
        c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return c + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def run_cases(run_dir: Path) -> set[str]:
    return {p.parent.name for p in run_dir.glob("*/vlm_result.json")}


def load_parsed(run_dir: Path, cid: str) -> dict | None:
    p = run_dir / cid / "vlm_parsed.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=PATH",
                    help="VLM run directory to score (repeatable)")
    ap.add_argument("--cases", nargs="+", help="score exactly these cases")
    ap.add_argument("--rules", type=Path, default=None,
                    help="reference-answer rules file (default configs/reference_answer_rules_v1.json)")
    ap.add_argument("--reference-only", action="store_true",
                    help="only derive reference answers (needs --cases)")
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        p = (ROOT / path) if not Path(path).is_absolute() else Path(path)
        if not label or not p.is_dir():
            ap.error(f"bad --run {spec!r} (expected LABEL=existing_dir)")
        runs[label] = p
    if not runs and not args.reference_only:
        ap.error("give at least one --run, or --reference-only with --cases")
    if args.reference_only and not args.cases:
        ap.error("--reference-only needs --cases")

    per_run = {lab: run_cases(p) for lab, p in runs.items()}
    if args.cases:
        cases = sorted(args.cases)
    else:
        cases = sorted(set.intersection(*per_run.values())) if per_run else []
    union = set().union(*per_run.values()) if per_run else set()
    excluded = sorted(union - set(cases))
    if not cases:
        print("no cases to score")
        return 1

    rules = load_rules(str(args.rules if args.rules.is_absolute() else ROOT / args.rules)
                       if args.rules else None)
    out = ROOT / "runs" / f"E5_vlm_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)

    refs, ref_errors = {}, {}
    for cid in cases:
        try:
            refs[cid] = load_case_reference(cid, ROOT, rules)
            a = refs[cid]["answers"]
            print(f"[{cid}] reference: {a['opacity_presence']} {a['distribution']}")
        except Exception as e:  # reported, never silently omitted
            ref_errors[cid] = f"{type(e).__name__}: {e}"
            print(f"[{cid}] REFERENCE FAILED: {ref_errors[cid]}")
    (out / "reference_answers.json").write_text(json.dumps(
        {"rules_version": rules["rules_version"], "rules_sha256": rules["_sha256"],
         "status": rules["status"], "cases": refs, "errors": ref_errors}, indent=2))

    rows, summary_rows = [], []
    for lab, rdir in runs.items():
        scored = []
        for cid in cases:
            if cid not in refs:
                continue
            parsed = load_parsed(rdir, cid) if cid in per_run[lab] else None
            sc = score_case(parsed, refs[cid])
            scored.append(sc)
            for f in FIELDS:
                rows.append({"run": lab, "case_id": cid, "field": f,
                             "model": sc[f]["model"], "reference": sc[f]["reference"],
                             "outcome": sc[f]["outcome"],
                             "case_in_run": cid in per_run[lab]})
        for f, s in summarise(scored).items():
            summary_rows.append({"run": lab, "field": f, **s})

    if rows:
        with open(out / "scores_per_case.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        cols = ["run", "field", "n_cases", *OUTCOMES, "agreement_all", "agreement_committed"]
        with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(summary_rows)
        for r in summary_rows:
            ac = r["agreement_committed"]
            print(f"{r['run']:>12} {r['field']:<17} agree {r['agree']}/{r['n_cases']}"
                  f"  disagree {r['disagree']}  indet {r['indeterminate']}"
                  f"  no_output {r['no_valid_output']}"
                  f"  committed-rate {'-' if ac is None else f'{ac:.0%}'}")

    manifest = {
        "run_id": out.name,
        "provenance": {"finished_utc": datetime.now(timezone.utc).isoformat(),
                       "git_commit": git_commit(),
                       "rules_file": str(Path(rules["_path"]).relative_to(ROOT)),
                       "rules_sha256": rules["_sha256"]},
        "runs": {lab: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                 for lab, p in runs.items()},
        "cases_scored": [c for c in cases if c in refs],
        "cases_excluded_not_in_all_runs": excluded,
        "reference_errors": ref_errors,
        "note": ("Agreement with a non-exhaustive expert infection annotation under PROVISIONAL "
                 "definitions; exploratory, not accuracy against a clinical diagnosis."),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
