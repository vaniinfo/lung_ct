"""Model against report for every condition (core env): NV-Reason-CT and COLIPRI on the TR02 calibration scans.

Descriptive. Reads the TR02 scoring output (runs/TR02_calibration_<time>/summary.json, pooled table: all scans; COLIPRI answers there
come from cut-offs fitted without the answered patient). For each condition and model:
  agree present   model present, report mentions it          (found)
  model only      model present, report does not mention it  (extra; may be a finding the report left out)
  report only     report mentions it, model says absent      (missed)
  agree absent    neither
  plus: share of report mentions the model found, share of the model's "present" the report confirms, balanced accuracy.

Usage: .venv/Scripts/python.exe scripts/model_vs_report_table.py --tr02-run runs/TR02_calibration_<time>
Writes <tr02-run>/model_vs_report_all_conditions.{csv,md}
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def row(lab: str, src: str, j: dict) -> dict:
    tp, fn, fp = j["found"], j["missed"], j["extra"]
    tn = j["absent_calls"] - fn
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    prec = tp / (tp + fp) if tp + fp else None
    return {"condition": lab, "model": src, "scans": j["scans"], "report_mentions": j["positive_scans"], "agree_present": tp, "model_only": fp,
            "report_only": fn, "agree_absent": tn, "not_answered": j["neither"],
            "share_of_report_mentions_found": None if sens is None else round(sens, 3),
            "share_of_model_present_confirmed_by_report": None if prec is None else round(prec, 3),
            "balanced_accuracy": None if sens is None or spec is None else round((sens + spec) / 2, 3)}


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f} %"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tr02-run", required=True)
    args = ap.parse_args()
    run = ROOT / args.tr02_run
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    rows = [row(lab, src, j) for lab in sorted(s["table"]["pooled"]) for src, j in s["table"]["pooled"][lab].items()]
    with open(run / "model_vs_report_all_conditions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = [f"# Model against report, every condition ({s['scans']['answered']} calibration scans, descriptive)", "",
          "Reference = CT-RATE report labels. 'Model only' can be a real finding the report left out. COLIPRI answers use cut-offs fitted without the answered patient.", "",
          "| Condition | Model | Report mentions | Agree present | Model only | Report only | Agree absent | Report mentions found | Model 'present' confirmed by report | Balanced accuracy |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['condition']} | {r['model']} | {r['report_mentions']} | {r['agree_present']} | {r['model_only']} | {r['report_only']} | {r['agree_absent']} | "
                  f"{pct(r['share_of_report_mentions_found'])} | {pct(r['share_of_model_present_confirmed_by_report'])} | {r['balanced_accuracy']:.2f} |")
    (run / "model_vs_report_all_conditions.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
