"""Export checklist answers of finished runs to one CSV (core env). Reads only; rewrites nothing.

One row per run and condition. The CURRENT assessment (`current_assessment`: present / suggested / absent / cannot tell /
cannot assess) and the provisional threshold result are separate columns; the latter is named
`provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT` so that a historical "none" cannot be read as "absent".

Usage: python scripts/export_checklist_csv.py --pages-file runs/test_logs/pages_58runs_with_v3_4.txt \
           --checklist-file checklist_checklist_v3_8.json --out runs/E16_.../checklist_v3_8_export.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.checklist import EXPORT_COLUMNS, export_row  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pages-file", required=True)
    ap.add_argument("--checklist-file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = [ROOT / Path(x.strip().replace("\\", "/")).parent for x in (ROOT / args.pages_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows, missing = [], []
    for run in runs:
        f = run / args.checklist_file
        if not f.exists():
            missing.append(run.name)
            continue
        cl = json.loads(f.read_text(encoding="utf-8"))
        cid = json.loads((run / "evidence.json").read_text(encoding="utf-8"))["case_id"]
        rows += [export_row(cid, cl["config_version"], it) for it in cl["items"]]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=EXPORT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", out, "| runs without the file:", missing)
    print(collections.Counter(r["current_assessment"] for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
