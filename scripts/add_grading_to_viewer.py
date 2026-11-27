"""Add the grading panel to viewer pages that were built before it existed (no recomputation).

Usage: python scripts/add_grading_to_viewer.py --page runs/M1_meeting_pack_20261119/viewer_1_x.html \
           --run runs/A1_agent_valid_301_a_2_20261118_152058 --checklist-file checklist_checklist_v3_2.json
The checklist file must be the one the page was built from (rows are matched in order).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from viewer_grading import CSS, grading_script  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--checklist-file", default="checklist.json")
    args = ap.parse_args()
    page = args.page if args.page.is_absolute() else ROOT / args.page
    run = args.run if args.run.is_absolute() else ROOT / args.run
    chk = json.loads((run / args.checklist_file).read_text(encoding="utf-8"))
    ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
    s = page.read_text(encoding="utf-8")
    if "lungct_grades_" in s:
        print(f"already has grading: {page.name}")
        return 0
    head, sep, tail = s.partition("<script>")
    n_rows = head.count("<tr class='")
    if n_rows != len(chk["items"]):
        raise SystemExit(f"{page.name}: {n_rows} checklist rows but {len(chk['items'])} items - wrong checklist file?")
    head = head.replace("</style>", CSS + "</style>", 1)
    i = tail.rfind("</body>")
    tail = tail[:i] + grading_script(ev["case_id"], run.name, chk["config_version"], chk) + "\n" + tail[i:]
    page.write_text(head + sep + tail, encoding="utf-8")
    print(f"grading added: {page.name} ({n_rows} conditions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
