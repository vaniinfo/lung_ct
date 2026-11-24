"""Add measurements to viewer pages built before they were shown there (no recomputation).

Adds (each only if missing): the headline number under every answer with its hover / tap details, and the
collapsible "Measurements behind the checklist" card.

Usage: python scripts/add_measurements_to_viewer.py --page runs/M1_meeting_pack_20261119/viewer_1_x.html
           --run runs/A1_agent_valid_301_a_2_20261118_152058 --checklist-file checklist_checklist_v3_2.json
The checklist file must be the one the page was built from (rows are matched in order).
Only the page text before the first <script> is changed; the scan, overlays and viewer code are untouched.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from viewer_measurements import ANSWER_CSS, ANSWER_MARK, CARD_CSS, MARK, answer_details, measurements_card  # noqa: E402

E = html.escape
ANCHOR = '<div class="card"><b>Report</b>'   # the card goes between the checklist table and the report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--checklist-file", default="checklist.json")
    ap.add_argument("--config", default=None, help="checklist config (default: configs/<config_version>.json)")
    args = ap.parse_args()
    page = args.page if args.page.is_absolute() else ROOT / args.page
    run = args.run if args.run.is_absolute() else ROOT / args.run
    ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
    chk = json.loads((run / args.checklist_file).read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / (args.config or f"configs/{chk['config_version']}.json")).read_text(encoding="utf-8"))
    conds = {c["key"]: c for c in cfg["conditions"]}
    rows = {r["id"]: r for r in ev["rows"]}
    s = page.read_text(encoding="utf-8")
    head, sep, tail = s.partition("<script>")
    if ev["case_id"] not in head:
        raise SystemExit(f"{page.name}: page is not for case {ev['case_id']} - wrong --run?")
    if "</style>" not in head:
        raise SystemExit(f"{page.name}: page layout not recognised (style block not found)")
    done = []

    if ANSWER_MARK not in head:
        if head.count("<tr class='") != len(chk["items"]):
            raise SystemExit(f"{page.name}: {head.count(chr(60) + 'tr class=')} checklist rows but "
                             f"{len(chk['items'])} items - wrong checklist file?")
        pos = 0
        for it in chk["items"]:
            cell = f"<td><span class='st {it['status'].replace(' ', '-')}'>{E(it.get('display', it['status']))}</span></td>"
            k = head.find(cell, pos)
            if k < 0:
                raise SystemExit(f"{page.name}: answer cell for '{it['label']}' not found in order - wrong checklist file?")
            new = cell[:-len("</td>")] + answer_details(it, conds.get(it["key"], {}), rows) + "</td>"
            head = head[:k] + new + head[k + len(cell):]
            pos = k + len(new)
        head = head.replace("</style>", ANSWER_CSS + "</style>", 1)
        done.append(f"numbers under {len(chk['items'])} answers")

    if MARK not in head:
        if head.count(ANCHOR) != 1:
            raise SystemExit(f"{page.name}: page layout not recognised (report card not found)")
        head = head.replace("</style>", CARD_CSS + "</style>", 1).replace(ANCHOR, measurements_card(ev) + chr(10) + " " + ANCHOR, 1)
        done.append("measurements card")

    if not done:
        print(f"already has measurements: {page.name}")
        return 0
    page.write_text(head + sep + tail, encoding="utf-8", newline="")
    print(f"{page.name}: added {' + '.join(done)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
