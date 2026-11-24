"""Measurements in the interactive viewer pages: the headline number under each answer (details on
hover / tap) and the collapsible "Measurements behind the checklist" card.

Same rows as the static review_checklist.html page (agent_render_checklist.measurement_rows), so the
two pages cannot drift apart. Used by build_viewer_page.py and add_measurements_to_viewer.py.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_render_checklist import measurement_rows  # noqa: E402

E = html.escape
MARK = 'id="meascard"'
CARD_CSS = (".id{font-family:Consolas,monospace;color:#57606a;font-size:11px;white-space:nowrap}"
            "#meascard summary{cursor:pointer}#meascard table{margin-top:6px}")
ANSWER_CSS = (
       ".mv{position:relative;display:inline-block;margin-top:4px;font-size:11.5px;font-weight:600;color:#1f2328;"
       "cursor:help;border-bottom:1px dotted #8b949e;outline:none}.mv.none{font-weight:400;color:#66707a}"
       ".mvq{display:block;font-weight:400;color:#66707a}"
       ".mvpop{display:none;position:absolute;left:0;top:100%;z-index:30;width:330px;margin-top:4px;background:#1f2328;"
       "color:#f0f3f6;font-weight:400;font-size:11.5px;line-height:1.4;padding:8px 10px;border-radius:8px;"
       "white-space:normal;box-shadow:0 4px 14px rgba(0,0,0,.35);cursor:default}"
       ".mvpop b{color:#fff}.mvpop .l{display:block;margin-top:3px}.mvpop .e{font-family:Consolas,monospace;color:#9fc2ff}"
       ".mv:hover .mvpop,.mv:focus .mvpop{display:block}")
CSS = CARD_CSS + ANSWER_CSS
ANSWER_MARK = "class='mv"


def _value_text(r: dict) -> str:
    v = r["value"]
    if isinstance(v, dict):
        return ", ".join(f"{k} {x}" for k, x in v.items() if k in ("side", "zone", "leading_lobe") or "side" not in v)
    if isinstance(v, (int, float)) and r.get("unit") == "p":
        return f"{v:.2f} (0 to 1, uncalibrated)"
    return f"{v:g} {r.get('unit') or ''}".strip() if isinstance(v, (int, float)) else ("not available" if v is None else str(v))


def answer_details(item: dict, cond: dict, rows: dict) -> str:
    """HTML placed under the answer badge: the headline number (always visible) and a hover / tap box with the
    cut-offs, every evidence row of this condition with its id, and the reason for the answer.

    item = checklist item, cond = its entry in the checklist config, rows = evidence rows by id.
    Display only: nothing here changes an answer."""
    unit = cond.get("unit", "")
    basis = item.get("basis", "")
    mv = item.get("measured_value")
    qual = ""
    if mv is not None:
        head = f"{mv:g} {unit}".strip()
        if basis.startswith("lungs poorly inflated"):
            qual = "shallow breath"
        elif "grey zone" in basis:
            qual = "between the cut-offs" if cond.get("absent_below") is not None else "low, but cannot rule out"
            if cond.get("grey_label") and basis.startswith("measurement in grey zone"):
                qual = cond["grey_label"].format(present=cond.get("present_at_least"))
    elif item.get("status") == "cannot assess":
        head = "needs contrast"
    elif cond.get("measured_by"):
        head = "from the rows above" if item.get("key") == "lung_density" else "not measured on this scan"
    else:
        head = "no measuring tool"
    pop = []
    if cond.get("measured_by"):
        pop.append(f"<span class='l'><b>Measured by:</b> {E(cond['measured_by'])}</span>")
    if "present_at_least" in cond:
        lo = cond.get("absent_below")
        pop.append(f"<span class='l'><b>Cut-offs:</b> present from {cond['present_at_least']:g} {E(unit)}; "
                   + (f"none measured below {lo:g} {E(unit)}; in between = cannot tell" if lo is not None
                      else "a low value is not proof of absence") + "</span>")
    ev = [i for i in item.get("evidence", []) if i in rows]
    if ev:
        pop.append("<span class='l'><b>Evidence:</b></span>" + "".join(
            f"<span class='l'><span class='e'>{E(i)}</span> {E(rows[i]['name'])}: <b>{E(_value_text(rows[i]))}</b></span>" for i in ev))
    if basis:
        pop.append(f"<span class='l'><b>Why this answer:</b> {E(basis)}</span>")
    if cond.get("note"):
        pop.append(f"<span class='l'><b>Note:</b> {E(cond['note'])}</span>")
    pop.append("<span class='l'>Provisional thresholds, not clinically validated; classifier scores are uncalibrated.</span>")
    cls = "mv" if mv is not None else "mv none"
    return (f"<br><span class='{cls}' tabindex='0'>{E(head)}"
            + (f"<span class='mvq'>{E(qual)}</span>" if qual else "")
            + f"<span class='mvpop'>{''.join(pop)}</span></span>")


def measurements_card(evidence: dict) -> str:
    """HTML card (closed by default) listing the evidence rows behind the checklist, with their ids."""
    rows = {r["id"]: r for r in evidence["rows"]}
    return (f'<div class="card" {MARK}><details><summary><b>Measurements behind the checklist</b> '
            f'<span class="muted">(click to open; E-numbers are the evidence ids)</span></summary>'
            f"<table><tr><th>id</th><th>measurement</th><th>value</th></tr>{measurement_rows(rows)}</table>"
            f'<div class="muted">Thresholds are provisional and not clinically validated. Volumes come from '
            f"automatic outlines (TotalSegmentator) and density ranges, not from a radiologist.</div></details></div>")
