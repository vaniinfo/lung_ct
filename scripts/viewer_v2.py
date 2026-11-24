"""Location-first presentation for the interactive viewer (build_viewer_page.py --viewer v2). Display only.

What changes against viewer v1 (which is kept so earlier pages can be rebuilt exactly):
  * The answer cell leads with WHERE: lobe names, the kind of region ("HU-selected region", and for an unsure
    answer "not a confirmed finding"), and a flag when the lobe assignment is uncertain. No number under the answer.
  * Numbers (mL, percentages, HU) are in a measurement panel that is always on the page and opened by a button,
    and in a per-condition "Numbers" box opened by a click. Nothing relies on hover.
  * Every percentage names its denominator: % of that lobe, % of total lung, or % of the selected region.
  * Geometric thirds of lung height are shown as such and never called segments.
Nothing here changes a measurement, a threshold or an answer.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_render_checklist import MEAS_IDS  # noqa: E402

E = html.escape
VIEWER_BUILD = "checklist_viewer_v2_location_first"
LOBE_ORDER = ["RUL", "RML", "RLL", "LUL", "LLL"]
# what each percentage is a percentage OF (shown next to the value)
DENOMINATOR = {"E81": "% of total lung", "E86": "% of total lung", "E30": "% of total lung", "E31": "% of total lung",
               "E32": "% of that lobe", "E13": "% of all dense tissue", "E14": "% of all dense tissue",
               "E15": "% of all dense tissue", "E20": "% of that lobe", "E21": "% of that lobe", "E22": "% of that lobe",
               "E23": "% of that lobe", "E24": "% of that lobe"}
REGION_ROWS = [("E83", "E81", "E82", "Ground-glass-range region"), ("E89", "E86", "E87", "Consolidation-range region (dense lung)"),
               ("E91", None, None, "Dense lung next to pleural fluid")]

CSS = """
.loc{margin-top:4px;line-height:1.5}
.lobe{display:inline-block;background:#0b3d91;color:#fff;font-weight:700;font-size:12px;border-radius:5px;padding:0 6px;margin-right:3px}
.reg{display:block;font-size:11px;color:#57606a}
.unc{display:block;font-size:11px;color:#9a3412;font-weight:600}
.plainloc{font-size:12px;font-weight:600}
.qual{display:block;font-size:11px;color:#66707a}
.mbtn{background:#fff;color:#0969da;border:1px solid #c9d1d9;border-radius:6px;font-size:11px;padding:1px 7px;margin-top:5px}
.mdet{margin-top:5px;background:#f6f8fa;border:1px solid #d8dee4;border-radius:7px;padding:6px 8px;font-size:11.5px;line-height:1.4}
.mdet .l{display:block;margin-top:2px}.mdet .e{font-family:Consolas,monospace;color:#57606a}
.panelbtn{background:#0969da;border-color:#0969da;color:#fff;font-weight:600;margin-top:8px}
#measpanel h3{font-size:13px;margin:10px 0 3px}#measpanel table{margin-top:4px}
#measpanel td.n,#measpanel th.n{text-align:right;font-variant-numeric:tabular-nums}
#measpanel tr.named td{background:#eef4ff}
.id{font-family:Consolas,monospace;color:#57606a;font-size:11px;white-space:nowrap}
"""


def unit_label(unit: str | None, rid: str | None = None) -> str:
    if rid in DENOMINATOR:
        return DENOMINATOR[rid]
    return {"% of lung": "% of total lung"}.get(unit or "", unit or "")


def _num(v) -> str:
    return "-" if v is None else f"{v:g}"


def _value_text(r: dict) -> str:
    v = r["value"]
    if isinstance(v, dict) and "lobes" in v:
        return f"{_num(v['region_ml'])} mL in total; per-lobe table in the measurement panel"
    if isinstance(v, dict) and "side" in v:
        return f"side {v['side']}; largest share in {v['leading_lobe']} ({v['leading_lobe_share_pct']} % of the selected region)"
    if isinstance(v, dict):
        return ", ".join(f"{k} {x}" for k, x in v.items()) + " " + unit_label(r.get("unit"), r["id"])
    if isinstance(v, (int, float)) and r.get("unit") == "p":
        return f"{v:.2f} (0 to 1, uncalibrated)"
    return f"{v:g} {unit_label(r.get('unit'), r['id'])}".strip() if isinstance(v, (int, float)) else ("not available" if v is None else str(v))


def _location_html(item: dict) -> str:
    lr = item.get("location_report")
    parts = lr.get("parts") if lr and lr.get("parts") else ([lr] if lr else [])
    if parts:
        out = ""
        for p in parts:
            lobes = "".join(f"<span class='lobe'>{E(n)}</span>" for n in p["lobes_named"]) or "<span class='plainloc'>no single lobe</span>"
            out += (f"<div class='loc'>{lobes}<span class='reg'>{E(p['region'])} &middot; {E(p['region_is'])}</span>"
                    + (f"<span class='unc'>&#9888; lobe assignment uncertain: {E('; '.join(p['uncertain_because']))}</span>"
                       if p["lobe_assignment_uncertain"] else "") + "</div>")
        return out
    if item.get("location"):  # conditions without a region row (e.g. pleural effusion: side)
        return f"<div class='loc'><span class='plainloc'>{E(item['location'])}</span></div>"
    return ""


def answer_cell(item: dict, cond: dict, rows: dict) -> str:
    """Badge, then location in words, then a click-to-open box with the numbers behind this answer."""
    unit = unit_label(cond.get("unit"))
    basis = item.get("basis", "")
    qual = "shallow breath" if basis.startswith("lungs poorly inflated") else ""
    det = []
    if cond.get("measured_by"):
        det.append(f"<span class='l'><b>Measured by:</b> {E(cond['measured_by'].replace('% of lung', '% of total lung'))}</span>")
    if item.get("measured_value") is not None:
        det.append(f"<span class='l'><b>Measured value:</b> {item['measured_value']:g} {E(unit)}</span>")
    ptr = item.get("provisional_threshold_result")
    if ptr:      # v3_8: the answer of the unchanged limits is kept apart from the headline and is never a current assessment
        det.append("<span class='l ptr'><b>Provisional threshold result (kept separately; not the current answer):</b> the earlier rule, "
                   f"unchanged, gives &ldquo;{E(PTR_WORDS.get(ptr['answer'], ptr['answer']))}&rdquo; for this estimate. Its limits "
                   f"({ptr['present_at_least']:g} and {ptr['absent_below']:g} {E(unit)}) are provisional reporting limits, not validated diagnostic "
                   "boundaries. A result below a limit is <b>not</b> a finding of no effusion.</span>")
    elif "present_at_least" in cond and not cond.get("hide_cutoffs"):
        lo = cond.get("absent_below")
        det.append(f"<span class='l'><b>Tool's provisional cut-offs:</b> present from {cond['present_at_least']:g} {E(unit)}; "
                   + (f"none measured below {lo:g} {E(unit)}; in between = cannot tell" if lo is not None
                      else "a low value is not proof of absence") + "</span>")
    ev = [i for i in item.get("evidence", []) if i in rows]
    if ev:
        det.append("<span class='l'><b>Evidence:</b></span>" + "".join(
            f"<span class='l'><span class='e'>{E(i)}</span> {E(rows[i]['name'])}: <b>{E(_value_text(rows[i]))}</b></span>" for i in ev))
    dc = item.get("dense_lung_candidates")
    if dc:       # v3_6 and later: tagged candidates, never a finding
        t = dc["tags_ml"]
        det.append("<span class='l'><b>Dense-lung candidates (density-selected, tagged; not a finding):</b> "
                   f"{dc['total_ml']:g} mL in total. Tags: no tag {t['no_spatial_tag']:g}; next to predicted pleural fluid {t['fluid_adjacent']:g}; "
                   f"posterior band {t['posterior']:g}; both tags {t['fluid_adjacent_and_posterior']:g} mL. A tag says where a region lies and names "
                   "no diagnosis. No limit is applied to these numbers.</span>")
    lr = item.get("location_report")
    if lr and not lr.get("parts"):
        det.append(f"<span class='l'><b>Lobes named by rule:</b> a lobe is named when it holds at least "
                   f"{lr['naming_cutoff_pct_of_region']:g} % of the selected region ({E(lr['rule'])}, provisional)</span>")
    if basis:
        det.append(f"<span class='l'><b>Why this answer:</b> {E(basis)}</span>")
    if cond.get("note"):
        det.append(f"<span class='l'><b>Note:</b> {E(cond['note'])}</span>")
    det.append("<span class='l'>Cut-offs are provisional and not clinically validated; classifier scores are uncalibrated.</span>")
    return (_location_html(item) + (f"<span class='qual'>{E(qual)}</span>" if qual else "")
            + "<button class='mbtn' type='button'>Numbers</button>"
            + f"<div class='mdet' hidden>{''.join(det)}</div>")


PTR_WORDS = {"present": "at or above the upper reporting limit", "undecided": "between the two reporting limits",
             "none": "below the lower reporting limit"}
TAG_LABELS = (("no_spatial_tag", "no spatial tag (the earlier baseline region)"), ("fluid_adjacent", "next to predicted pleural fluid"),
              ("posterior", "posterior band"), ("fluid_adjacent_and_posterior", "both tags"))


def dense_candidates_table(rows: dict) -> str:
    """v3_6 and later: the retained dense-lung candidates by spatial tag and lobe (rows E92-E94). Candidates, not findings."""
    v = (rows.get("E94") or {}).get("value")
    if not isinstance(v, dict) or "tags" not in v:
        return ""
    tr = ""
    for k, lab in TAG_LABELS:
        t = v["tags"][k]
        tr += (f"<tr><td>{E(lab)}</td><td class='n'>{_num(t['ml'])}</td>"
               + "".join(f"<td class='n'>{_num(t['per_lobe_ml'].get(n))}</td>" for n in LOBE_ORDER) + "</tr>")
    return ("<h3>Dense-lung candidates, by spatial tag <span class='id'>E92 E93 E94</span></h3>"
            "<table><tr><th>spatial tag</th><th class='n'>mL</th>" + "".join(f"<th class='n'>{n}</th>" for n in LOBE_ORDER) + "</tr>"
            f"{tr}<tr class='named'><td><b>all candidates</b></td><td class='n'><b>{_num(v['total_ml'])}</b></td>"
            + "<td></td>" * len(LOBE_ORDER) + "</tr></table>"
            "<div class='muted'>Lung selected by CT density only (consolidation HU range). These are candidates, not findings: the tool does not "
            "determine whether they are consolidation, atelectasis, dependent density or something else. Nothing is left out because of where it "
            "lies; the tags only say where. The earlier limits for consolidation are not applied to these numbers.</div>")


def _region_table(rows: dict, did: str, pid: str | None, lid: str | None, title: str, named: list[str], cut) -> str:
    v = (rows.get(did) or {}).get("value")
    if not isinstance(v, dict):
        why = (rows.get(did) or {}).get("source") or "not measured on this scan"
        return f"<h3>{E(title)} <span class='id'>{did}</span></h3><div class='muted'>{E(why)}</div>" if did in rows else ""
    if not v["region_voxels"]:
        return f"<h3>{E(title)} <span class='id'>{did}</span></h3><div class='muted'>No voxel selected.</div>"
    tr = ""
    for n in LOBE_ORDER:
        d = v["lobes"].get(n)
        if d is None:
            continue
        tr += (f"<tr class='{'named' if n in named else ''}'><td><b>{n}</b></td><td class='n'>{_num(d['ml'])}</td>"
               f"<td class='n'>{_num(d['pct_of_lobe'])}</td><td class='n'>{_num(d['pct_of_region'])}</td>"
               f"<td class='n'>{_num(d.get('near_fissure_pct_of_lobe_part'))}</td><td>{'yes' if n in named else ''}</td></tr>")
    tot = f"Selected region in total: <b>{_num(v['region_ml'])} mL</b>"
    if pid and rows.get(pid, {}).get("value") is not None:
        tot += f" = <b>{rows[pid]['value']:g} % of total lung</b> <span class='id'>{pid}</span>"
    extra = [tot + "."]
    if v.get("near_fissure_pct_of_region") is not None:
        extra.append(f"{_num(v['near_fissure_pct_of_region'])} % of the selected region lies within {v['near_fissure_mm']:g} mm "
                     "of a predicted fissure (lobe label less certain there).")
    if v.get("outside_lobes_pct_of_region"):
        extra.append(f"{_num(v['outside_lobes_pct_of_region'])} % of the selected region lies in no lobe outline.")
    loc = rows.get(lid, {}).get("value") if lid else None
    if isinstance(loc, dict):
        z = loc["zone_shares_pct"]
        extra.append(f"Right lung holds {loc['right_share_pct']} % of the selected region. By height (geometric thirds of the lungs, "
                     f"not anatomical segments), as % of the selected region: upper {z['upper']}, middle {z['middle']}, lower {z['lower']} "
                     f"<span class='id'>{lid}</span>.")
    return (f"<h3>{E(title)} <span class='id'>{did}</span></h3>"
            "<table><tr><th>lobe</th><th class='n'>mL in this lobe</th><th class='n'>% of that lobe</th>"
            "<th class='n'>% of the selected region</th><th class='n'>of this part, % within 5 mm of a fissure</th>"
            f"<th>named in the sentence</th></tr>{tr}</table>"
            f"<div class='muted'>{' '.join(extra)}"
            + (f" A lobe is named when it holds at least {cut:g} % of the selected region (provisional rule)." if cut is not None else "")
            + "</div>")


def measurement_panel(evidence: dict, chk: dict) -> str:
    """The always-present measurement panel: per-lobe tables with explicit denominators, then every measurement row."""
    rows = {r["id"]: r for r in evidence["rows"]}
    items = {i["key"]: i for i in chk["items"]}
    named, cut = {}, None
    for key, did in (("ground_glass", "E83"), ("consolidation", "E89"), ("atelectasis", "E91")):
        lr = (items.get(key) or {}).get("location_report")
        if lr:
            named[did], cut = lr["lobes_named"], lr["naming_cutoff_pct_of_region"]
    if cut is None:
        cut = (chk.get("location_rule") or {}).get("cutoff")
    tables = "".join(_region_table(rows, d, p, l, t, named.get(d, []), cut) for d, p, l, t in REGION_ROWS)
    allrows = ""
    for i in MEAS_IDS:
        if i not in rows:
            continue
        r = rows[i]
        allrows += (f"<tr><td class='id'>{i}</td><td>{E(r['name'])}</td><td><b>{E(_value_text(r))}</b>"
                    + (f"<div class='muted'>{E(r['note'])}</div>" if r.get("note") else "") + "</td></tr>")
    return ('<div class="card agentonly" id="measpanel"><b>Measurement panel</b> '
            '<span class="muted">numbers behind the answers; E-numbers are the evidence ids</span>'
            '<div class="muted">These are the tool\'s own measurements from automatic outlines (TotalSegmentator) and HU ranges. '
            'An HU-selected region is lung in a density range; it is not a confirmed ground-glass or consolidation finding. '
            'Cut-offs are provisional and not clinically validated.</div>'
            + (f"<h3>Where each HU-selected region lies</h3>{tables}" if tables else "")
            + dense_candidates_table(rows)
            + f"<h3>All measurements</h3><table><tr><th>id</th><th>measurement</th><th>value</th></tr>{allrows}</table></div>")
