"""One-page HTML review of the condition checklist for one agent run (core env).

Shows: the scan with the density types, the checklist (status, location, basis, evidence ids,
classifier score, readers) next to the CT-RATE report labels when the case comes from CT-RATE,
the final report, and the measurements behind the checklist. The CT-RATE report is NOT an input
to the agent; it is only shown here for review.

Usage: python scripts/agent_render_checklist.py --run runs/A1_agent_<case>_<time>
Writes <run>/review_checklist.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.report import _fmt_value  # noqa: E402

E = html.escape
CSS = """
*{box-sizing:border-box}body{margin:0;font:13px/1.35 'Segoe UI',Arial,sans-serif;color:#1f2328;background:#f4f5f7}
.page{display:grid;grid-template-columns:0.95fr 1.45fr 0.9fr;grid-template-rows:auto 1fr;gap:10px;padding:10px;height:100vh}
header{grid-column:1/4;display:flex;justify-content:space-between;align-items:baseline;gap:12px}
h1{font-size:20px;margin:0}h2{font-size:14px;margin:0 0 6px}.muted{color:#66707a;font-size:11.5px}
.card{background:#fff;border:1px solid #d5d9de;border-radius:10px;padding:9px 10px;overflow:auto}
.col{display:flex;flex-direction:column;gap:10px;min-height:0}
img{width:100%;border-radius:6px;background:#000}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #eceef1;padding:3px 4px;text-align:left;vertical-align:top}
th{font-size:11.5px;color:#57606a}.id{font-family:Consolas,monospace;color:#57606a;font-size:11px}
.st{display:inline-block;border-radius:9px;padding:0 8px;font-size:11.5px;font-weight:600;white-space:nowrap}
.present{background:#ffd8d3}.absent{background:#dcf2e0}.cannot-tell{background:#eceef1}.cannot-assess{background:#e3e0f7}
.m{display:inline-block;border-radius:9px;padding:0 6px;font-size:11px;white-space:nowrap}
.agree{background:#d6f0da}.missed{background:#fbd5d0}.extra{background:#fde7c4}.open{background:#eceef1}
.impr{font-size:15px;font-weight:600;margin:0 0 6px}ul{margin:4px 0 0 16px;padding:0}li{margin:2px 0}
.cite{font-family:Consolas,monospace;font-size:11px;color:#0969da}.flag{color:#cf222e;font-size:11.5px}
.why{font-size:11px;color:#57606a;line-height:1.25}td{line-height:1.25}
.pill{display:inline-block;background:#eef1f4;border-radius:9px;padding:0 8px;margin-right:4px;font-size:12px}
"""


def match(status: str, label_pos: bool | None) -> tuple[str, str]:
    if label_pos is None:
        return "", ""
    if status == "suggested":      # v3_8: predicted mask only; never shown as an agreement or an extra finding
        return ("open", "suggested only; report: " + ("present" if label_pos else "not mentioned"))
    if status == "present":
        return ("agree", "matches report") if label_pos else ("extra", "not in report")
    if status == "absent":
        return ("missed", "report says present") if label_pos else ("agree", "matches report")
    return ("open", "report: present" if label_pos else "report: not mentioned")


MEAS_IDS = ["E69", "E95", "E96", "E70", "E71", "E72", "E73", "E79", "E76", "E77", "E78", "E30", "E31", "E32", "E33", "E80", "E81", "E82", "E85", "E86", "E87", "E88", "E90"]


def measurement_rows(rows: dict) -> str:
    """Table rows (id, name, value + note) for the measurements behind the checklist; also used by the viewer pages."""
    return "".join(
        f"<tr><td class='id'>{i}</td><td>{E(rows[i]['name'])}</td><td><b>{E(_fmt_value(rows[i]) if not isinstance(rows[i]['value'], dict) else ', '.join(f'{k} {v}' for k, v in rows[i]['value'].items() if k in ('side', 'zone', 'leading_lobe') or 'side' not in rows[i]['value']))}</b>"
        + (f"<div class='muted'>{E(rows[i]['note'])}</div>" if rows[i].get("note") else "") + "</td></tr>"
        for i in MEAS_IDS if i in rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--checklist-file", default="checklist.json",
                    help="e.g. checklist_checklist_v3_2.json written by rescore_checklist.py")
    ap.add_argument("--rule-based", action="store_true",
                    help="show the rule-based report built from this checklist (ignore final_report.json)")
    ap.add_argument("--out", default=None, help="output path (default <run>/review_checklist.html)")
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
    chk = json.loads((run / args.checklist_file).read_text(encoding="utf-8"))
    fr_path = run / "final_report.json"
    if args.rule_based:
        from lung_ct.agent.checklist import rule_based_checklist_report
        fr = {"source": f"rule_based ({chk['config_version']})", "validation": {"flags": [], "errors": []},
              "report": rule_based_checklist_report(chk)}
    else:
        fr = (json.loads(fr_path.read_text(encoding="utf-8")) if fr_path.exists() else
              {"source": "rule_based (language model not run)", "validation": {"flags": [], "errors": []},
               "report": json.loads((run / "report_rule_based.json").read_text(encoding="utf-8"))})
    rep, val = fr["report"], fr.get("validation", {})
    img = base64.b64encode((run / "checklist_view.png").read_bytes()).decode()
    rows = {r["id"]: r for r in ev["rows"]}

    gt = None
    cj = ROOT / "data" / "raw" / "ct_rate" / "cases.json"
    if cj.exists():
        gt = json.loads(cj.read_text(encoding="utf-8"))["cases"].get(ev["case_id"])
    labels = set(gt["labels"]) if gt else None

    trs, tally = "", {"agree": 0, "missed": 0, "extra": 0, "open": 0}
    for it in chk["items"]:
        lp = None if labels is None or not it.get("ct_rate_label") else it["ct_rate_label"] in labels
        mc, mt = match("suggested" if it["status"] == "present" and it.get("headline") else it["status"], lp)
        if mc:
            tally[mc] += 1
        rd = ", ".join(f"{E(n)}: {E(s.replace('_', ' '))}" for n, s in it.get("readers", {}).items()) or "&ndash;"
        cp = f"{it['classifier_p']:.2f}" if it.get("classifier_p") is not None else "&ndash;"
        trs += (f"<tr><td><b>{E(it['label'])}</b></td>"
                f"<td><span class='st {it['status'].replace(' ', '-')}'>{E(it.get('display', it['status']))}</span></td>"
                f"<td>{E(it.get('statement') if it['status'] == 'absent' and it.get('statement') else (it.get('location') or ''))}<div class='why'>{E(it['basis'])}</div>"
                + "".join(f"<div class='flag'>&#9888; {E(f)}</div>" for f in it["flags"])
                + f"</td><td class='id'>{E(', '.join(it['evidence']))}</td><td>{cp}</td><td class='muted'>{rd}</td>"
                + (f"<td><span class='m {mc}'>{E(mt)}</span></td>" if labels is not None else "") + "</tr>")
    head = ("<tr><th>condition</th><th>status</th><th>where / why</th><th>evidence</th><th>classifier</th>"
            "<th>AI readers</th>" + ("<th>vs CT-RATE report</th>" if labels is not None else "") + "</tr>")

    flags = {f["finding"]: f["flag"] for f in val.get("flags", []) if isinstance(f, dict)}
    finds = "".join(f"<li>{E(f['text'])} <span class='cite'>[{', '.join(E(c) for c in f.get('evidence', []))}]</span>"
                    + (f" <span class='flag'>&#9888; {E(flags[k])}</span>" if k in flags else "") + "</li>"
                    for k, f in enumerate(rep.get("findings", [])))
    def _item(x):
        # the report writer sometimes returns {"text": ..., "evidence": [...]} instead of a plain string
        if isinstance(x, dict):
            ev = [str(c) for c in x.get("evidence") or [] if str(c).lower() != "none"]
            return E(str(x.get("text", ""))) + (f" <span class='cite'>[{E(', '.join(ev))}]</span>" if ev else "")
        return E(str(x))
    chk_items = "".join(f"<li>{_item(x)}</li>" for x in rep.get("disagreements") or []) or "<li>none</li>"
    lim = "".join(f"<li>{_item(x)}</li>" for x in rep.get("limitations") or [])

    mrows = measurement_rows(rows)

    gt_box = ""
    if gt:
        gt_box = (f"<div class='card'><h2>Radiologist report (CT-RATE) &mdash; not shown to the agent</h2>"
                  f"<p><b>Labels:</b> {E(', '.join(gt['labels']) or 'none')}</p>"
                  f"<p><b>Clinical question:</b> {E(gt.get('clinical_information') or '-')}</p>"
                  f"<p><b>Impression:</b> {E(gt.get('impression') or '-')}</p>"
                  f"<details><summary>Full findings</summary><p class='muted'>{E(gt.get('findings') or '-')}</p></details>"
                  f"<div class='muted'>Labels were extracted from the report text by a model; 'not mentioned' is not proof of absence.</div></div>")
    c = chk["counts"]
    nm_ = sum(i.get('display') == 'none measured' for i in chk['items'])
    summ = (f"<span class='pill'>present {c['present']}</span><span class='pill'>none measured {nm_}</span>"
            f"<span class='pill'>not suggested {c['absent'] - nm_}</span>"
            f"<span class='pill'>cannot tell {c['cannot tell']}</span><span class='pill'>cannot assess {c['cannot assess']}</span>")
    if labels is not None:
        summ += (f" &nbsp; vs report: <span class='m agree'>agree {tally['agree']}</span> "
                 f"<span class='m missed'>missed {tally['missed']}</span> <span class='m extra'>extra {tally['extra']}</span> "
                 f"<span class='m open'>open {tally['open']}</span>")
    page = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>Checklist review - {E(ev['case_id'])}</title>
<style>{CSS}</style></head><body><div class='page'>
<header><h1>ThoraxCheck condition checklist &mdash; {E(ev['case_id'])}</h1><div>{summ}</div>
<span class='muted'>Research use only &middot; scan type: {E(chk['scan_type'])} &middot; {E(chk['config_version'])} (provisional thresholds) &middot; run {E(run.name)} &middot; report writer: {E(fr['source'])}</span></header>
<div class='col'>
 <div class='card'><h2>What the measurements see</h2><img src='data:image/png;base64,{img}'></div>
 {gt_box}
</div>
<div class='col'>
 <div class='card' style='flex:1'><h2>Condition checklist</h2><table>{head}{trs}</table>
 <div class='muted' style='margin-top:6px'>Rule: {E(' '.join(chk['rule']))}</div></div>
</div>
<div class='col'>
 <div class='card'><h2>Report</h2><p class='impr'>{E(rep.get('impression', ''))}</p><ul>{finds}</ul>
 <h2 style='margin-top:8px'>Please check</h2><ul>{chk_items}</ul>
 <h2 style='margin-top:8px'>Limitations</h2><ul class='muted'>{lim}</ul></div>
 <div class='card' style='flex:1'><h2>Measurements behind the checklist</h2><table>{mrows}</table></div>
</div></div></body></html>"""
    out = Path(args.out) if args.out else run / "review_checklist.html"
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
