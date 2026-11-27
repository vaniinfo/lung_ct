"""Grounded agent demo: one-page HTML review sheet for a clinician (core env).

Everything on one screen (no vertical scrolling at 1920x1080; reviewers zoom themselves):
evidence picture, evidence table, AI readers, cross-check, final report. A small box at the
bottom compares the agent's answers with the expert annotation for THIS pilot only; it is
labelled as such and can be left out with --no-reference.

Usage: python scripts/agent_render_review.py --run runs/A1_agent_<case>_<time> [--no-reference]
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

from lung_ct.agent.report import FIELD_LABEL, _fmt_value  # noqa: E402

E = html.escape
CSS = """
*{box-sizing:border-box}body{margin:0;font:13px/1.35 'Segoe UI',Arial,sans-serif;color:#1f2328;background:#f4f5f7}
.page{display:grid;grid-template-columns:1.05fr 1fr 1fr;grid-template-rows:auto 1fr;gap:10px;padding:10px;height:100vh}
header{grid-column:1/4;display:flex;justify-content:space-between;align-items:baseline}
h1{font-size:20px;margin:0}h2{font-size:14px;margin:0 0 6px}.muted{color:#66707a;font-size:11.5px}
.card{background:#fff;border:1px solid #d5d9de;border-radius:10px;padding:9px 10px;overflow:auto}
.col{display:flex;flex-direction:column;gap:10px;min-height:0}
img{width:100%;border-radius:6px;background:#000}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #eceef1;padding:3px 4px;text-align:left;vertical-align:top}
th{font-size:11.5px;color:#57606a}.id{font-family:Consolas,monospace;color:#57606a;white-space:nowrap}
.tag{display:inline-block;border-radius:9px;padding:0 7px;font-size:11px;white-space:nowrap}
.agree{background:#d6f0da}.conflict,.no_valid_output{background:#fbd5d0}.readers_disagree,.reader_abstained{background:#fbf0c2}
.rel-high{color:#1a7f37}.rel-mod{color:#9a6700}.rel-low{color:#cf222e}
.impr{font-size:15px;font-weight:600;margin:0 0 6px}ul{margin:4px 0 0 16px;padding:0}li{margin:2px 0}
.cite{font-family:Consolas,monospace;font-size:11px;color:#0969da}.flag{color:#cf222e;font-size:11.5px}
"""


def rel_class(r: str) -> str:
    r = r.lower()
    return "rel-high" if r.startswith("high") else "rel-low" if r.startswith("low") else "rel-mod"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--no-reference", action="store_true")
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
    cc = json.loads((run / "crosscheck.json").read_text(encoding="utf-8"))
    rd = json.loads((run / "readers.json").read_text(encoding="utf-8"))
    fr_path = run / "final_report.json"
    if fr_path.exists():
        fr = json.loads(fr_path.read_text(encoding="utf-8"))
    else:
        fr = {"source": "rule_based (step 5 language model not run yet)",
              "report": json.loads((run / "report_rule_based.json").read_text(encoding="utf-8")),
              "validation": {"flags": []}}
    rep = fr["report"]
    img = base64.b64encode((run / "evidence_view.png").read_bytes()).decode()

    ev_rows = "".join(
        f"<tr><td class='id'>{r['id']}</td><td>{E(r['name'])}</td><td><b>{E(_fmt_value(r))}</b>"
        + (f"<div class='muted'>{E(r['note'])}</div>" if r.get("note") and len(r["note"]) < 110 else "")
        + f"</td><td class='{rel_class(r['reliability'])}'>{E(r['reliability'].split(' - ')[0])}</td></tr>"
        for r in ev["rows"] if (r["group"] != "lungs" or r["id"] in ("E01", "E02")) and r["group"] != "classifier"
        or (r["group"] == "classifier" and r["value"] is None))
    cl_rows = [r for r in ev["rows"] if r["group"] == "classifier" and r["value"] is not None]
    if cl_rows:
        hi = (ev.get("classifier") or {}).get("high_threshold", 0.7)
        cells = ", ".join(
            (f"<b>{E(r['name'].replace(' (classifier probability)', ''))} {r['value']:.2f}</b>" if r["value"] >= hi
             else f"{E(r['name'].replace(' (classifier probability)', ''))} {r['value']:.2f}")
            for r in sorted(cl_rows, key=lambda r: -r["value"]))
        ev_rows += (f"<tr><td class='id'>{cl_rows[0]['id']}&ndash;{cl_rows[-1]['id']}</td><td>Step 2 &middot; finding "
                    f"classifier ({E(ev['classifier']['model'])}), probability present</td><td>{cells}</td>"
                    f"<td class='rel-low'>uncalibrated</td></tr>")

    rdr = ""
    for n, p in rd["parsed"].items():
        if p is None:
            rdr += f"<tr><td><b>{E(n)}</b></td><td colspan=4>no valid answer</td></tr>"
            continue
        d = p.get("distribution", {})
        rdr += (f"<tr><td><b>{E(n)}</b></td><td>{E(str(p.get('opacity_presence')))}</td><td>{E(str(d.get('laterality')))}</td>"
                f"<td>{E(str(d.get('craniocaudal')))}</td><td>{E(str(d.get('axial')))}</td></tr>"
                f"<tr><td></td><td colspan=4 class='muted'>&ldquo;{E(p.get('description', ''))}&rdquo;</td></tr>")
    da = ev["derived_answers"]["distribution"]
    rdr += (f"<tr><td><b>measurements</b></td><td>(see E10)</td><td>{E(str(da.get('laterality')))}</td>"
            f"<td>{E(str(da.get('craniocaudal')))}</td><td>{E(str(da.get('axial')))}</td></tr>")

    cc_rows = "".join(
        f"<tr><td>{E(i['reader'])}</td><td>{E(FIELD_LABEL.get(i['field'], i['field']))}</td>"
        f"<td><span class='tag {i['status']}'>{E(i['status'].replace('_', ' '))}</span></td><td>"
        + (f"reader <b>{E(str(i.get('reader_value')))}</b> vs measured <b>{E(str(i.get('evidence_value')))}</b>. " if "reader_value" in i else "")
        + E(i.get("detail", "")) + "</td></tr>"
        for i in cc["items"])

    flags = {f["finding"]: f["flag"] for f in fr.get("validation", {}).get("flags", [])}
    finds = "".join(
        f"<li>{E(f['text'])} <span class='cite'>[{', '.join(E(c) for c in f.get('evidence', []))}]</span>"
        + (f" <span class='flag'>&#9888; {E(flags[k])}</span>" if k in flags else "") + "</li>"
        for k, f in enumerate(rep.get("findings", [])))
    dis = "".join(f"<li>{E(x)}</li>" for x in rep.get("disagreements", []))
    lim = "".join(f"<li>{E(x)}</li>" for x in rep.get("limitations", []))

    ref_box = ""
    ctrate = ROOT / "data" / "raw" / "ct_rate" / "cases.json"
    gt = None
    if ctrate.exists():
        gt = json.loads(ctrate.read_text(encoding="utf-8"))["cases"].get(ev["case_id"])
    if gt is not None and not args.no_reference:
        labs = ", ".join(gt.get("labels") or []) or "none (normal by labels)"
        ref_box = (f"<div class='card'><h2>Radiologist report from CT-RATE (not shown to the agent)</h2>"
                   f"<p><b>Labels</b> (extracted from the report by a text model, not image annotations): {E(labs)}</p>"
                   f"<p><b>Clinical question:</b> {E(gt.get('clinical_information') or '-')}</p>"
                   f"<p><b>Impression:</b> {E(gt.get('impression') or '-')}</p>"
                   f"<details><summary>Full findings</summary><p class='muted'>{E(gt.get('findings') or '-')}</p></details>"
                   f"<div class='muted'>CT-RATE validation set, CC BY-NC-SA 4.0, research use only.</div></div>")
    elif not args.no_reference:
        sys.path.insert(0, str(ROOT / "src"))
        from lung_ct.evaluation.reference_answers import load_case_reference, load_rules
        ref = load_case_reference(ev["case_id"], ROOT, load_rules(str(ROOT / "configs" / "reference_answer_rules_v2.json")))
        ra = ref["answers"]["distribution"]
        cells = "".join(
            f"<tr><td>{FIELD_LABEL[f]}</td><td>{E(str(ra[f]))}</td><td>{E(str(cc['decisions'][f]['value']))}</td>"
            + "".join(f"<td>{E(str((p or {}).get('distribution', {}).get(f)))}</td>" for p in rd["parsed"].values())
            + "</tr>" for f in ("laterality", "craniocaudal", "axial"))
        ref_box = (f"<div class='card'><h2>Pilot check vs expert annotation (not shown to the agent)</h2><table>"
                   f"<tr><th></th><th>expert mask</th><th>agent</th>{''.join(f'<th>{E(n)}</th>' for n in rd['parsed'])}</tr>"
                   f"{cells}</table><div class='muted'>Rules v2, provisional definitions; this case is part of the "
                   f"10-case pilot used to set thresholds.</div></div>")

    page = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>Agent review - {E(ev['case_id'])}</title>
<style>{CSS}</style></head><body><div class='page'>
<header><h1>Grounded lung CT agent &mdash; {E(ev['case_id'])}</h1>
<span class='muted'>Research use only &middot; not a clinical report &middot; run {E(run.name)} &middot; report writer: {E(fr['source'])}</span></header>
<div class='col'>
 <div class='card'><h2>Step 1 &middot; What the measurements see</h2><img src='data:image/png;base64,{img}'></div>
 <div class='card'><h2>Step 3 &middot; AI readers (images only)</h2><table>
 <tr><th></th><th>opacities</th><th>side</th><th>zone</th><th>periph.</th></tr>{rdr}</table></div>
</div>
<div class='col'>
 <div class='card' style='flex:1'><h2>Step 1 &middot; Evidence table</h2><table>
 <tr><th>id</th><th>measure</th><th>value</th><th>reliability</th></tr>{ev_rows}</table></div>
</div>
<div class='col'>
 <div class='card'><h2>Step 5 &middot; Final report</h2><p class='impr'>{E(rep.get('impression', ''))}</p>
 <ul>{finds}</ul><h2 style='margin-top:8px'>Please check</h2><ul>{dis or '<li>none</li>'}</ul>
 <h2 style='margin-top:8px'>Limitations</h2><ul class='muted'>{lim}</ul></div>
 <div class='card' style='flex:1'><h2>Step 4 &middot; Cross-check ({cc['n_conflicts']} flagged)</h2><table>{cc_rows}</table>
 <div class='muted'>{E(cc['rule'])}</div></div>
 {ref_box}
</div></div></body></html>"""
    (run / "review.html").write_text(page, encoding="utf-8")
    print(f"-> {run / 'review.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
