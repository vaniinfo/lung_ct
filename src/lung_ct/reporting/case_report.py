"""Standalone per-case HTML reports, reviewer rows and model-input contact sheets.

Reports keep three kinds of information visibly separate:
  REFERENCE  - computed from supplied expert masks (not model output)
  PREDICTED  - TotalSegmentator masks (unvalidated where no reference exists)
  MODEL OBSERVATION - MedGemma text (unvalidated, not a diagnosis)
All images are embedded as base64; the HTML references no external resources.
All model text is HTML-escaped.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

DISCLAIMER = ("Research use only. Not a diagnostic report. Model observations are "
              "unvalidated outputs of a pretrained model; reference values come from "
              "supplied expert masks; predicted volumes without a reference are unvalidated.")

REVIEW_COLUMNS = ["case_id", "finding", "reference_availability", "model_observation",
                  "automated_flags", "clinician_judgment", "reason_for_disagreement",
                  "reviewer_id", "review_date"]

from lung_ct.airway.sensitivity import (  # noqa: E402
    CENTRAL_STATEMENT, PARAMETER_VS_ANATOMY, TORTUOSITY_STATEMENT, central_exceptions)

AIRWAY_NOTES = [
    "Our own Python implementation (lung_ct airway tree v2: nibabel, scikit-image, Skan). "
    "Not AirQuant; no claim of equivalence. Unvalidated: no airway reference in this dataset.",
    "Input is the TotalSegmentator predicted airway mask. Fewer detected branches do not "
    "indicate airway loss (segmentation, slice thickness, inspiration, reconstruction).",
    "Generation = graph depth from the trachea. Visual review of all 10 cases: beyond "
    "generation ~3 it does not correspond to anatomical generation (side buds and residual "
    "loops add depth).",
    "Tortuosity (path length / endpoint distance): " + TORTUOSITY_STATEMENT
    + " See the smoothing ranges in feasibility_summary.md.",
    "Diameter, tapering and wall thickness are not computed.",
]
STABILITY_HELP = ("Parameter stability from a one-at-a-time sweep (pruning, contraction, "
                  "loop contraction, smoothing): stable / moderate / unstable. Unstable values "
                  "depend materially on a processing choice and must not be compared between "
                  "patients. " + PARAMETER_VS_ANATOMY)

_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;max-width:1150px;color:#222}
h1{font-size:22px;margin-bottom:4px}h2{font-size:17px;border-bottom:1px solid #ccc;
padding-bottom:3px;margin-top:26px}table{border-collapse:collapse;margin:8px 0}
td,th{border:1px solid #ccc;padding:3px 8px;font-size:13px;text-align:left;vertical-align:top}
th{background:#f2f2f2}.disc{background:#fff4e5;border:1px solid #f0b46c;padding:8px;
font-size:13px}.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:12px;
color:#fff}.ok{background:#2e7d32}.warn{background:#ef6c00}.fail{background:#c62828}
.na{background:#757575}.ref{border-left:4px solid #1565c0;padding-left:10px}
.pred{border-left:4px solid #6a1b9a;padding-left:10px}.vlm{border-left:4px solid #ef6c00;
padding-left:10px}pre{background:#f7f7f7;border:1px solid #ddd;padding:8px;
white-space:pre-wrap;font-size:12px}img{max-width:100%;border:1px solid #ddd}
.imgs img{width:32%;margin-right:1%}small{color:#555}
.imgs4 img{width:24%;margin-right:0.5%}.air{border-left:4px solid #00838f;padding-left:10px}
.flag{background:#fdecea;border:1px solid #e57373;padding:4px 8px;margin:4px 0;font-size:13px}
.moderate{background:#f9a825}.unstable{background:#c62828}.stable{background:#2e7d32}
"""


def contact_sheet(image_paths: list[Path], cols: int = 17, thumb: int = 128) -> bytes:
    """PNG grid of the exact model-visible images (thumbnails, in prompt order)."""
    if not image_paths:
        raise ValueError("no images")
    rows = -(-len(image_paths) // cols)
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (0, 0, 0))
    for i, p in enumerate(image_paths):
        with Image.open(p) as im:
            t = im.convert("RGB")
            t.thumbnail((thumb, thumb))
            sheet.paste(t, ((i % cols) * thumb, (i // cols) * thumb))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def img_tag(png: bytes, alt: str) -> str:
    return (f'<img alt="{_e(alt)}" src="data:image/png;base64,'
            f'{base64.b64encode(png).decode("ascii")}">')


def _tag(status: str) -> str:
    cls = {"ok": "ok", "valid": "ok", "warn": "warn", "not_run": "na",
           "excluded": "na", "stable": "stable", "moderate": "moderate",
           "unstable": "unstable"}.get(status, "fail")
    return f'<span class="tag {cls}">{_e(status)}</span>'


def _num(x, nd=1, suffix="") -> str:
    return "&ndash;" if x is None else f"{x:.{nd}f}{suffix}"


def _kv(rows: list[tuple]) -> str:
    return "<table>" + "".join(f"<tr><th>{_e(k)}</th><td>{v}</td></tr>" for k, v in rows) \
        + "</table>"


def render_case_html(case: dict, images: dict[str, str]) -> str:
    """case: assembled case record (scripts/build_reports.py).
    images: name -> HTML snippet built with img_tag (ref_overlays, pred_overlays,
    model_input_sheet)."""
    cid = case["case_id"]
    qc, ref, pred, cmp_, vlm = (case.get(k) or {} for k in
                                ("qc", "reference", "predicted", "comparison", "vlm"))
    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{_e(cid)}</title>"
             f"<style>{_CSS}</style></head><body>",
             f"<h1>Case {_e(cid)}</h1><small>report run {_e(case['report_run_id'])}</small>",
             f'<p class="disc">{_e(DISCLAIMER)}</p>']

    parts.append("<h2>Stage status</h2><table><tr><th>Stage</th><th>Status</th>"
                 "<th>Category</th><th>Detail</th></tr>")
    for st, rec in case["stages"].items():
        parts.append(f"<tr><td>{_e(st)}</td><td>{_tag(rec['status'])}</td>"
                     f"<td>{_e((rec.get('category') or '').replace('_', ' '))}</td>"
                     f"<td>{_e(rec.get('detail', ''))}</td></tr>")
    parts.append("</table>")

    ct = qc.get("ct", {})
    parts.append("<h2>Input analysed</h2>" + _kv([
        ("CT file", _e(ct.get("file"))), ("SHA-256", f"<small>{_e(ct.get('sha256'))}</small>"),
        ("Shape / spacing (mm)", _e(f"{ct.get('shape')} / "
                                    f"{[round(s, 4) for s in ct.get('spacing_mm', [])]}")),
        ("Orientation (on disk)", _e(ct.get("orientation"))),
        ("Intensity", _e(f"{ct.get('intensity_range')} {ct.get('dtype_on_disk')}")),
        ("HU valid", _e(f"{ct.get('hu_valid')} - {ct.get('hu_reason')}")),
        ("Geometry CT/masks", _e(json.dumps(qc.get("geometry")))),
        ("Reference lung laterality", _e((qc.get("laterality") or {}).get("reason"))),
    ]))

    parts.append('<h2>Reference (expert masks)</h2><div class="ref">')
    rl, rb, rs = ref.get("lung_volumes") or {}, ref.get("infection_burden") or {}, \
        ref.get("infection_by_side") or {}
    rows = [("Total / left / right lung (mL)",
             f"{_num(rl.get('total_lung_volume_ml'))} / {_num(rl.get('left_lung_volume_ml'))}"
             f" / {_num(rl.get('right_lung_volume_ml'))}"),
            ("Reference infection annotation, % of lung",
             _num(rb.get("infection_burden_pct"), 2, " %"))]
    if rs:
        rows.append(("Infection annotation left / right (mL, % of that lung)",
                     f"{_num(rs['left']['infection_ml'])} ({_num(rs['left']['infection_pct_of_lung'], 2)} %)"
                     f" / {_num(rs['right']['infection_ml'])} ({_num(rs['right']['infection_pct_of_lung'], 2)} %)"))
    parts.append(_kv(rows) + "<small>The infection mask is a COVID-19 infection annotation; it "
                 "is not a ground-glass/consolidation/fibrosis label. Low burden does not mean "
                 "healthy.</small>")
    if "ref_overlays" in images:
        parts.append(f'<div class="imgs">{images["ref_overlays"]}</div>')
    parts.append("</div>")

    parts.append('<h2>Predicted (TotalSegmentator)</h2><div class="pred">')
    if pred:
        lobes = pred.get("lobes", {})
        parts.append("<table><tr><th>Structure</th><th>Status</th><th>Volume (mL)</th></tr>" +
                     "".join(f"<tr><td>{_e(k)}</td><td>{_tag(v['status'])}</td>"
                             f"<td>{_num(v.get('volume_ml'))}</td></tr>"
                             for k, v in {**lobes, **pred.get('airways_vessels', {})}.items())
                     + "</table><small>No reference exists for lobes, airways or vessels in "
                     "this dataset: these volumes are unvalidated predictions.</small>")
        if cmp_:
            parts.append("<p><b>Predicted vs reference lung</b></p><table><tr><th></th>"
                         "<th>Dice</th><th>Pred (mL)</th><th>Ref (mL)</th><th>Error</th></tr>")
            for side in ("lung_left", "lung_right", "lung_total"):
                c = cmp_.get(side, {})
                parts.append(f"<tr><td>{side}</td><td>{_num(c.get('dice'), 3)}</td>"
                             f"<td>{_num(c.get('pred_volume_ml'))}</td>"
                             f"<td>{_num(c.get('ref_volume_ml'))}</td>"
                             f"<td>{_num(c.get('volume_error_pct'), 2, ' %')}</td></tr>")
            parts.append("</table>")
        if "pred_overlays" in images:
            parts.append("<p><small>Predicted lobes, one colour per lobe; sagittal views pass "
                         "through the centre of each lung.</small></p>"
                         f'<div class="imgs4">{images["pred_overlays"]}</div>')
    else:
        parts.append(f"<p>Not available: {_e(case['stages']['segmentation'].get('detail'))}</p>")
    parts.append("</div>")

    parts.append('<h2>Model observation (MedGemma 1.5 4B, unvalidated)</h2><div class="vlm">')
    if vlm:
        mi = vlm.get("model_input", {})
        rep = mi.get("representation", {})
        res = vlm.get("result", {})
        parts.append(_kv([
            ("Model", _e(f"{vlm.get('model_id')} @ {vlm.get('model_revision')}")),
            ("Representation", _e(f"{rep.get('representation')}: {rep.get('n_images')} of "
                                  f"{rep.get('n_source_slices')} axial slices, superior->inferior, "
                                  f"z {rep.get('superior_most_z_mm')} to {rep.get('inferior_most_z_mm')} mm, "
                                  f"windows {rep.get('windows_hu')}, {rep.get('image_format')}")),
            ("Prompt", _e(f"{mi.get('prompt_version')} sha256 {mi.get('prompt_sha256')}")),
            ("Tokens in / out", _e(f"{res.get('input_tokens')} / {res.get('output_tokens')}")),
            ("Generate time / peak GPU", _e(f"{res.get('generate_seconds')} s / "
                                            f"{res.get('peak_gpu_memory_mb')} MB")),
            ("Parse status", _tag(res.get("parse_status", "failed")) + " " +
             _e("; ".join(res.get("errors", []) + res.get("warnings", [])))),
        ]))
        if "model_input_sheet" in images:
            parts.append("<p><b>Exact model-visible images</b> (thumbnails, prompt order)</p>" +
                         images["model_input_sheet"])
        parsed = vlm.get("parsed")
        if parsed:
            d = parsed["distribution"]
            parts.append(_kv([("Assessability", _e(parsed["assessability"])),
                              ("Opacity presence", _e(parsed["opacity_presence"])),
                              ("Distribution", _e(f"laterality={d['laterality']}, "
                                                  f"craniocaudal={d['craniocaudal']}, axial={d['axial']}")),
                              ("Description", _e(parsed["description"])),
                              ("Limitations", _e("; ".join(parsed["limitations"]) or "(none stated)"))]))
        flags = vlm.get("review_flags") or []
        if flags:
            parts.append("<p><b>Automated flags</b> <small>(exploratory rules "
                         f"{_e(flags[0].get('rules_version', ''))}, written after inspecting "
                         "this pilot; not prospectively defined criteria. "
                         "input_description_error = the model described the supplied input "
                         "incorrectly; review_flag = prioritise review, not a confirmed "
                         "error)</small></p>" +
                         "".join(f'<div class="flag"><b>{_e(f["flag"])}</b> '
                                 f'[{_e(f.get("kind", ""))}]: {_e(f["detail"])}</div>'
                                 for f in flags))
        parts.append("<p><b>Raw response</b> (verbatim)</p>"
                     f"<pre>{_e(vlm.get('raw_text'))}</pre>")
    else:
        parts.append(f"<p>Not run: {_e(case['stages']['vlm'].get('detail'))}</p>")
    parts.append("</div>")

    air = case.get("airway")
    parts.append('<h2>Airway centreline, branch lengths, tortuosity (unvalidated)</h2>'
                 '<div class="air">')
    if air:
        parts.append("<ul>" + "".join(f"<li><small>{_e(n)}</small></li>" for n in AIRWAY_NOTES)
                     + "</ul>")
        if "airway_qc" in images:
            parts.append(images["airway_qc"])
        rows = air.get("stability") or []
        if rows:
            exc = central_exceptions(rows)
            parts.append(f"<p><b>Central airways (trachea, main bronchi):</b> "
                         f"{_e(CENTRAL_STATEMENT)}</p>")
            if exc:
                parts.append('<div class="flag"><b>Exception in this case</b>: ' +
                             "; ".join(_e(f"{r['metric']} ranges {r['min']} to {r['max']} "
                                          f"across settings (default {r['default']}; most "
                                          f"influential {r['most_influential']})")
                                       for r in exc) +
                             ". Do not use this value without landmark review.</div>")
            parts.append(f"<p><small>{_e(STABILITY_HELP)}</small></p><table><tr><th>Metric</th>"
                         "<th>Value (default settings)</th><th>Range across settings</th>"
                         "<th>Processing stability</th><th>Most influential setting</th></tr>")
            for r in rows:
                parts.append(f"<tr><td>{_e(r['metric'])}</td><td>{_e(r['default'])}</td>"
                             f"<td>{_e(r['min'])} &ndash; {_e(r['max'])}</td>"
                             f"<td>{_tag(r['rating'])}</td><td>{_e(r['most_influential'])}</td></tr>")
            parts.append("</table>")
        sm = air.get("summary") or {}
        parts.append(_kv([("Method", _e(sm.get("method"))),
                          ("Flags", _e("; ".join(sm.get("flags", [])) or "none")),
                          ("Airway mask kept / discarded fragments (mL)",
                           _e(f"{sm.get('preprocess', {}).get('kept_volume_ml')} / "
                              f"{sm.get('preprocess', {}).get('discarded_volume_ml')}"))]))
    else:
        parts.append(f"<p>Not available: {_e((case['stages'].get('airway') or {}).get('detail'))}</p>")
    parts.append("</div>")

    parts.append("<h2>Provenance</h2>"
                 f"<pre>{_e(json.dumps(case.get('provenance', {}), indent=1))}</pre>")
    parts.append("</body></html>")
    return "\n".join(parts)


def reviewer_rows(case: dict) -> list[dict]:
    """One row per model finding; clinician columns left blank."""
    vlm = case.get("vlm") or {}
    parsed = vlm.get("parsed")
    ref = case.get("reference") or {}
    rb, rs = ref.get("infection_burden") or {}, ref.get("infection_by_side") or {}
    inf_ref = "none"
    if rb:
        inf_ref = f"expert infection annotation: {rb['infection_burden_pct']:.2f}% of lung"
        if rs:
            inf_ref += (f" (L {rs['left']['infection_ml']:.0f} mL, "
                        f"R {rs['right']['infection_ml']:.0f} mL)")
    from lung_ct.reporting.review_flags import flag_finding_map
    FLAG_FINDING = flag_finding_map()
    by_finding = {}
    for f in vlm.get("review_flags") or []:
        by_finding.setdefault(FLAG_FINDING.get(f["flag"], "model_output"), []).append(
            f"{f['flag']}: {f['detail']}")
    base = {"case_id": case["case_id"], "clinician_judgment": "",
            "reason_for_disagreement": "", "reviewer_id": "", "review_date": ""}
    if not parsed:
        status = (vlm.get("result") or {}).get("parse_status") or case["stages"]["vlm"]["status"]
        return [{**base, "finding": "model_output", "reference_availability": inf_ref,
                 "model_observation": f"NO VALID OUTPUT ({status})", "automated_flags": ""}]
    d = parsed["distribution"]
    findings = [
        ("assessability", "none", parsed["assessability"]),
        ("opacity_presence", inf_ref, parsed["opacity_presence"]),
        ("distribution_laterality", inf_ref, d["laterality"]),
        ("distribution_craniocaudal", "infection annotation available; distribution not derived",
         d["craniocaudal"]),
        ("distribution_axial", "infection annotation available; distribution not derived",
         d["axial"]),
        ("description", "image review only", parsed["description"]),
        ("limitations", "image review only", "; ".join(parsed["limitations"]) or "(none stated)"),
    ]
    return [{**base, "finding": f, "reference_availability": r, "model_observation": o,
             "automated_flags": " | ".join(by_finding.get(f, []))}
            for f, r, o in findings]
