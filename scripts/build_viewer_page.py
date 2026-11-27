"""Interactive one-page review: a real CT viewer (NiiVue, WebGL2) next to the condition checklist,
with a grading panel (scripts/viewer_grading.py) so a reviewer can grade every condition and save a CSV.

Click a checklist row and the viewer turns on that finding's overlay and jumps to it. Scroll through
slices, switch lung / soft-tissue / bone windows, and switch each overlay on or off. Everything
(scan, overlays, viewer code) is inside ONE html file: it works offline and sends nothing anywhere.
The page contains CT-RATE images and report text: share only inside the research group.

Overlays (all from predicted masks or the checklist measurements, never from reference labels):
pleural fluid, pericardial fluid, consolidation range, ground-glass range, compressed lung next to
fluid, aortic calcium, heart, and the back-of-lung band that is not counted.

Usage (core env):
  python scripts/build_viewer_page.py --run runs/A1_agent_valid_301_a_2_20261118_152058 \
      --checklist-file checklist_checklist_v3_2.json --rule-based --out runs/M1_meeting_pack_20261119/x.html

--viewer v1 (default) is the page as first sent to reviewers and is kept so those pages can be rebuilt.
--viewer v2 is the location-first page (scripts/viewer_v2.py) with grading form v2 (scripts/viewer_grading_v2.py):
  lobes first, numbers in a measurement panel, lobe grading. Use it with a checklist_v3_4 checklist file.
--viewer v3 is the v2 page with a different loader (scripts/viewer_v3.py, build checklist_viewer_v3_lazy): still one
  standalone file for file://, CT stored losslessly in less space, payloads decoded step by step, one overlay loaded
  at a time, visible loading and error states. Answers, measurements, grading form and CSV are those of v2.
--viewer v4 is the same page with a 2D canvas viewer and NO WebGL / NiiVue (scripts/viewer_v4.py, build
  checklist_viewer_v4_canvas2d): one slice drawn at a time, voxels reordered losslessly to RAS+ from the affine,
  oblique scans refused, one outline at a time. Made for laptops with integrated graphics. v3 is NOT a confirmed fix
  for those machines (docs/STATUS.md).
--blinded (v2 / v3 / v4) builds the staged page: step 1 shows the scan and the form only; the tool's answers, locations,
  measurements and outlines appear after the reviewer locks step 1. The CT-RATE report is left out of staged pages.
  The hidden content is inside the HTML file, so this is a reading procedure, not a security barrier.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_render_checklist import match  # noqa: E402
from viewer_grading import CSS as GRADING_CSS, grading_script, load_grades_csv  # noqa: E402
from viewer_measurements import CSS as MEAS_CSS, answer_details, measurements_card  # noqa: E402
import viewer_grading_v2 as gv2  # noqa: E402
import viewer_v2 as v2  # noqa: E402
import viewer_v3 as v3  # noqa: E402
import viewer_v4 as v4  # noqa: E402
import viewer_v5 as v5  # noqa: E402
import viewer_v6 as v6  # noqa: E402
import viewer_v7 as v7  # noqa: E402
from lung_ct.agent.checklist import assessment_category, load_checklist_config, rule_based_checklist_report  # noqa: E402
from lung_ct.agent.conditions import image_noise, measure_conditions  # noqa: E402
from lung_ct.agent.evidence import LOBES  # noqa: E402

E = html.escape
NIIVUE = ROOT / "scripts" / "vendor" / "niivue-0.69.0.umd.js"
# All overlays start OFF (reviewer request 2026-11-21: read the plain scan first). An overlay is loaded into the
# viewer only when it is first switched on, so the page opens with the CT alone.
OVERLAYS = [  # key, label, NiiVue colormap, swatch colour, on at start
    ("pleural", "Pleural fluid", "blue", "#3b82f6", False),
    ("pericardial", "Pericardial fluid", "violet", "#a855f7", False),
    ("consolidation", "Consolidation range", "red", "#ef4444", False),
    ("ground_glass", "Ground-glass range", "warm", "#facc15", False),      # 'gold' ends in white: looked grey on lung
    ("compressed", "Compressed lung next to fluid", "copper", "#c98a4b", False),
    ("calcium", "Aortic calcium", "green", "#22c55e", False),
    ("heart", "Heart outline", "winter", "#06b6d4", False),
    ("dependent", "Back-of-lung band (not counted)", "cool", "#94a3b8", False),
    ("nodules", "Nodule candidates (segmentation model)", "hot", "#f97316", False),     # only loaded for checklist_v3_10 and later
]
JUMP = {"pleural_effusion": "pleural", "pericardial_effusion": "pericardial", "consolidation": "consolidation",
        "ground_glass": "ground_glass", "atelectasis": "compressed", "arterial_calcification": "calcium",
        "cardiomegaly": "heart", "lung_density": "consolidation", "lung_nodule": "nodules"}


SHOW_V2 = "<br><span class='show agentonly'>&#128065; Show in scan</span>"


def nii_b64(arr: np.ndarray, affine: np.ndarray) -> str:
    img = nib.Nifti1Image(arr, affine)
    img.header.set_xyzt_units("mm")
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    return base64.b64encode(gzip.compress(img.to_bytes(), compresslevel=6)).decode()


def target_mm(mask: np.ndarray, affine: np.ndarray) -> list[float] | None:
    """World position (mm) of the centre of the largest connected piece of a mask."""
    if mask is None or not mask.any():
        return None
    lbl, n = ndimage.label(mask)
    big = int(np.argmax(np.bincount(lbl.ravel())[1:])) + 1
    c = np.array(ndimage.center_of_mass(lbl == big))
    return [round(float(v), 1) for v in (affine @ np.append(c, 1.0))[:3]]


POLICY_CSS = """
.suggested{background:#ffe9b8}
.policybar{background:#e7f0ff;border:1px solid #9db9ee;border-radius:10px;padding:8px 10px;margin-top:8px;font-size:13px;color:#10305f}
.ptr{background:#f6f8fa;border-left:3px solid #9aa4b2;padding-left:6px}
.earlier table{font-size:12px}.earlier summary{cursor:pointer}
.refcol{font-size:12px;min-width:92px}.refnote{font-weight:400;font-size:10.5px;color:#57606a}
"""
# wording that earlier location reports still carry inside checklist files of the current policy; replaced in the displayed text only
POLICY_DISPLAY_ONLY = (("pleural effusion present (compressed lung)", "pleural fluid outlined by the segmentation model (lobe outlines are less reliable there)"),)


def earlier_grades_card(path: Path, case_id: str, prefilled: bool = False) -> str:
    """Read-only copy of grades the reviewer sent earlier, labelled with the page policy they were made on. The file is not changed."""
    import csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(r["case_id"] != case_id for r in rows):
        raise SystemExit(f"{path}: not a grades file for {case_id}")
    cfgs = sorted({r["checklist_config"] for r in rows if r.get("checklist_config")})
    form = rows[0].get("form_version") or "grades_v1"
    tr = "".join(f"<tr><td>{E(r['condition'])}</td><td>{E(r['agent_answer'])}</td><td><b>{E(r['doctor_answer'] or '-')}</b></td>"
                 f"<td>{E(r.get('severity') or '')}</td><td>{E(r.get('comment') or '')}</td></tr>" for r in rows if r["condition_key"] != "_overall")
    overall = next((r["comment"] for r in rows if r["condition_key"] == "_overall"), "")
    return (f'<div class="card agentonly earlier"><details><summary><b>Earlier review of this case</b> &middot; {E(rows[0]["reviewer"])}, '
            f'{E(rows[0]["saved_utc"][:10])} &middot; made on a page showing <b>{E(", ".join(cfgs) or "an earlier checklist")}</b> (form {E(form)})</summary>'
            '<div class="muted">Shown for reference only. These grades were given against the answers of that earlier page (second column), not against the '
            f'answers on this page. The file is kept unchanged ({E(path.name)}) '
            + ('and its answers are filled into the form above as a starting point: please check each one against this page.</div>' if prefilled
               else 'and nothing is copied into the form above.</div>')
            + f'<table><tr><th>condition</th><th>tool answer on that page</th><th>reviewer</th><th>severity</th><th>comment</th></tr>{tr}</table>'
            + (f'<div class="muted">Overall comment: {E(overall)}</div>' if overall else "") + "</details></div>")


def policy_pack_page(page: str, payload_html: str, chk: dict, ev: dict, pack: str, earlier: Path | None, prefilled: bool = False) -> str:
    """Additions for a policy-labelled pack. Everything before the scan payload only; payload and viewer code are untouched."""
    cut = page.index(payload_html)
    head, tail = page[:cut], page[cut:]
    for a, b in POLICY_DISPLAY_ONLY:
        head = head.replace(E(a), E(b)).replace(a, b)
    assert head.count("</style></head>") == 1
    head = head.replace("</style></head>", POLICY_CSS + "</style></head>", 1)
    tip = "<div class='tip agentonly'>"
    assert head.count(tip) == 1
    bar = (f'<div class="policybar"><b>Answer policy shown on this page: {E(chk["config_version"])}</b> &middot; pack {E(pack)} &middot; '
           'development page, open (the tool&rsquo;s answers are visible; not blinded, not an evaluation).<br>'
           + (('In this policy: &ldquo;present&rdquo; comes from a measurement, or, for pleural effusion, aortic and coronary calcification, from two sources that agree: '
           'AI reader NV-Reason-CT says present and the classifier is above its cut-off (or, for pleural effusion, the segmented fluid is at least 20 mL). '
           'A predicted outline alone gives &ldquo;cannot tell&rdquo;, with what it outlined listed; an empty or small outline never means &ldquo;absent&rdquo;; '
           'a classifier score alone never decides; dense lung is listed as tagged candidates. Limits and cut-offs are provisional, fitted on development scans, not validated. ') if chk.get("config_version", "") >= "checklist_v3_12" else
              ('In this policy: a classifier score is supporting evidence only; an empty or small predicted outline never means &ldquo;absent&rdquo;; '
               'a positive answer that rests only on a predicted outline is shown as &ldquo;suggested by segmentation&rdquo;; dense lung is listed as '
               'tagged candidates, not as findings. Limits are provisional reporting limits, not validated diagnostic boundaries. ')) +
           'Earlier packs (M1 to M4) show earlier policies and are kept unchanged.</div>')
    head = head.replace(tip, bar + tip, 1)
    if earlier:
        ep = earlier if earlier.is_absolute() else ROOT / earlier
        anchor = '<div class="card agentonly" id="measpanel">'
        assert head.count(anchor) == 1
        head = head.replace(anchor, earlier_grades_card(ep, ev["case_id"], prefilled) + "\n " + anchor, 1)
    return head + tail


READER_CSS = """
.aird{font-size:12.5px;color:#3d4a5c;margin-top:5px;padding-top:4px;border-top:1px dashed #d0d7de}
.aishown{color:#6e7781}
.aird[title],.st[title]{cursor:help}
.aird.aiused{color:#1a4b8c;border-top-color:#9db9ee}.aird.aiused .aishown{color:#1a4b8c;font-weight:600}
.aireport{white-space:pre-wrap;font:inherit;margin:8px 0 0;background:#f6f8fa;border:1px solid #d8dee4;border-radius:8px;padding:10px 12px}
"""


def reader_card(reader: dict) -> str:
    """Closed card with the AI reader's own report text, verbatim, and what the reader is. Display only."""
    cut = ", ".join(f"{v:g} mm {k.replace('_', ' ')}" for k, v in reader["outside_mm"].items())
    return (f'<div class="card agentonly" id="aireader"><details><summary><b>AI reader report ({E(reader["name"])})</b> '
            f'<span class="muted">&middot; shown only, not counted &middot; click to open or close</span></summary>'
            f'<p class="muted">{E(reader["name"])} is a research model. It read this scan as one 3D block at 2 mm resolution'
            + (f'; part of the scan lay outside its field ({E(cut)})' if cut else '')
            + '. It was trained on reports from the same hospital as this scan, so its wording resembles those reports. '
              'It can describe findings that are not in the image, and it was not offered &ldquo;cannot tell&rdquo;. '
              + (f'On this page its &ldquo;present&rdquo; is used as the tool&rsquo;s answer for: {E(", ".join(reader["used_for"]))} (policy decision, not validated). '
                 'For every other condition it is shown only. ' if reader.get("used_for") else 'Its answers do not change any answer of the tool on this page. ')
            + 'The text below is the model&rsquo;s own, unedited.</p>'
            f'<pre class="aireport">{E(reader["report"]) or "no report text"}</pre></details></div>')


def slab_source_info(case_id: str, gt: dict | None) -> dict:
    """What is known about slice thickness and other reconstructions, for the Details panel of viewer v6. Spacing comes from the affine."""
    info = {"reported_thickness_mm": v6.reported_thickness_mm((gt or {}).get("technique")),
            "thickness_source": "radiology report text (technique sentence)" if gt else None}
    md = sorted((ROOT / "data" / "raw" / "ct_rate" / "hf_cache").glob("**/metadata/validation_metadata.csv"))
    stem = case_id.rsplit("_", 1)[0] + "_"
    if md:
        import csv as _csv
        with open(md[0], newline="", encoding="utf-8") as fh:
            sib = {r["VolumeName"].split(".")[0]: float(r["ZSpacing"]) for r in _csv.DictReader(fh)
                   if r["VolumeName"].startswith(stem) and r["VolumeName"].split(".")[0] != case_id and r.get("ZSpacing")}
        info["other_reconstructions_z_spacing_mm"] = sib
        match = [k for k, z in sib.items() if abs(z - v6.AXIAL_MM) < 0.05]
        others = ", ".join(f"{z:g} mm apart" for z in sorted(set(sib.values())))
        info["reconstruction_note"] = (
            (f"The data set holds a reconstruction with slices 2.5 mm apart ({', '.join(match)}); it is not on this page, which shows a calculated view."
             if match else
             "No 2.5 mm reconstruction of this scan exists in the data set" + (f" (the other one has slices {others})" if sib else "")
             + ", so the 2.5 mm view is calculated from these slices."))
    else:
        info["reconstruction_note"] = "Whether a 2.5 mm reconstruction exists was not checked; the 2.5 mm view is calculated from these slices."
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--checklist-file", default="checklist.json")
    ap.add_argument("--config", default=None, help="checklist config for the overlays (default: the run's)")
    ap.add_argument("--rule-based", action="store_true", help="report built from the checklist")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--grades-csv", type=Path, default=None, help="grades CSV a reviewer already sent for this case; shown filled in")
    ap.add_argument("--viewer", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7"], default="v1",
                    help="v2 = location-first page with grading form v2; v3 = v2 with the lazy, lossless loader (WebGL); "
                         "v4 = v2 with the 2D canvas viewer (no WebGL); v5 = v4 plus an optional axial sliding-slab MIP mode "
                         "(scripts/viewer_v5.py); v6 = v4 opening on a derived 2.5 mm axial mean slab, with a 10 mm MIP and native slices under Details "
                         "(scripts/viewer_v6.py); v7 = v6 plus a bar with the outlines of TotalSegmentator and, with --second-seg, of a second "
                         "segmentation model (scripts/viewer_v7.py); earlier pages are unchanged")
    ap.add_argument("--blinded", action="store_true", help="v2 / v3: staged page, tool output hidden until step 1 is locked")
    ap.add_argument("--corrected-wording", action="store_true",
                    help="v4 only: corrected review copy. Displayed text no longer asserts inspiration ('poorly inflated', 'shallow breath') and no longer "
                         "shows the pericardial 'greatest thickness' with small / moderate / large wording. Evidence, checklist files, answers and the viewer "
                         "code are unchanged; only displayed strings differ (lung_ct.agent.wording)")
    ap.add_argument("--policy-pack", default=None, metavar="PACK",
                    help="v4 only: page of a policy-labelled pack (checklist_v3_8 and later). The page states the checklist version and pack, "
                         "the grading form keeps its own storage and records the policy in every CSV row, and earlier grades are never filled in")
    ap.add_argument("--prefill-earlier-grades", action="store_true",
                    help="with --earlier-grades-csv: fill the reviewer's earlier grades into the form. The page says which policy they were made "
                         "on, and every exported row records whether the answer was carried over unchanged, changed, or entered on this page")
    ap.add_argument("--reference-columns", action="store_true",
                    help="with --policy-pack: an always-visible column with the CT-RATE report label, compared with the answer on this page. "
                         "The label is derived from the report text, and the column header says so")
    ap.add_argument("--extra-evidence-rows", type=Path, default=None,
                    help="JSON with additional evidence rows computed after the run (dense-lung inventory replay, rows E92-E94). Merged for "
                         "display only; evidence.json is not rewritten")
    ap.add_argument("--earlier-grades-csv", type=Path, default=None,
                    help="with --policy-pack: grades the reviewer sent earlier for this case, shown read-only with the policy they were made on")
    ap.add_argument("--reader-run", default=None, metavar="NAME=RUNDIR",
                    help="with --policy-pack: show ONE AI reader's answers (run laid out like scripts/run_nvreason.py: <run>/<case>/vlm_parsed.json and "
                         "report.txt) under each tool answer and its report in a closed card. Display only: checklist files and answers are not changed")
    ap.add_argument("--second-seg", type=Path, default=None, metavar="RUNDIR",
                    help="with --viewer v7: run folder of scripts/run_nvsegment.py holding this case. Its outlines are shown in the "
                         "segmentation bar for comparison. Display only: no answer or measurement uses them")
    args = ap.parse_args()
    if args.second_seg and args.viewer != "v7":
        ap.error("--second-seg needs --viewer v7")
    if args.reader_run and not args.policy_pack:
        ap.error("--reader-run needs --policy-pack")
    if (args.policy_pack or args.earlier_grades_csv) and args.viewer not in ("v4", "v5", "v6", "v7"):
        ap.error("--policy-pack / --earlier-grades-csv need --viewer v4")
    if args.policy_pack and (args.grades_csv or args.blinded):
        ap.error("--policy-pack cannot be combined with --grades-csv or --blinded")
    if args.earlier_grades_csv and not args.policy_pack:
        ap.error("--earlier-grades-csv needs --policy-pack")
    if args.prefill_earlier_grades and not args.earlier_grades_csv:
        ap.error("--prefill-earlier-grades needs --earlier-grades-csv")
    if args.reference_columns and not args.policy_pack:
        ap.error("--reference-columns needs --policy-pack")
    for _k in ("grades_csv", "earlier_grades_csv"):          # a '*' in the file name stands for the reader's name: exactly one file must match
        _p = getattr(args, _k)
        if _p is not None and "*" in _p.name:
            _d = _p.parent if _p.is_absolute() else ROOT / _p.parent
            _m = sorted(_d.glob(_p.name))
            if len(_m) != 1:
                ap.error(f"--{_k.replace('_', '-')} {_p}: {len(_m)} files match, need exactly one")
            setattr(args, _k, _m[0])
    V3 = args.viewer == "v3"
    V4 = args.viewer in ("v4", "v5", "v6", "v7")    # v5 = the v4 canvas page plus the MIP mode; v6 = plus the 2.5 mm mean slab; same build path
    vv = {"v5": v5, "v6": v6, "v7": v7}.get(args.viewer, v4)    # page script, html and css of the canvas viewer
    V2 = args.viewer in ("v2", "v3", "v4", "v5", "v6", "v7")    # v3 to v6 share the v2 layout and grading form
    VIEWER_BUILD = vv.VIEWER_BUILD if V4 else v3.VIEWER_BUILD if V3 else v2.VIEWER_BUILD
    if args.corrected_wording and not V4:
        ap.error("--corrected-wording needs --viewer v4")
    if args.blinded and not V2:
        ap.error("--blinded needs --viewer v2 or v3")
    if args.blinded and args.grades_csv:
        ap.error("--blinded cannot be combined with --grades-csv")
    run = args.run if args.run.is_absolute() else ROOT / args.run
    ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
    chk = json.loads((run / args.checklist_file).read_text(encoding="utf-8"))
    qc = json.loads((run / "qc.json").read_text(encoding="utf-8"))
    if args.extra_evidence_rows:       # display only: rows computed after the run; the stored evidence.json stays as it is
        xp = args.extra_evidence_rows if args.extra_evidence_rows.is_absolute() else ROOT / args.extra_evidence_rows
        extra_rows = json.loads(xp.read_text(encoding="utf-8"))
        if extra_rows["case_id"] != ev["case_id"] or extra_rows["run"] != run.name:
            raise SystemExit(f"{xp}: rows belong to {extra_rows['case_id']} / {extra_rows['run']}, not to this run")
        new_by_id = {r["id"]: r for r in extra_rows["rows"]}
        ev["rows"] = [new_by_id.pop(r["id"], r) for r in ev["rows"]] + list(new_by_id.values())
    cfg_path = args.config or f"configs/{chk['config_version']}.json"
    ck = load_checklist_config(ROOT / cfg_path)
    if args.rule_based or not (run / "final_report.json").exists():
        rep, src = rule_based_checklist_report(chk), f"rule_based ({chk['config_version']})"
    else:
        fr = json.loads((run / "final_report.json").read_text(encoding="utf-8"))
        rep, src = fr["report"], fr["source"]

    # ---- scan and overlays, recomputed with the same code as the agent ----
    ct = nib.load(str(ROOT / qc["ct"]["file"].replace("\\", "/")))
    hu = np.asanyarray(ct.dataobj).astype(np.float32)
    seg = ROOT / "data" / "derived" / ev["case_id"] / "segmentation"

    def load(rel):
        p = seg / f"{rel}.nii.gz"
        return (np.asanyarray(nib.load(str(p)).dataobj) > 0) if p.exists() else None
    lobes = {n: load(f"total/{n}") for n in LOBES}
    lab = np.zeros(hu.shape, np.uint8)
    for n, (_, side_label, _) in LOBES.items():
        lab[lobes[n]] = side_label
    hu_thr, _, noise = image_noise(hu, ct, seg, ck)
    _, extra = measure_conditions(hu_thr, ct, lab, lobes, {n: v[2] for n, v in LOBES.items()}, seg, ck, noise)
    masks = dict(extra["arrays"])
    masks["pleural"] = load("pleural_pericard_effusion/pleural_effusion")
    masks["pericardial"] = load("pleural_pericard_effusion/pericardial_effusion")
    masks["heart"] = load("total/heart")
    if any(c["key"] == "lung_nodule" and "present_at_least" in c for c in ck["conditions"]):      # v3_10 and later; earlier pages are unchanged
        masks["nodules"] = load("lung_nodules/lung_nodules")

    # crop to the body (+ margin) to keep the file reasonable; affine shifted accordingly
    body = ndimage.binary_opening(hu > -500, iterations=2) | (lab > 0)
    idx = np.argwhere(body)
    lo = np.maximum(idx.min(0) - 8, 0)
    hi = np.minimum(idx.max(0) + 9, hu.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    aff = ct.affine.copy()
    aff[:3, 3] = (ct.affine @ np.append(lo, 1.0))[:3]
    ct_arr = np.clip(hu[sl], -1024, 3071).astype(np.int16)   # same voxels in every viewer version
    payload_html, vmeta = "", {}
    if V4:
        vols = {}
        ornt, can_aff, orig_codes, oblique = v4.canonical_transform(aff, ct_arr.shape)   # refuses oblique scans
        payload_html, ct_meta = v4.ct_payload(v4.to_canonical(ct_arr, ornt), can_aff, orig_codes, oblique)
    elif V3:
        vols = {}
        payload_html, ct_meta = v3.ct_payload(ct_arr, aff)
    else:
        vols = {"ct": nii_b64(ct_arr, aff)}
    del ct_arr
    ov, targets = [], {}
    neutral_dense = bool((load_checklist_config(ROOT / cfg_path).get("measurements") or {}).get("dense_lung_inventory"))   # v3_6 and later
    for key, label, cmap, sw, on in OVERLAYS:
        if neutral_dense and key == "compressed":      # spatial tag, no diagnosis named
            label = "Dense-lung candidates next to predicted pleural fluid (tag)"
        if neutral_dense and key == "dependent":
            label = "Posterior band (tag)"
        m = masks.get(key)
        if m is None or not np.asarray(m).any():
            continue
        m = np.asarray(m, bool)[sl]
        if V4:
            h, vmeta[key] = v4.overlay_payload(key, v4.to_canonical(m.astype(np.uint8), ornt))
            payload_html += h
        elif V3:
            h, vmeta[key] = v3.overlay_payload(key, m.astype(np.uint8), aff)
            payload_html += h
        else:
            vols[key] = nii_b64(m.astype(np.uint8), aff)
        targets[key] = target_mm(m, aff)
        ov.append({"key": key, "label": label, "cmap": cmap, "swatch": sw, "on": on})
    seg_meta = None
    if args.viewer == "v7":            # display only: label maps of the two segmentation models, on the same cropped canonical grid as the scan
        def can8(a):
            return v4.to_canonical(np.ascontiguousarray(np.asarray(a, np.uint8)[sl]), ornt)
        nvcfg, seg_info = json.loads((ROOT / "configs" / "nvsegment_ct_v1.json").read_text(encoding="utf-8")), {}
        ts_masks = {x: load(rel) for rel, x in (("total/lung_upper_lobe_left", 28), ("total/lung_lower_lobe_left", 29), ("total/lung_upper_lobe_right", 30),
                                                ("total/lung_middle_lobe_right", 31), ("total/lung_lower_lobe_right", 32), ("total/trachea", 57),
                                                ("total/heart", 115), ("total/aorta", 6))}
        ts_lab, seg_info["totalsegmentator_overlapping_voxels"] = v7.ts_label_map({x: m for x, m in ts_masks.items() if m is not None})
        seg_info["totalsegmentator_missing_files"] = [nvcfg["class_names"][str(x)] for x, m in ts_masks.items() if m is None]
        maps = {"tsseg": can8(ts_lab)}
        aw = load("lung_vessels/lung_airways")
        if aw is not None:
            maps["tsaw"] = can8(aw.astype(np.uint8) * 132)
        if args.second_seg:
            sdir = (args.second_seg if args.second_seg.is_absolute() else ROOT / args.second_seg) / ev["case_id"]
            sman = json.loads((sdir / "seg_manifest.json").read_text(encoding="utf-8"))
            if sman["status"] != "ok":
                raise SystemExit(f"{sdir}: the second segmentation failed for this case ({sman['issues']}); build the page without --second-seg and say so")
            nvcfg = json.loads((sdir.parent / "run_manifest.json").read_text(encoding="utf-8"))["config"]      # the config that run used

            def same_grid(path):
                img = nib.load(str(path))
                if tuple(img.shape[:3]) != tuple(ct.shape[:3]) or np.abs(np.asarray(img.affine, float) - np.asarray(ct.affine, float)).max() > 1e-3:
                    raise SystemExit(f"{path}: not on the grid of the scan; an outline is never drawn without that check")
                return np.asanyarray(img.dataobj)
            maps["nvseg"] = can8(same_grid(sdir / "nvseg_labels.nii.gz"))
            extra = {x: same_grid(sdir / "extra" / (re.sub(r"[^a-z0-9]+", "_", nvcfg["class_names"][str(x)].lower()).strip("_") + ".nii.gz")) > 0
                     for x in nvcfg["passes"]["extra"]["labels"]}
            merged, seg_info["second_model_extra_classes_overlapping_voxels"] = v7.merge_extra(extra, nvcfg["viewer_extra_priority"]["order"])
            maps["nvsegx"] = can8(merged)
            seg_info.update(second_model=nvcfg["model"]["model_id"], second_model_revision=nvcfg["model"]["revision"],
                            second_model_run=str(args.second_seg).replace("\\", "/"), second_model_files_sha256=sman["files_sha256"],
                            second_model_note="shown for comparison only; no answer, measurement or grade on this page uses it")
        h, seg_vols, seg_ov, seg_meta = v7.seg_bundle(maps, nvcfg["viewer_groups"] if args.second_seg else [], nvcfg["class_names"], seg_info)
        payload_html += h
        vmeta.update(seg_vols)
        ov += seg_ov
        del maps
    del hu, hu_thr, masks

    # ---- checklist rows ----
    gt = None
    cj = ROOT / "data" / "raw" / "ct_rate" / "cases.json"
    if cj.exists():
        gt = json.loads(cj.read_text(encoding="utf-8"))["cases"].get(ev["case_id"])
    if args.blinded:
        gt = None  # staged pages never contain the original radiology report or its labels
    labels = set(gt["labels"]) if gt else None
    rows_html = ""
    earlier_rows, earlier_cfg = {}, ""
    if args.earlier_grades_csv:
        import csv as _csv
        ep_ = args.earlier_grades_csv if args.earlier_grades_csv.is_absolute() else ROOT / args.earlier_grades_csv
        with open(ep_, newline="", encoding="utf-8-sig") as fh:
            _er = list(_csv.DictReader(fh))
        earlier_rows = {r["condition_key"]: r for r in _er}
        earlier_cfg = ", ".join(sorted({r["checklist_config"] for r in _er if r.get("checklist_config")}))

    def ref_cells(it: dict, cat: str, lp) -> str:
        """Reference cell: CT-RATE report label. Comparison uses the CURRENT assessment category, so 'suggested' is never counted as a confirmed agreement."""
        def tag(pos):
            if cat in ("cannot tell", "cannot assess"):
                return "open", "tool undecided"
            if cat == "suggested":
                return "open", "tool: suggested only"
            if cat == "present":
                return ("agree", "same as tool") if pos else ("extra", "tool positive, reference not")
            return ("missed", "tool says absent") if pos else ("agree", "same as tool")
        if lp is None:
            c1 = "<td class='agentonly refcol'><span class='muted'>no label for this item</span></td>"
        else:
            c, t = tag(lp)
            c1 = f"<td class='agentonly refcol'><b>{'mentioned' if lp else 'not mentioned'}</b><br><span class='m {c}'>{E(t)}</span></td>"
        return c1
    conds = {c["key"]: c for c in ck["conditions"]}
    ev_rows = {r["id"]: r for r in ev["rows"]}
    reader = None
    if args.reader_run:                # display only: the checklist file and its answers are read, never changed
        rname, _, rdir = args.reader_run.partition("=")
        rcase = (ROOT / rdir if not Path(rdir).is_absolute() else Path(rdir)) / ev["case_id"]
        rres = json.loads((rcase / "vlm_result.json").read_text(encoding="utf-8"))
        if rres.get("ct_sha256") != qc["ct"]["sha256"]:
            raise SystemExit(f"{rcase}: the reader ran on a different file than this page shows")
        reader = {"name": rname, "run": rdir, "conditions": json.loads((rcase / "vlm_parsed.json").read_text(encoding="utf-8"))["conditions"],
                  "report": (rcase / "report.txt").read_text(encoding="utf-8") if (rcase / "report.txt").exists() else "",
                  "outside_mm": {k: v for k, v in (rres.get("outside_cube_mm") or {}).items() if v},
                  "used_for": [c["label"] for c in ck["conditions"] if c.get("reader_decides_present") == rname
                               or (c.get("two_source_present") or {}).get("reader") == rname]}
        want_src = (chk.get("reader_sources") or {}).get(rname)
        if reader["used_for"] and want_src != rdir:
            raise SystemExit(f"this checklist file was built from reader run {want_src}, the page was asked to show {rdir}")

    def reader_line(key: str) -> str:
        r = reader and reader["conditions"].get(key)
        if not r:
            return ""
        side = r.get("side")
        where = {"right": ", right side", "left": ", left side", "both": ", both sides"}.get(side, "")
        used = bool((chk_by_key.get(key) or {}).get("reader_decided"))
        tip = reader_tip(reader["name"], r)
        return (f"<div class='aird agentonly{' aiused' if used else ''}' title=\"{E(tip)}\">AI reader {E(reader['name'])}: <b>{E(r['status'].replace('_', ' '))}</b>{where} "
                f"<span class='aishown'>({'used for this answer' if used else 'shown only, not counted'})</span></div>")

    def reader_tip(name: str, r: dict) -> str:
        sc = r.get("score_yes")
        return (f"{name} score for 'present': {sc:.2f} (0 = no, 1 = yes; uncalibrated, not a probability of disease)" if isinstance(sc, (int, float))
                else f"{name}: no score stored for this answer")
    chk_by_key = {i["key"]: i for i in chk["items"]}
    for it in chk["items"]:
        if reader and it["key"] in reader["conditions"]:      # the stored basis says "no reader answer" because no reader ran when it was written
            it = {**it, "basis": it["basis"].replace("no reader answer", f"AI reader {reader['name']}: "
                                                     f"{reader['conditions'][it['key']]['status'].replace('_', ' ')} (shown only, not counted)")}
        lp = None if labels is None or not it.get("ct_rate_label") else it["ct_rate_label"] in labels
        mc, mt = match(it["status"], lp)
        jk = JUMP.get(it["key"])
        jump = jk if jk in targets else ""
        if it["key"] == "lung_density" and "consolidation" not in targets and "ground_glass" in targets:
            jump = "ground_glass"
        stmt = it.get("statement") or it.get("location") or ""
        if V2 and args.policy_pack:        # badge class from the CURRENT assessment: a mask-based positive is 'suggested', not 'present'
            cat = assessment_category(it)
            rows_html += (f"<tr class='{'jump' if jump else ''}' data-jump='{jump}'>"
                          f"<td><b>{E(it['label'])}</b>{SHOW_V2 if jump else ''}</td>"
                          f"<td class='agentonly'><span class='st {cat.replace(' ', '-')}'"
                          + (f" title=\"{E('Answer taken from AI reader ' + it['reader_decided']['reader'] + '. ' + reader_tip(it['reader_decided']['reader'], it['reader_decided']))}\""
                             if it.get("reader_decided") else "")
                          + f">{E(it.get('display', it['status']))}</span>"
                          f"{v2.answer_cell(it, conds.get(it['key'], {}), ev_rows)}{reader_line(it['key'])}</td>"
                          f"<td class='whycol agentonly'>{E(stmt)}<div class='why'>{E(it['basis'])}</div></td>"
                          + (ref_cells(it, cat, lp) if args.reference_columns else
                             (f"<td class='whycol agentonly'><span class='m {mc}'>{E(mt)}</span></td>" if labels is not None else "")) + "</tr>")
            continue
        if V2:
            rows_html += (f"<tr class='{'jump' if jump else ''}' data-jump='{jump}'>"
                          f"<td><b>{E(it['label'])}</b>{SHOW_V2 if jump else ''}</td>"
                          f"<td class='agentonly'><span class='st {it['status'].replace(' ', '-')}'>{E(it.get('display', it['status']))}</span>"
                          f"{v2.answer_cell(it, conds.get(it['key'], {}), ev_rows)}</td>"
                          f"<td class='whycol agentonly'>{E(stmt)}<div class='why'>{E(it['basis'])}</div></td>"
                          + (f"<td class='whycol agentonly'><span class='m {mc}'>{E(mt)}</span></td>" if labels is not None else "") + "</tr>")
            continue
        rows_html += (f"<tr class='{'jump' if jump else ''}' data-jump='{jump}'>"
                      f"<td><b>{E(it['label'])}</b>{'<br><span class=show>&#128065; Show in scan</span>' if jump else ''}</td>"
                      f"<td><span class='st {it['status'].replace(' ', '-')}'>{E(it.get('display', it['status']))}</span>"
                      f"{answer_details(it, conds.get(it['key'], {}), ev_rows)}</td>"
                      f"<td class='whycol'>{E(stmt)}<div class='why'>{E(it['basis'])}</div></td>"
                      + (f"<td><span class='m {mc}'>{E(mt)}</span></td>" if labels is not None else "") + "</tr>")
    rep_html = (f"<p class='impr'>{E(rep.get('impression', ''))}</p><ul>"
                + "".join(f"<li>{E(f['text'])}</li>" for f in rep.get("findings", [])) + "</ul>")
    gt_html = ""
    if gt:
        gt_html = (f"<details><summary><b>Radiologist report (CT-RATE), not shown to the agent</b></summary>"
                   f"<p><b>Labels:</b> {E(', '.join(gt['labels']) or 'none')}</p>"
                   f"<p><b>Impression:</b> {E(gt.get('impression') or '-')}</p>"
                   f"<p class='muted'>{E(gt.get('findings') or '')}</p></details>")
    data = {"vols": vols, "overlays": ov, "targets": targets}
    grading_css = GRADING_CSS + MEAS_CSS
    meas_html = measurements_card(ev)
    if V2:
        grading_css = GRADING_CSS + v2.CSS + gv2.CSS
        meas_html = v2.measurement_panel(ev, chk)
        grading_js = gv2.grading_script(ev["case_id"], run.name, chk["config_version"], chk, VIEWER_BUILD,
                                        ({**gv2.load_grades_csv(args.earlier_grades_csv if args.earlier_grades_csv.is_absolute() else ROOT / args.earlier_grades_csv,
                                                                 ev["case_id"], chk), "made_on": earlier_cfg or "an earlier checklist"}
                                         if args.prefill_earlier_grades else
                                         gv2.load_grades_csv(args.grades_csv, ev["case_id"], chk) if args.grades_csv else None),
                                        blinded=args.blinded, allow_policy_prefill=args.prefill_earlier_grades,
                                        **({"policy": {"config": chk["config_version"], "pack": args.policy_pack}} if args.policy_pack else {}))
        head_html = ("<tr><th>condition</th><th class='agentonly'>agent answer and location</th>"
                     "<th class='whycol agentonly'>what it means / why</th>"
                     + (("<th class='agentonly refcol'>CT-RATE report label<div class='refnote'>from the report text; &ldquo;not mentioned&rdquo; is not &ldquo;absent&rdquo;</div></th>")
                        if args.reference_columns else
                        ("<th class='whycol agentonly'>vs report</th>" if labels is not None else "")) + "</tr>")
        tip_html = ("<div class='tip agentonly'><b>&#128073; How to read this page:</b> each answer shows <b>where</b> first "
                    "(lobe names). A region is lung in an HU range picked by the tool, not a confirmed finding. Numbers are "
                    "under each answer's <b>Numbers</b> button and in the <b>Measurement panel</b>. Click a row with a blue "
                    "<span class='show'>&#128065; Show in scan</span> button to switch on its outline and jump to it. "
                    "The scan opens without outlines.<br><button class='panelbtn' type='button' "
                    "onclick=\"document.getElementById('measpanel').scrollIntoView({behavior:'smooth'})\">"
                    "Open measurement panel &darr;</button></div>")
    else:
        grading_js = grading_script(ev["case_id"], run.name, chk["config_version"], chk,
                                    load_grades_csv(args.grades_csv, ev["case_id"], chk) if args.grades_csv else None)
        head_html = ("<tr><th>condition</th><th>agent answer</th><th class=\"whycol\">what it means / why</th>"
                     + ("<th>vs report</th>" if labels is not None else "") + "</tr>")
        tip_html = ('<div class="tip"><b>&#128073; Tip for reviewers:</b> click any row with a blue <span class="show">&#128065; '
                    "Show in scan</span> button. The viewer on the left switches on that finding's coloured outline and jumps "
                    "to it. The scan opens without any outlines; switch them on or off in the bar above the scan. Scroll the "
                    "mouse wheel to move through the slices.</div>")
    ao = " agentonly" if V2 else ""
    # extra script lines for viewer v2 only, so that a viewer v1 page is rebuilt exactly as before
    v2_js = (""" document.querySelectorAll('.mbtn').forEach(b => b.onclick = e => {   // v2: numbers open on click, not on hover
   e.stopPropagation(); const d = b.nextElementSibling; d.hidden = !d.hidden; b.textContent = d.hidden ? 'Numbers' : 'Hide numbers';
 });
 document.querySelectorAll('.mdet').forEach(d => d.onclick = e => e.stopPropagation());
""" if V2 else "")
    v2_guard = ("   if (document.body.classList.contains('stage1')) return;          // staged page: no outline before step 1 is locked\n"
                if V2 else "")
    body_cls = ' class="stage1"' if args.blinded else ""
    build_note = f" &middot; {VIEWER_BUILD}, form {gv2.FORM_VERSION}{', staged' if args.blinded else ''}" if V2 else ""
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Interactive review - {E(ev['case_id'])}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;font:13px/1.35 'Segoe UI',Arial,sans-serif;background:#0f1115;color:#e6e6e6}}
.page{{display:grid;grid-template-columns:1.35fr 1fr;height:100vh}}
.viewer{{display:flex;flex-direction:column;border-right:1px solid #2a2f3a;min-width:0}}
.bar{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:8px 10px;background:#161a22;border-bottom:1px solid #2a2f3a}}
.bar b{{font-size:12px;color:#9aa4b2;margin-right:2px}}
button{{background:#232937;color:#e6e6e6;border:1px solid #333b4d;border-radius:6px;padding:4px 9px;cursor:pointer;font:inherit}}
button.on{{background:#2f6feb;border-color:#2f6feb}}
.ov{{display:inline-flex;align-items:center;gap:5px;padding:3px 8px;border-radius:6px;border:1px solid #333b4d;cursor:pointer;user-select:none}}
.ov.off{{opacity:.45}}.sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
canvas{{flex:1;width:100%;min-height:0;display:block}}
#loc{{padding:5px 10px;font:12px Consolas,monospace;color:#9aa4b2;background:#161a22;border-top:1px solid #2a2f3a}}
.side{{overflow:auto;padding:10px 12px;background:#f4f5f7;color:#1f2328}}
h1{{font-size:18px;margin:0 0 2px}}.muted{{color:#66707a;font-size:11.5px}}
.card{{background:#fff;border:1px solid #d5d9de;border-radius:10px;padding:9px 10px;margin-top:9px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #eceef1;padding:4px;text-align:left;vertical-align:top}}
tr.jump{{cursor:pointer}}tr.jump:hover{{background:#eef4ff}}tr.sel{{background:#dbe8ff}}.show{{display:inline-block;margin-top:4px;background:#2f6feb;color:#fff;font-weight:600;font-size:12px;padding:3px 10px;border-radius:12px;white-space:nowrap}}
tr.jump:hover .show{{background:#1a56c4}}
.tip{{background:#fff4cc;border:1px solid #f0d36b;border-radius:10px;padding:8px 10px;margin-top:8px;font-size:13.5px;color:#3d3000}}
.why{{font-size:11px;color:#57606a}}
.hidewhy .whycol{{display:none}}.whybtn{{background:#fff;color:#57606a;border-color:#c9d1d9;font-size:11.5px;padding:2px 8px;margin-bottom:5px}}
.st{{display:inline-block;border-radius:9px;padding:0 8px;font-size:11.5px;font-weight:600;white-space:nowrap}}
.present{{background:#ffd8d3}}.absent{{background:#dcf2e0}}.cannot-tell{{background:#eceef1}}.cannot-assess{{background:#e3e0f7}}
.m{{display:inline-block;border-radius:9px;padding:0 6px;font-size:11px;white-space:nowrap}}
.agree{{background:#d6f0da}}.missed{{background:#fbd5d0}}.extra{{background:#fde7c4}}.open{{background:#eceef1}}
.impr{{font-weight:600;margin:0 0 4px}}ul{{margin:4px 0 0 16px;padding:0}}
#err{{display:none;padding:20px;color:#ffb4a8}}
{grading_css}
</style></head><body{body_cls}><div class="page">
<div class="viewer">
 <div class="bar"><b>Window</b>
  <button data-w="-1350,150" class="on">Lung</button><button data-w="-160,240">Soft tissue</button><button data-w="-500,1300">Bone</button>
  <b style="margin-left:10px">View</b>
  <button data-v="0" class="on">Axial</button><button data-v="1">Coronal</button><button data-v="2">Sagittal</button><button data-v="3">3 planes</button>
 </div>
 <div class="bar{ao}" id="ovbar"><b>Overlays (off to start; click to show)</b></div>
 <canvas id="gl"></canvas><div id="err">This viewer needs WebGL2 (current Chrome or Edge).</div>
 <div id="loc">Scroll: next/previous slice &middot; click: move cross-hair &middot; right-drag: brightness/contrast</div>
</div>
<div class="side">
 <h1>{E(ev['case_id'])} &mdash; ThoraxCheck condition checklist</h1>
 <div class="muted">Research use only &middot; scan type: {E(chk['scan_type'])} &middot; {E(chk['config_version'])} (provisional) &middot; report: {E(src)} &middot; patient's right is on the image left{build_note}</div>
 {tip_html}
 <div class="card"><button id="whybtn" class="whybtn{ao}" onclick="var t=document.getElementById('chk');this.textContent=t.classList.toggle('hidewhy')?'Show what it means / why':'Hide what it means / why'">Show what it means / why</button><table id="chk" class="hidewhy">{head_html}{rows_html}</table></div>
 {meas_html}
 <div class="card{ao}"><b>Report</b>{rep_html}</div>
 <div class="card">{gt_html}<div class="muted">Contains CT-RATE images and report text (CC BY-NC-SA, research use): keep within the research group.</div></div>
</div></div>
<script>
/* NiiVue 0.69.0, BSD-2-Clause, (c) NiiVue contributors, https://github.com/niivue/niivue (see scripts/vendor/NIIVUE_LICENSE.txt) */
{NIIVUE.read_text(encoding='utf-8')}
</script>
<script>
const DATA = {json.dumps(data)};
(async () => {{
 const nv = new niivue.Niivue({{isRadiologicalConvention: true, backColor: [0, 0, 0, 1], crosshairColor: [1, 0.85, 0.1, 1],
   show3Dcrosshair: false, isColorbar: false, multiplanarShowRender: niivue.SHOW_RENDER.NEVER}});
 try {{ await nv.attachTo('gl'); }} catch (e) {{ document.getElementById('err').style.display = 'block'; return; }}
 const ct = await niivue.NVImage.loadFromBase64({{base64: DATA.vols.ct, name: 'ct.nii.gz', colormap: 'gray', cal_min: -1350, cal_max: 150}});
 nv.addVolume(ct);
 const idx = {{}};
 for (const o of DATA.overlays) {{
   const el = document.createElement('span'); el.className = 'ov' + (o.on ? '' : ' off'); el.dataset.k = o.key;
   el.innerHTML = '<span class="sw" style="background:' + o.swatch + '"></span>' + o.label;
   el.onclick = () => setOv(o.key, el.classList.contains('off'));
   document.getElementById('ovbar').appendChild(el);
 }}
 let busy = Promise.resolve();
 function setOv(k, on) {{              // an overlay is added to the viewer the first time it is switched on
   busy = busy.then(async () => {{
     if (on && idx[k] === undefined) {{
       const o = DATA.overlays.find(x => x.key === k);
       const img = await niivue.NVImage.loadFromBase64({{base64: DATA.vols[k], name: k + '.nii.gz', colormap: o.cmap,
         opacity: 0.55, cal_min: 0.5, cal_max: 1}});
       nv.addVolume(img); idx[k] = nv.volumes.length - 1;
     }}
     if (idx[k] !== undefined) nv.setOpacity(idx[k], on ? 0.55 : 0);
     document.querySelector('.ov[data-k="' + k + '"]').classList.toggle('off', !on);
   }});
   return busy;
 }}
 for (const o of DATA.overlays) if (o.on) setOv(o.key, true);
 nv.setSliceType(nv.sliceTypeAxial);
 nv.onLocationChange = d => {{ document.getElementById('loc').textContent = d.string || ''; }};
 document.querySelectorAll('[data-w]').forEach(b => b.onclick = () => {{
   const [lo, hi] = b.dataset.w.split(',').map(Number);
   nv.volumes[0].cal_min = lo; nv.volumes[0].cal_max = hi; nv.updateGLVolume();
   document.querySelectorAll('[data-w]').forEach(x => x.classList.toggle('on', x === b));
 }});
 document.querySelectorAll('[data-v]').forEach(b => b.onclick = () => {{
   nv.setSliceType(Number(b.dataset.v));
   document.querySelectorAll('[data-v]').forEach(x => x.classList.toggle('on', x === b));
 }});
{v2_js} document.querySelectorAll('tr.jump').forEach(tr => tr.onclick = async () => {{
{v2_guard}   const k = tr.dataset.jump, t = DATA.targets[k];
   if (!t) return;
   await setOv(k, true);
   nv.scene.crosshairPos = nv.mm2frac(t); nv.drawScene();
   if (nv.onLocationChange) nv.createOnLocationChange && nv.createOnLocationChange();
   document.querySelectorAll('tr.sel').forEach(x => x.classList.remove('sel')); tr.classList.add('sel');
 }});
}})();
</script>
{grading_js}
</body></html>"""
    if V4:
        # Same page; the viewer column, the viewer script and the payloads are replaced. No NiiVue code is embedded.
        a = page.index("<script>\n/* NiiVue 0.69.0")
        b = page.index("</script>", page.index("<script>\nconst DATA = ", a)) + len("</script>")
        assert page[b:].strip().startswith("<script>") and page.rstrip().endswith("</body></html>"), "unexpected page layout"
        meta = {"build": VIEWER_BUILD, "case_id": ev["case_id"], "ct": ct_meta, "vols": vmeta, "overlays": ov, "targets": targets}
        if seg_meta is not None:
            meta["seg"] = seg_meta
        if args.viewer in ("v6", "v7"):
            meta["slab"] = slab_source_info(ev["case_id"], gt)
        page = (page[:a] + "<script>\n" + vv.EARLY_JS + "\n</script>\n" + grading_js + "\n" + payload_html
                + vv.viewer_script(meta) + "\n</body></html>")
        va, vb = page.index('<div class="viewer">'), page.index('<div class="side">')
        page = page[:va] + vv.viewer_html(True) + page[vb:]
        page = page.replace("</style></head>", vv.CSS + "</style></head>", 1)
        page = page.replace("switch them on or off in the bar above the scan", "switch one on or off in the bar above the scan")
        # Order of the cards under the checklist (reviewer request 2026-11-22): radiologist report first, then the
        # measurement panel (collapsible, closed at first), then the tool's report. v1-v3 pages keep their order.
        card_meas = meas_html
        card_rep = f'<div class="card{ao}"><b>Report</b>{rep_html}</div>'
        card_gt = (f'<div class="card">{gt_html}<div class="muted">Contains CT-RATE images and report text (CC BY-NC-SA, research use): '
                   'keep within the research group.</div></div>')
        old = card_meas + "\n " + card_rep + "\n " + card_gt
        assert page.count(old) == 1, "unexpected card layout"
        head = '<div class="card agentonly" id="measpanel"><b>Measurement panel</b> '
        assert card_meas.startswith(head) and card_meas.endswith("</div>")
        rest = card_meas[len(head):-len("</div>")]
        cut_at = rest.index("</span>") + len("</span>")
        meas4 = ('<div class="card agentonly" id="measpanel"><details id="measdet"><summary><b>Measurement panel</b> '
                 + rest[:cut_at] + ' <span class="muted">&middot; click to open or close</span></summary>' + rest[cut_at:] + "</details></div>")
        page = page.replace(old, card_gt + "\n " + meas4 + "\n " + card_rep + ("<!--AIREADER-->" if reader else ""))
        jump_old = "document.getElementById('measpanel').scrollIntoView({behavior:'smooth'})"
        assert page.count(jump_old) == 1
        page = page.replace(jump_old, "document.getElementById('measdet').open=true;" + jump_old)
        if args.corrected_wording:
            from lung_ct.agent.wording import banned_hits, neutral_text
            cut_ = page.index(payload_html)
            page = neutral_text(page[:cut_]) + page[cut_:]          # html and grading data only; payload and viewer code untouched
            left = banned_hits(page[:cut_])
            assert not left, f"corrected copy still contains: {left}"
            page = page.replace(f"{VIEWER_BUILD}, form grades_v2", f"{VIEWER_BUILD}, form grades_v2, corrected wording copy", 1)
        if args.policy_pack:
            page = policy_pack_page(page, payload_html, chk, ev, args.policy_pack, args.earlier_grades_csv, args.prefill_earlier_grades)
        if reader:                     # inserted last: the model's own words are shown as written, not passed through the wording filter
            assert page.count("<!--AIREADER-->") == 1
            page = page.replace("<!--AIREADER-->", "\n " + reader_card(reader))
            page = page.replace("</style></head>", READER_CSS + "</style></head>", 1)
        code_only = page.replace(payload_html, "")          # the base64 payload can contain any letter sequence by chance
        assert "niivue" not in code_only.lower() and "webgl" not in code_only.lower(), "viewer library code left in a v4 page"
    elif V3:
        # Same page; only the scan loader and the way the payloads are embedded differ. Order matters: the grading form
        # and the small UI script come BEFORE the large payload blocks, so the form works while the scan still loads.
        a = page.index("<script>\nconst DATA = ")
        b = page.index("</script>", a) + len("</script>")
        meta = {"build": VIEWER_BUILD, "case_id": ev["case_id"], "ct": ct_meta, "vols": vmeta, "overlays": ov, "targets": targets}
        tail = page[b:]
        assert tail.strip().startswith("<script>") and tail.rstrip().endswith("</body></html>"), "unexpected page layout"
        page = (page[:a] + "<script>\n" + v3.EARLY_JS + "\n</script>\n" + grading_js + "\n" + payload_html
                + v3.viewer_script(meta) + "\n</body></html>")
        page = page.replace("</style></head>", v3.CSS + "</style></head>", 1)
        c = '<canvas id="gl"></canvas>'
        assert page.count(c) == 1
        page = page.replace(c, v3.STATUS_HTML.split('<div id="vstat">')[0] + c + '<div id="vstat">' + v3.STATUS_HTML.split('<div id="vstat">')[1])
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"-> {out}  ({out.stat().st_size / 1e6:.1f} MB; overlays: {', '.join(o['key'] for o in ov)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
