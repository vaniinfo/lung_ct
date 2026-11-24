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
           'In this policy: a classifier score is supporting evidence only; an empty or small predicted outline never means &ldquo;absent&rdquo;; '
           'a positive answer that rests only on a predicted outline is shown as &ldquo;suggested by segmentation&rdquo;; dense lung is listed as '
           'tagged candidates, not as findings. Limits are provisional reporting limits, not validated diagnostic boundaries. '
           'Earlier packs (M1 to M4) show earlier policies and are kept unchanged.</div>')
    head = head.replace(tip, bar + tip, 1)
    if earlier:
        ep = earlier if earlier.is_absolute() else ROOT / earlier
        anchor = '<div class="card agentonly" id="measpanel">'
        assert head.count(anchor) == 1
        head = head.replace(anchor, earlier_grades_card(ep, ev["case_id"], prefilled) + "\n " + anchor, 1)
    return head + tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--checklist-file", default="checklist.json")
    ap.add_argument("--config", default=None, help="checklist config for the overlays (default: the run's)")
    ap.add_argument("--rule-based", action="store_true", help="report built from the checklist")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--grades-csv", type=Path, default=None, help="grades CSV a reviewer already sent for this case; shown filled in")
    ap.add_argument("--viewer", choices=["v1", "v2", "v3", "v4"], default="v1",
                    help="v2 = location-first page with grading form v2; v3 = v2 with the lazy, lossless loader (WebGL); "
                         "v4 = v2 with the 2D canvas viewer (no WebGL)")
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
    args = ap.parse_args()
    if (args.policy_pack or args.earlier_grades_csv) and args.viewer != "v4":
        ap.error("--policy-pack / --earlier-grades-csv need --viewer v4")
    if args.policy_pack and (args.grades_csv or args.blinded):
        ap.error("--policy-pack cannot be combined with --grades-csv or --blinded")
    if args.earlier_grades_csv and not args.policy_pack:
        ap.error("--earlier-grades-csv needs --policy-pack")
    if args.prefill_earlier_grades and not args.earlier_grades_csv:
        ap.error("--prefill-earlier-grades needs --earlier-grades-csv")
    if args.reference_columns and not args.policy_pack:
        ap.error("--reference-columns needs --policy-pack")
    V3 = args.viewer == "v3"
    V4 = args.viewer == "v4"
    V2 = args.viewer in ("v2", "v3", "v4")    # v3 and v4 share the v2 layout and grading form
    VIEWER_BUILD = v4.VIEWER_BUILD if V4 else v3.VIEWER_BUILD if V3 else v2.VIEWER_BUILD
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
    for it in chk["items"]:
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
                          f"<td class='agentonly'><span class='st {cat.replace(' ', '-')}'>{E(it.get('display', it['status']))}</span>"
                          f"{v2.answer_cell(it, conds.get(it['key'], {}), ev_rows)}</td>"
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
 <h1>{E(ev['case_id'])} &mdash; condition checklist</h1>
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
        page = (page[:a] + "<script>\n" + v4.EARLY_JS + "\n</script>\n" + grading_js + "\n" + payload_html
                + v4.viewer_script(meta) + "\n</body></html>")
        va, vb = page.index('<div class="viewer">'), page.index('<div class="side">')
        page = page[:va] + v4.viewer_html(True) + page[vb:]
        page = page.replace("</style></head>", v4.CSS + "</style></head>", 1)
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
        page = page.replace(old, card_gt + "\n " + meas4 + "\n " + card_rep)
        jump_old = "document.getElementById('measpanel').scrollIntoView({behavior:'smooth'})"
        assert page.count(jump_old) == 1
        page = page.replace(jump_old, "document.getElementById('measdet').open=true;" + jump_old)
        if args.corrected_wording:
            from lung_ct.agent.wording import banned_hits, neutral_text
            cut_ = page.index(payload_html)
            page = neutral_text(page[:cut_]) + page[cut_:]          # html and grading data only; payload and viewer code untouched
            left = banned_hits(page[:cut_])
            assert not left, f"corrected copy still contains: {left}"
            page = page.replace("checklist_viewer_v4_canvas2d, form grades_v2", "checklist_viewer_v4_canvas2d, form grades_v2, corrected wording copy", 1)
        if args.policy_pack:
            page = policy_pack_page(page, payload_html, chk, ev, args.policy_pack, args.earlier_grades_csv, args.prefill_earlier_grades)
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
