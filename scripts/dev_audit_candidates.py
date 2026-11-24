"""Development audit, parts 5 and 6 (core env): nodule candidates and pericardial fluid outlines, prepared for
candidate-level clinician review.

Nodules: every connected piece of the TotalSegmentator lung_nodules outline is a CANDIDATE, not a nodule. For each one
the script records exploratory review attributes (size, shape, attenuation, overlap with the dense-lung region),
matches it to the lobes the reviewer named (lobe level only; the reviewer's lobes are not assumed to be a complete
list) and to HoloCT's nodule outlines, and keeps unmatched and ambiguous candidates. Nothing is deleted.
"Equivalent-sphere diameter" is computed from the outline volume and is NOT a caliper diameter.

Pericardial fluid: the predicted outline of each case is described (volume, greatest thickness, attenuation inside,
where it lies around the heart) without deciding presence or size.

Two standalone review pages are written (images embedded, nothing sent anywhere, answers exported as CSV):
  nodule_candidate_review.html      per candidate: nodule / non-nodule / cannot adjudicate; mask usable / unusable /
                                    uncertain; lobe; optional caliper long and short axis (mm); comment
  pericardial_outline_review.html   per case: is the outline pericardial fluid; is an effusion present; size word

DEVELOPMENT MATERIAL: the cases were graded on open pages. No rule is derived here.

Usage: python scripts/dev_audit_candidates.py --cases valid_301_a_2 valid_541_a_2 valid_832_a_2 valid_988_a_2
           --pages-file ... --grades runs/grades_received --out runs/D1_dev_audit_20261122
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dev_audit_density import Case  # noqa: E402
from lung_ct.agent.checklist import load_checklist_config  # noqa: E402
from lung_ct.agent.conditions import _load, _to_ras, greatest_thickness_mm  # noqa: E402

LOBE_ORDER = ["RUL", "RML", "RLL", "LUL", "LLL"]


def page_numbers(c: Case, vox) -> dict:
    """Image numbers as the v4 viewer page shows them (the page holds the body-cropped volume)."""
    body = ndimage.binary_opening(c.hu_r > -500, iterations=2) | c.lung
    idx = np.argwhere(body)
    lo = np.maximum(idx.min(0) - 8, 0)
    hi = np.minimum(idx.max(0) + 9, np.array(body.shape))
    n = hi - lo
    i, j, k = (np.array(vox) - lo).astype(int)
    return {"axial": f"{int(n[2] - k)} of {int(n[2])}", "coronal": f"{int(n[1] - j)} of {int(n[1])}", "sagittal": f"{int(n[0] - i)} of {int(n[0])}"}


def png_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def three_plane(c: Case, centre, mask, half_mm: float, window, outline: bool, colour="#ff3b30") -> str:
    """Axial / coronal / sagittal crops around a voxel, radiological display, physical aspect ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i, j, k = [int(round(x)) for x in centre]
    h = [max(2, int(round(half_mm / z))) for z in c.zr]
    sx, sy, sz = [slice(max(0, a - b), a + b + 1) for a, b in zip((i, j, k), h)]
    views = [("axial", c.hu_r[sx, sy, k][::-1, ::-1].T, mask[sx, sy, k][::-1, ::-1].T, c.zr[1] / c.zr[0], "R", "L"),
             ("coronal", c.hu_r[sx, j, sz][::-1, ::-1].T, mask[sx, j, sz][::-1, ::-1].T, c.zr[2] / c.zr[0], "R", "L"),
             ("sagittal", c.hu_r[i, sy, sz][::-1, ::-1].T, mask[i, sy, sz][::-1, ::-1].T, c.zr[2] / c.zr[1], "A", "P")]
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for a, (name, im, mk, asp, l, r) in zip(axs, views):
        a.imshow(im, cmap="gray", vmin=window[0], vmax=window[1], aspect=asp, interpolation="bilinear")
        if outline and mk.any():
            a.contour(mk.astype(float), levels=[0.5], colors=[colour], linewidths=1.2)
        a.set_title(name, color="#cfd6df", fontsize=10)
        a.text(0.02, 0.5, l, color="#9aa4b2", transform=a.transAxes, fontsize=10, va="center")
        a.text(0.98, 0.5, r, color="#9aa4b2", transform=a.transAxes, fontsize=10, va="center", ha="right")
        a.set_xticks([]); a.set_yticks([])
    fig.patch.set_facecolor("black")
    return png_b64(fig)


PAGE_CSS = """
*{box-sizing:border-box}body{margin:0;background:#f4f5f7;font:18px/1.5 'Segoe UI',Arial,sans-serif;color:#1f2328}
.page{max-width:1180px;margin:0 auto;padding:20px}h1{font-size:26px;margin:0 0 4px}h2{font-size:21px;margin:22px 0 4px}
.muted{color:#4f5a66;font-size:15.5px}.card{background:#fff;border:1px solid #d5d9de;border-radius:12px;padding:14px 16px;margin-top:12px}
img{max-width:100%;border-radius:8px;display:block;background:#000}.imgs{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.q{margin-top:10px}.q b{display:block;font-size:16px;margin-bottom:3px}
button.c{font:inherit;font-size:16px;background:#fff;border:1px solid #c9d1d9;border-radius:8px;padding:5px 12px;margin:2px 4px 2px 0;cursor:pointer}
button.c.sel{background:#0969da;border-color:#0969da;color:#fff}
input,textarea{font:inherit;font-size:16px;padding:4px 8px;border:1px solid #c9d1d9;border-radius:8px}input.mm{width:90px}textarea{width:100%;min-height:44px}
.top{position:sticky;top:0;background:#eef4ff;border:1px solid #b9cdf5;border-radius:12px;padding:10px 14px;z-index:5}
.save{background:#1a7f37;color:#fff;border:0;border-radius:8px;font:inherit;font-weight:600;padding:7px 14px;cursor:pointer}
.tag{display:inline-block;background:#1f2328;color:#fff;border-radius:10px;padding:0 10px;font-size:15px;margin-right:6px}
"""

PAGE_JS = r"""
(function(){
 const D = __DATA__, KEY = 'lungct_candreview_' + D.kind + '_' + D.build;
 let st = {reviewer: '', a: {}};
 try { const s = JSON.parse(localStorage.getItem(KEY) || 'null'); if (s && s.a) st = s; } catch (e) {}
 const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} count(); };
 const rev = document.getElementById('rev'); rev.value = st.reviewer || ''; rev.oninput = () => { st.reviewer = rev.value; save(); };
 document.querySelectorAll('[data-item]').forEach(card => {
   const id = card.dataset.item, a = st.a[id] || (st.a[id] = {});
   card.querySelectorAll('[data-f]').forEach(grp => {
     const f = grp.dataset.f, multi = grp.dataset.multi === '1';
     grp.querySelectorAll('button.c').forEach(b => {
       const paint = () => b.classList.toggle('sel', multi ? (a[f] || []).indexOf(b.dataset.v) >= 0 : a[f] === b.dataset.v);
       b.onclick = () => {
         if (multi) { let w = (a[f] || []).slice(); const k = w.indexOf(b.dataset.v); if (b.dataset.x === '1') w = k >= 0 ? [] : [b.dataset.v]; else { w = w.filter(x => grp.querySelector('button.c[data-v="' + x + '"]').dataset.x !== '1'); const k2 = w.indexOf(b.dataset.v); if (k2 >= 0) w.splice(k2, 1); else w.push(b.dataset.v); } a[f] = w; }
         else a[f] = a[f] === b.dataset.v ? '' : b.dataset.v;
         grp.querySelectorAll('button.c').forEach(x => x.classList.toggle('sel', multi ? (a[f] || []).indexOf(x.dataset.v) >= 0 : a[f] === x.dataset.v)); save(); };
       paint();
     });
   });
   card.querySelectorAll('[data-t]').forEach(inp => { inp.value = a[inp.dataset.t] || ''; inp.oninput = () => { a[inp.dataset.t] = inp.value; save(); }; });
 });
 function count() { const n = D.items.filter(i => (st.a[i.id] || {})[D.first]).length; document.getElementById('cnt').textContent = n + ' of ' + D.items.length + ' answered (autosaved in this browser)'; }
 count();
 const q = v => '"' + String(v == null ? '' : (Array.isArray(v) ? v.join(' ') : v)).replace(/"/g, '""') + '"';
 document.getElementById('save').onclick = () => {
   if (!st.reviewer.trim()) { rev.focus(); alert('Please type your name first.'); return; }
   const now = new Date().toISOString(), cols = ['item_id','case_id','reviewer','saved_utc','review_kind','package_build'].concat(D.fields);
   const lines = [cols.join(',')];
   D.items.forEach(i => { const a = st.a[i.id] || {}; lines.push([i.id, i.case_id, st.reviewer, now, D.kind, D.build].concat(D.fields.map(f => a[f])).map(q).join(',')); });
   const el = document.createElement('a'); el.href = URL.createObjectURL(new Blob(['﻿' + lines.join('\r\n')], {type: 'text/csv'}));
   el.download = D.kind + '_review_' + st.reviewer.trim().replace(/[^A-Za-z0-9]+/g, '_') + '_' + now.slice(0, 10) + '.csv'; document.body.appendChild(el); el.click(); el.remove();
 };
})();
"""


def choice(f: str, opts, multi=False, exclusive=()) -> str:
    return (f"<span data-f='{f}' data-multi='{1 if multi else 0}'>"
            + "".join(f"<button class='c' type='button' data-v='{v}' data-x='{1 if v in exclusive else 0}'>{lab}</button>" for v, lab in opts) + "</span>")


def review_page(title: str, intro: str, kind: str, build: str, cards: str, items: list, fields: list, first: str) -> str:
    data = {"kind": kind, "build": build, "items": items, "fields": fields, "first": first}
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title><style>{PAGE_CSS}</style></head><body><div class='page'>"
            f"<h1>{title}</h1><div class='muted'>{intro}</div>"
            "<div class='top card'>Reviewer: <input id='rev' placeholder='your name'> <button class='save' id='save'>&#128190; Save answers (CSV)</button> <span id='cnt' class='muted'></span></div>"
            f"{cards}<div class='muted' style='margin-top:14px'>Contains CT-RATE images (CC BY-NC-SA, research use): keep within the research group. Research use only, not a diagnosis.</div>"
            f"</div><script>{PAGE_JS.replace('__DATA__', json.dumps(data))}</script></body></html>")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--pages-file", nargs="+", required=True)
    ap.add_argument("--grades", required=True, type=Path)
    ap.add_argument("--config", default="configs/checklist_v3_4.json")
    ap.add_argument("--holoct-run", default="runs/P3_holoct_meeting4_20261120_182147")
    ap.add_argument("--verdicts", default="runs/A3_audit_verdicts_usability_20261120_192144/verdicts.json")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    ck = load_checklist_config(ROOT / args.config)
    runs = {}
    for pf in args.pages_file:
        for line in (ROOT / pf).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = ROOT / Path(line.strip().replace("\\", "/")).parent
                runs[json.loads((r / "evidence.json").read_text(encoding="utf-8"))["case_id"]] = r
    ver = json.loads((ROOT / args.verdicts).read_text(encoding="utf-8"))
    hrun = ROOT / args.holoct_run
    reader = {}
    from dev_audit_reconcile import authenticated_grade_files
    for f in authenticated_grade_files(ROOT / args.grades):
        for r in csv.DictReader(open(f, newline="", encoding="utf-8-sig")):
            reader[(r["case_id"], r["condition_key"])] = r

    nod_rows, nod_cards, nod_items, case_summ = [], "", [], []
    peri_rows, peri_cards, peri_items = [], "", []
    for cid in args.cases:
        print(f"[{cid}]", flush=True)
        c = Case(cid, runs[cid], ck)
        # ------------------------------------------------ nodules
        g = reader.get((cid, "lung_nodule"), {})
        lobes_form = g.get("location_where", "").split() if g.get("location_status") == "stated" else []
        lobes_comment = re.findall(r"\b(RUL|RML|RLL|LUL|LLL)\b", g.get("comment", ""))
        rl = lobes_form or lobes_comment
        rl_src = "form field" if lobes_form else "free-text comment" if lobes_comment else "none recorded"
        nf = c.seg / "lung_nodules" / "lung_nodules.nii.gz"
        hmasks = []
        for v in ver["verdicts"]:
            if v["case_id"] == cid and "nodule" in v["parsed"]["types"]:
                f = hrun / cid / "findings_grounded" / f"mask_{int(v['finding_id'].split('|')[-1]):03d}.nii.gz"
                if f.exists():
                    hmasks.append({"statement": v["statement"], "mask": _to_ras(np.asanyarray(nib.load(str(f)).dataobj) > 0, c.aff)})
        hol = np.zeros(c.lung.shape, bool)
        for h in hmasks:
            hol |= h["mask"]
        hlab, hn = ndimage.label(hol, structure=np.ones((3, 3, 3)))
        hcent = [np.array(x) for x in ndimage.center_of_mass(hol, hlab, range(1, hn + 1))] if hn else []
        cons_band = c.stages("consolidation")
        cands = []
        if nf.exists():
            nm = _to_ras(np.asanyarray(nib.load(str(nf)).dataobj) > 0, c.aff)
            lab, n = ndimage.label(nm, structure=np.ones((3, 3, 3)))
            surf_mm = ndimage.distance_transform_edt(c.lung, sampling=c.zr) if n else None
            for kk, sl in enumerate(ndimage.find_objects(lab), 1):
                m = lab[sl] == kk
                nv = int(m.sum())
                vol = nv * c.vml * 1000.0
                full = np.zeros(c.lung.shape, bool)
                full[sl] = m
                cen = np.array(ndimage.center_of_mass(m)) + [s.start for s in sl]
                lv = np.bincount(c.lobe_idx[full], minlength=6)
                ext = [(s.stop - s.start) * z for s, z in zip(sl, c.zr)]
                d_eq = (6 * vol / math.pi) ** (1 / 3)
                hu = c.hu_r[full]
                ov_h = int((full & hol).sum())
                dh = min((float(np.linalg.norm((cen - hc) * c.zr)) for hc in hcent), default=None)
                cands.append({"mask": full, "centre": cen, "voxels": nv, "volume_mm3": round(vol, 1), "equivalent_sphere_diameter_mm": round(d_eq, 1),
                              "extent_mm": [round(x, 1) for x in ext], "longest_extent_mm": round(max(ext), 1),
                              "tool_lobe": c.short[int(lv[1:].argmax())] if lv[1:].sum() else "outside lobes",
                              "median_hu": round(float(np.median(hu))), "max_hu": round(float(hu.max())), "pct_voxels_over_200_hu": round(100 * float((hu > 200).mean()), 1),
                              "review_hypotheses": "",
                              "fill_of_bounding_sphere": round(vol / (math.pi / 6 * max(ext) ** 3), 2), "elongation_longest_over_shortest_extent": round(max(ext) / max(min(ext), 1e-6), 2),
                              "pct_overlap_dense_lung_in_outline": round(100 * float((full & cons_band[0]).sum()) / nv, 1),
                              "pct_overlap_measured_consolidation_region": round(100 * float((full & cons_band[5]).sum()) / nv, 1),
                              "pct_overlap_dense_lung_next_to_fluid": round(100 * float((full & cons_band[3] & ~cons_band[4]).sum()) / nv, 1),
                              "distance_to_lung_surface_mm": round(float(surf_mm[tuple(int(round(x)) for x in cen)]), 1),
                              "holoct_overlap_voxels": ov_h, "holoct_nearest_outline_mm": None if dh is None else round(dh, 1)})
        cands.sort(key=lambda r: -r["volume_mm3"])
        cl = {r["tool_lobe"] for r in cands}
        for n_, r in enumerate(cands, 1):
            r["id"] = f"{cid}|nodule_candidate|{n_:02d}"
            hyp = []      # hypotheses for the reviewer to confirm or reject; none of these is an adjudicated fact and none removes a candidate
            if r["pct_overlap_dense_lung_in_outline"] >= 50 and r["equivalent_sphere_diameter_mm"] >= 10:
                hyp.append("may be part of a larger dense-lung region rather than a nodule")
            if r["pct_overlap_dense_lung_next_to_fluid"] >= 50:
                hyp.append("lies mostly in lung next to outlined pleural fluid")
            if r["max_hu"] >= 300:
                hyp.append("contains very dense voxels (possible calcification; not adjudicated)")
            if r["voxels"] <= 10:
                hyp.append("only a few voxels: may be too small to judge from the outline")
            r["review_hypotheses"] = "; ".join(hyp)
            if not rl:
                r["reader_lobe_match"] = "no reader location recorded" if g.get("doctor_answer") == "present" else "reader graded nodule absent" if g.get("doctor_answer") == "absent" else "no reader grade"
            else:
                r["reader_lobe_match"] = "in a reader-named lobe" if r["tool_lobe"] in rl else "not in a reader-named lobe (reader's list may be incomplete)"
            r["holoct_match"] = ("no HoloCT nodule outline" if not hn else "overlaps a HoloCT nodule outline" if r["holoct_overlap_voxels"] else
                                 "near a HoloCT nodule outline (<= 10 mm)" if r["holoct_nearest_outline_mm"] is not None and r["holoct_nearest_outline_mm"] <= 10 else "no HoloCT outline nearby")
            pn = page_numbers(c, r["centre"])
            row = {"candidate_id": r["id"], "case_id": cid, "reader_nodule_answer": g.get("doctor_answer", ""), "reader_lobes": " ".join(rl), "reader_lobes_source": rl_src,
                   **{k: (" x ".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items() if k not in ("mask", "centre", "id")},
                   "viewer_axial_image": pn["axial"], "viewer_coronal_image": pn["coronal"], "viewer_sagittal_image": pn["sagittal"]}
            nod_rows.append(row)
            half = max(30.0, r["longest_extent_mm"] * 0.9)
            a = three_plane(c, r["centre"], r["mask"], half, (-1350, 150), False)
            b = three_plane(c, r["centre"], r["mask"], half, (-1350, 150), True)
            nod_items.append({"id": r["id"], "case_id": cid})
            nod_cards += (f"<div class='card' data-item='{r['id']}'><span class='tag'>{cid.replace('_a_2', '')}</span><b>Candidate {n_}</b> "
                          f"<span class='muted'>&middot; in the viewer: axial image {pn['axial']}, coronal {pn['coronal']}, sagittal {pn['sagittal']} &middot; "
                          f"outline volume {r['volume_mm3']:.0f} mm&sup3; (equivalent-sphere diameter {r['equivalent_sphere_diameter_mm']} mm: computed from the outline, not a caliper measurement)</span>"
                          f"<div class='imgs'><div><div class='muted'>plain (lung window, centred on the candidate)</div><img src='data:image/png;base64,{a}'></div>"
                          f"<div><div class='muted'>with the tool's outline in red</div><img src='data:image/png;base64,{b}'></div></div>"
                          f"<div class='q'><b>What is it?</b>{choice('finding', [('nodule', 'Nodule'), ('non_nodule', 'Not a nodule'), ('cannot_adjudicate', 'Cannot adjudicate')])}"
                          f" <input data-t='non_nodule_what' placeholder='if not a nodule: what is it?' style='width:330px'></div>"
                          f"<div class='q'><b>Is the red outline usable for measuring it?</b>{choice('mask', [('usable', 'Usable'), ('unusable', 'Unusable'), ('uncertain', 'Uncertain')])}</div>"
                          f"<div class='q'><b>Which lobe?</b>{choice('lobe', [(x, x) for x in LOBE_ORDER] + [('indeterminate', 'Indeterminate'), ('not_assessed', 'Not assessed')], multi=True, exclusive=('indeterminate', 'not_assessed'))}</div>"
                          f"<div class='q'><b>Your caliper measurement, if you take one (optional)</b>long axis <input class='mm' data-t='caliper_long_mm' placeholder='mm'> &nbsp; short axis <input class='mm' data-t='caliper_short_mm' placeholder='mm'></div>"
                          f"<div class='q'><textarea data-t='comment' placeholder='comment (optional)'></textarea></div></div>")
        # a per-case card for nodules the reviewer sees that the tool did not outline
        mid = f"{cid}|nodules_not_listed"
        nod_items.append({"id": mid, "case_id": cid})
        nod_cards = (f"<h2>{cid.replace('_a_2', '')}: {len(cands)} candidate(s) outlined by the tool</h2>" + nod_cards.split(f"<div class='card' data-item='{cid}|", 1)[0]
                     if False else nod_cards)
        nod_cards += (f"<div class='card' data-item='{mid}'><span class='tag'>{cid.replace('_a_2', '')}</span><b>Nodules you see in this scan that are NOT among the candidates above</b>"
                      f"<div class='q'>{choice('finding', [('none', 'None'), ('some', 'Yes, at least one'), ('not_assessed', 'Not assessed')])}</div>"
                      f"<div class='q'><b>Where?</b>{choice('lobe', [(x, x) for x in LOBE_ORDER] + [('indeterminate', 'Indeterminate'), ('not_assessed', 'Not assessed')], multi=True, exclusive=('indeterminate', 'not_assessed'))}</div>"
                      f"<div class='q'><textarea data-t='comment' placeholder='image numbers / description (optional)'></textarea></div></div>")
        case_summ.append({"case_id": cid, "reader_answer": g.get("doctor_answer", ""), "reader_lobes": rl, "reader_lobes_source": rl_src, "candidates": len(cands),
                          "candidate_lobes": sorted(cl), "reader_lobes_without_candidate": [x for x in rl if x not in cl],
                          "holoct_nodule_statements": [h["statement"] for h in hmasks], "holoct_nodule_pieces": int(hn),
                          "holoct_pieces_without_candidate": int(sum(1 for i in range(1, hn + 1) if not any((r["mask"] & (hlab == i)).any() for r in cands))) if hn else 0,
                          "upper_lobe_candidates": [{"id": r["id"], "lobe": r["tool_lobe"], "equivalent_sphere_diameter_mm": r["equivalent_sphere_diameter_mm"], "median_hu": r["median_hu"],
                                                     "holoct": r["holoct_match"]} for r in cands if r["tool_lobe"] in ("RUL", "LUL")],
                          "largest_candidate": ({"id": cands[0]["id"], "lobe": cands[0]["tool_lobe"], "equivalent_sphere_diameter_mm": cands[0]["equivalent_sphere_diameter_mm"],
                                                 "pct_overlap_dense_lung_in_outline": cands[0]["pct_overlap_dense_lung_in_outline"],
                                                 "pct_overlap_dense_lung_next_to_fluid": cands[0]["pct_overlap_dense_lung_next_to_fluid"]} if cands else None)})
        # ------------------------------------------------ pericardial fluid
        pc = _load(c.seg, "pleural_pericard_effusion/pericardial_effusion", c.ct)
        heart = _load(c.seg, "total/heart", c.ct)
        gp = reader.get((cid, "pericardial_effusion"), {})
        rec = {"case_id": cid, "reader_answer": gp.get("doctor_answer", ""), "reader_severity": gp.get("severity", ""), "reader_confidence": gp.get("confidence", ""),
               "reader_comment": gp.get("comment", ""), "outline_available": pc is not None}
        if pc is not None and pc.any():
            pr = _to_ras(pc, c.aff)
            hr = _to_ras(heart, c.aff) if heart is not None else None
            hu = c.hu_r[pr]
            lab, n = ndimage.label(pr)
            sizes = np.bincount(lab.ravel())[1:] * c.vml
            dist = ndimage.distance_transform_edt(pr, sampling=c.zr)
            thick_at = np.unravel_index(int(dist.argmax()), dist.shape)
            rec.update(volume_ml=round(float(pr.sum()) * c.vml, 1), largest_inscribed_sphere_diameter_mm=round(greatest_thickness_mm(pc, c.aff), 1),
                       largest_inscribed_sphere_note="2 x the largest distance from a voxel inside the predicted outline to the outline's edge (3-D, in mm): the diameter of the "
                                                     "biggest sphere that fits inside the outline. Stored as evidence row E79 'greatest thickness'. It is a property of the predicted "
                                                     "mask, not a caliper measurement of fluid on a chosen image, and has not been compared with one.",
                       pieces=int(n),
                       largest_piece_ml=round(float(sizes.max()), 1), median_hu=round(float(np.median(hu))), hu_p10=round(float(np.percentile(hu, 10))),
                       hu_p90=round(float(np.percentile(hu, 90))), pct_voxels_below_minus30_hu=round(100 * float((hu < -30).mean()), 1),
                       pct_voxels_minus30_to_30_hu=round(100 * float(((hu >= -30) & (hu <= 30)).mean()), 1),
                       pct_voxels_above_30_hu=round(100 * float((hu > 30).mean()), 1))
            if hr is not None and hr.any():
                hc = np.array(ndimage.center_of_mass(hr))
                idx = np.argwhere(pr)
                d = (idx - hc) * c.zr
                rec.update(pct_in_front_of_heart_centre=round(100 * float((d[:, 1] > 0).mean())), pct_below_heart_centre=round(100 * float((d[:, 2] < 0).mean())),
                           pct_left_of_heart_centre=round(100 * float((d[:, 0] < 0).mean())),
                           pct_touching_heart_outline=round(100 * float((pr & ndimage.binary_dilation(hr, iterations=2)).sum()) / int(pr.sum())))
            ha = hrun / cid / "answers.json"
            if ha.exists():
                rec["holoct_yes_no"] = next((a["answer"] for a in json.loads(ha.read_text(encoding="utf-8"))["answers"] if a["ct_rate_label"] == "Pericardial effusion"), "not available")
            pn = page_numbers(c, thick_at)
            rec["viewer_axial_image_at_thickest_spot"] = pn["axial"]
            a = three_plane(c, thick_at, pr, 75.0, (-160, 240), False)
            b = three_plane(c, thick_at, pr, 75.0, (-160, 240), True, "#c084fc")
            iid = f"{cid}|pericardial_outline"
            peri_items.append({"id": iid, "case_id": cid})
            peri_cards += (f"<div class='card' data-item='{iid}'><span class='tag'>{cid.replace('_a_2', '')}</span><b>Outlined pericardial fluid</b> "
                           f"<span class='muted'>&middot; centred on the thickest spot of the outline &middot; in the viewer: axial image {pn['axial']}, coronal {pn['coronal']}, sagittal {pn['sagittal']}</span>"
                           f"<div class='imgs'><div><div class='muted'>plain (soft-tissue window)</div><img src='data:image/png;base64,{a}'></div>"
                           f"<div><div class='muted'>with the tool's outline in violet</div><img src='data:image/png;base64,{b}'></div></div>"
                           f"<div class='q'><b>1. Is what the violet outline covers pericardial fluid?</b>{choice('mask', [('fluid', 'Yes, fluid'), ('partly', 'Partly'), ('not_fluid', 'No'), ('cannot_adjudicate', 'Cannot adjudicate')])}"
                           f" <input data-t='not_fluid_what' placeholder='if no / partly: what is it?' style='width:330px'></div>"
                           f"<div class='q'><b>2. Does it miss fluid that you can see?</b>{choice('missed', [('no', 'No'), ('yes', 'Yes'), ('cannot_tell', 'Cannot tell')])}</div>"
                           f"<div class='q'><b>3. Is a pericardial effusion present in this scan?</b>{choice('presence', [('present', 'Present'), ('absent', 'Absent'), ('cannot_tell', 'Cannot tell')])}</div>"
                           f"<div class='q'><b>4. If present, how would you describe its size?</b>{choice('size', [('trace', 'Trace'), ('mild', 'Mild'), ('moderate', 'Moderate'), ('large', 'Large')])}</div>"
                           f"<div class='q'><textarea data-t='comment' placeholder='comment (optional)'></textarea></div></div>")
        peri_rows.append(rec)
        del c
    with open(out / "candidates" / "nodule_candidates.csv", "w", newline="", encoding="utf-8") as f:
        fields = []
        for r in nod_rows:
            fields += [k for k in r if k not in fields]
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(nod_rows)
    with open(out / "candidates" / "pericardial_outlines.csv", "w", newline="", encoding="utf-8") as f:
        fields = []
        for r in peri_rows:
            fields += [k for k in r if k not in fields]
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(peri_rows)
    (out / "candidates" / "candidates_summary.json").write_text(json.dumps({"nodules_per_case": case_summ, "pericardial": peri_rows}, indent=2), encoding="utf-8")
    build = "candidate_review_v2"
    (out / "candidates" / "nodule_candidate_review.html").write_text(review_page(
        "Nodule candidates: review of the tool's outlines",
        "Each card is ONE piece outlined by an automatic tool. It is a candidate, not a confirmed nodule: please say what it is. Images are centred on the candidate; "
        "patient's right is on the image left. The size given is computed from the outline volume and is not a caliper measurement. Use the full viewer page of the same scan to scroll if needed.",
        "nodule_candidates", build, nod_cards, nod_items, ["finding", "non_nodule_what", "mask", "lobe", "caliper_long_mm", "caliper_short_mm", "comment"], "finding"), encoding="utf-8")
    (out / "candidates" / "pericardial_outline_review.html").write_text(review_page(
        "Pericardial fluid: review of the tool's outline",
        "One card per scan. The three questions are separate on purpose: whether the outline is right, whether an effusion is present, and how large you would call it. "
        "Images are centred on the thickest spot of the outline; patient's right is on the image left.",
        "pericardial_outline", build, peri_cards, peri_items, ["mask", "not_fluid_what", "missed", "presence", "size", "comment"], "mask"), encoding="utf-8")
    print(json.dumps(case_summ, indent=1)[:3500])
    for r in peri_rows:
        print({k: v for k, v in r.items()})
    print(f"-> {out / 'candidates'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
