"""Consolidated clinician review package for the development audit (core env): ONE standalone page, staged.

DEVELOPMENT ADJUDICATION. The scans were inspected before and graded on open pages, so this is not blinded
validation and produces no performance estimate. It asks a clinician to adjudicate specific automatic outlines.

Three sections:
  A  retained dense-lung regions      what the region represents; outline adequacy; lobe          (+ per scan: presence)
  B  nodule candidates                nodule identity; outline usability; lobe; optional calipers  (+ per scan: presence, missed nodules)
  C  pericardial-fluid outlines       outline correctness; missed fluid                            (+ presence, size word)
Condition presence, location and mask adequacy are always separate questions.

Staging: until the reviewer saves and locks the initial assessment, the page shows only images, outlines and
questions. Hidden until then: analyst hypotheses, which model drew an outline, tool measurements and tags, checklist
answers, HoloCT statements and answers, checker information, and the reviewer's own earlier grades. After the lock
the initial answers are frozen and shown read-only, the hidden information appears per card, and a SEPARATE
post-reveal assessment can be entered. The export holds both. The hidden information is inside the HTML file: this is
a reading procedure, not a security barrier.

Dense regions: only a subset gets a card (rule recorded, with actual volume coverage and the number of regions
selected and omitted per scan and tag). The page says so; the cards do not stand for every region.

Nothing is derived here: no threshold, no deletion of candidates, no diagnosis assigned to a region.

Usage: python scripts/dev_audit_review_package.py --cases valid_301_a_2 valid_541_a_2 valid_832_a_2 valid_988_a_2
           --pages-file ... --grades runs/grades_received --out runs/D1_dev_audit_20261122
"""
from __future__ import annotations

import argparse
import csv
import html
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
from dev_audit_candidates import PAGE_CSS, page_numbers, three_plane  # noqa: E402
from dev_audit_density import Case  # noqa: E402
from dev_audit_reconcile import authenticated_grade_files  # noqa: E402
from lung_ct.agent.checklist import load_checklist_config  # noqa: E402
from lung_ct.agent.conditions import _load, _to_ras, greatest_thickness_mm  # noqa: E402

E = html.escape
BUILD = "dev_adjudication_package_v1"
LOBES = ["RUL", "RML", "RLL", "LUL", "LLL"]
TAGS = ["baseline_measured", "fluid_adjacent", "posterior", "fluid_adjacent_and_posterior"]
TAG_TEXT = {"baseline_measured": "not next to outlined pleural fluid and not in the posterior band; the baseline counts it in its consolidation measure",
            "fluid_adjacent": "within 15 mm of outlined pleural fluid; the baseline counts it as 'compressed lung' (towards atelectasis), not as consolidation",
            "posterior": "in the posterior band (back fifth of the lung); the baseline sets it aside and uses it for no answer",
            "fluid_adjacent_and_posterior": "within 15 mm of outlined pleural fluid AND in the posterior band; the baseline counts it as 'compressed lung'"}
MAX_PER_TAG, COVER = 3, 0.8
SELECTION_RULE = (f"Within each scan and tag, connected regions are sorted by volume and given a card, largest first, until at least {int(COVER * 100)}% of that "
                  f"tag's volume is shown or {MAX_PER_TAG} regions have a card. This is a practical limit for the review page, not a diagnostic rule; regions "
                  "without a card were not reviewed.")
LOBE_Q = {"f": "lobe", "label": "Location: which lobe?", "kind": "multi", "group": "location",
          "opts": [[x, x] for x in LOBES] + [["indeterminate", "Indeterminate"], ["not_assessed", "Not assessed"]], "exclusive": ["indeterminate", "not_assessed"]}
COMMENT_Q = {"f": "comment", "label": "Comment (optional)", "kind": "textarea", "group": "other"}

JS = r"""
(function(){
 const D = __DATA__, KEY = 'lungct_devadj_' + D.build;
 let st = {reviewer: '', locked_utc: null, a: {}, post: {}};
 try { const s = JSON.parse(localStorage.getItem(KEY) || 'null'); if (s && s.a) st = Object.assign(st, s); } catch (e) {}
 const store = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} head(); };
 const $ = s => document.querySelector(s);
 const esc = t => String(t == null ? '' : t).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
 function form(item, bag, disabled, prefix) {      // one set of questions; bag = the answers it reads and writes
   const box = document.createElement('div');
   item.q.forEach(q => {
     const d = document.createElement('div'); d.className = 'q'; d.innerHTML = '<b>' + esc(q.label) + '</b>';
     if (q.kind === 'single' || q.kind === 'multi') {
       (q.opts || []).forEach(o => {
         const b = document.createElement('button'); b.type = 'button'; b.className = 'c'; b.textContent = o[1]; b.disabled = disabled;
         const on = () => q.kind === 'multi' ? (bag[q.f] || []).indexOf(o[0]) >= 0 : bag[q.f] === o[0];
         const paint = () => d.querySelectorAll('button.c').forEach((x, i) => x.classList.toggle('sel', q.kind === 'multi' ? (bag[q.f] || []).indexOf(q.opts[i][0]) >= 0 : bag[q.f] === q.opts[i][0]));
         b.onclick = () => {
           if (q.kind === 'multi') { const ex = q.exclusive || []; let w = (bag[q.f] || []).slice();
             if (ex.indexOf(o[0]) >= 0) w = w.indexOf(o[0]) >= 0 ? [] : [o[0]];
             else { w = w.filter(x => ex.indexOf(x) < 0); const k = w.indexOf(o[0]); if (k >= 0) w.splice(k, 1); else w.push(o[0]); }
             bag[q.f] = w; }
           else bag[q.f] = bag[q.f] === o[0] ? '' : o[0];
           paint(); store(); };
         if (on()) b.classList.add('sel');
         d.appendChild(b);
       });
     } else {
       const inp = document.createElement(q.kind === 'textarea' ? 'textarea' : 'input'); if (q.kind === 'mm') inp.className = 'mm';
       inp.placeholder = q.placeholder || ''; inp.value = bag[q.f] || ''; inp.disabled = disabled; if (q.kind === 'text') inp.style.width = '360px';
       inp.oninput = () => { bag[q.f] = inp.value; store(); };
       d.appendChild(inp);
     }
     box.appendChild(d);
   });
   return box;
 }
 function render() {
   const locked = !!st.locked_utc;
   document.body.classList.toggle('locked', locked);
   D.items.forEach(item => {
     const card = document.querySelector('[data-item="' + item.id + '"] .forms'); card.innerHTML = '';
     const a = st.a[item.id] || (st.a[item.id] = {});
     const h1 = document.createElement('div'); h1.className = 'stagehead'; h1.textContent = locked ? 'Initial assessment (locked ' + st.locked_utc.slice(0, 16).replace('T', ' ') + ' UTC)' : 'Your assessment';
     card.appendChild(h1); card.appendChild(form(item, a, locked, 'initial'));
     if (locked) {
       const r = document.createElement('div'); r.className = 'reveal'; r.innerHTML = '<b>Shown only now: what the tools and earlier records said</b>' + item.reveal; card.appendChild(r);
       const p = st.post[item.id] || (st.post[item.id] = {});
       const h2 = document.createElement('div'); h2.className = 'stagehead post'; h2.textContent = 'Assessment after seeing this information (separate; leave empty if nothing changes)';
       card.appendChild(h2); card.appendChild(form(item, p, false, 'post'));
     }
   });
   head();
 }
 function answered(bag, item) { return item.q.some(q => { const v = bag[q.f]; return Array.isArray(v) ? v.length : !!v; }); }
 function head() {
   const locked = !!st.locked_utc, n = D.items.filter(i => answered(st.a[i.id] || {}, i)).length;
   $('#cnt').textContent = n + ' of ' + D.items.length + ' cards have an initial answer' + (locked ? ' (locked)' : ' (autosaved in this browser)');
   $('#lock').style.display = locked ? 'none' : ''; $('#save').style.display = locked ? '' : 'none';
   $('#stage').innerHTML = locked
     ? '<b>Step 2 of 2.</b> Your initial answers are locked and kept. Each card now also shows what the tools and earlier records said, and a separate set of questions for an assessment after seeing that. Then press <b>Save answers (CSV)</b>.'
     : '<b>Step 1 of 2: your own reading.</b> Only images, outlines and questions are shown. What the tools measured or concluded, which tool drew an outline, and your own earlier grades stay hidden until you lock this step.';
 }
 const rev = $('#rev'); rev.value = st.reviewer || ''; rev.oninput = () => { st.reviewer = rev.value; store(); };
 $('#lock').onclick = () => {
   if (!st.reviewer.trim()) { rev.focus(); alert('Please type your name first.'); return; }
   const open = D.items.filter(i => !answered(st.a[i.id] || {}, i)).length;
   if (!confirm((open ? open + ' card(s) have no answer yet; they will be recorded as not answered in the initial assessment. ' : '') + 'Lock the initial assessment? It cannot be changed afterwards.')) return;
   st.locked_utc = new Date().toISOString(); st.initial_reviewer = st.reviewer; store(); render(); window.scrollTo(0, 0);
 };
 const q = v => '"' + String(v == null ? '' : (Array.isArray(v) ? v.join(' ') : v)).replace(/"/g, '""') + '"';
 $('#save').onclick = () => {
   const now = new Date().toISOString();
   const cols = ['item_id','case_id','section','item_kind','reviewer','saved_utc','package_build','review_type','initial_locked_utc','initial_answered','post_reveal_answered']
     .concat(D.fields.map(f => 'initial_' + f)).concat(D.fields.map(f => 'post_reveal_' + f));
   const lines = [cols.join(',')];
   D.items.forEach(i => { const a = st.a[i.id] || {}, p = st.post[i.id] || {};
     lines.push([i.id, i.case_id, i.section, i.kind, st.reviewer, now, D.build, D.review_type, st.locked_utc, answered(a, i) ? 'yes' : 'no', answered(p, i) ? 'yes' : 'no']
       .concat(D.fields.map(f => a[f])).concat(D.fields.map(f => p[f])).map(q).join(',')); });
   const el = document.createElement('a'); el.href = URL.createObjectURL(new Blob(['﻿' + lines.join('\r\n')], {type: 'text/csv'}));
   el.download = 'development_adjudication_' + st.reviewer.trim().replace(/[^A-Za-z0-9]+/g, '_') + '_' + now.slice(0, 10) + '.csv'; document.body.appendChild(el); el.click(); el.remove();
 };
 render();
 window.__pkg = {state: () => st, key: KEY};
})();
"""

CSS = PAGE_CSS + """
.stagebar{background:#fff4cc;border:1px solid #f0d36b;border-radius:10px;padding:8px 12px;margin-top:8px;color:#3d3000}
.stagehead{font-weight:700;margin-top:12px;color:#0b3d91}.stagehead.post{color:#7a3b00;border-top:1px dashed #c9d1d9;padding-top:10px}
.reveal{background:#f6f8fa;border:1px solid #d8dee4;border-radius:10px;padding:10px 12px;margin-top:12px;font-size:16px}
.reveal ul{margin:6px 0 0 20px;padding:0}.reveal li{margin:3px 0}
button.c:disabled{opacity:.75;cursor:default}button.c.sel:disabled{opacity:1}
.lockbtn{background:#9a6700;color:#fff;border:0;border-radius:8px;font:inherit;font-weight:600;padding:7px 14px;cursor:pointer}
.sec{background:#eef4ff;border:1px solid #b9cdf5;border-radius:12px;padding:10px 14px;margin-top:26px}
"""


def card(item: dict, imgs: list, sub: str, note: str = "") -> str:
    pics = "".join(f"<div><div class='muted'>{E(cap)}</div><img src='data:image/png;base64,{b}'></div>" for cap, b in imgs)
    return (f"<div class='card' data-item='{E(item['id'])}'><span class='tag'>{E(item['case_id'].replace('_a_2', ''))}</span><b>{E(item['title'])}</b> "
            f"<span class='muted'>{sub}</span>" + (f"<div class='imgs'>{pics}</div>" if pics else "") + (f"<div class='muted' style='margin-top:6px'>{note}</div>" if note else "")
            + "<div class='forms'></div></div>")


def ul(rows: list) -> str:
    return "<ul>" + "".join(f"<li>{r}</li>" for r in rows if r) + "</ul>"


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
    out = (args.out if args.out.is_absolute() else ROOT / args.out) / "review_package"
    out.mkdir(parents=True, exist_ok=True)
    ck = load_checklist_config(ROOT / args.config)
    runs = {}
    for pf in args.pages_file:
        for line in (ROOT / pf).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = ROOT / Path(line.strip().replace("\\", "/")).parent
                runs[json.loads((r / "evidence.json").read_text(encoding="utf-8"))["case_id"]] = r
    ver = json.loads((ROOT / args.verdicts).read_text(encoding="utf-8"))
    hrun = ROOT / args.holoct_run
    reader, shown_cfg = {}, {}
    for f in authenticated_grade_files(ROOT / args.grades):
        for r in csv.DictReader(open(f, newline="", encoding="utf-8-sig")):
            reader[(r["case_id"], r["condition_key"])] = r
            shown_cfg[r["case_id"]] = r["checklist_config"]

    def prior(cid, key):
        r = reader.get((cid, key))
        if not r:
            return "no earlier grade on record"
        loc = r.get("location_where") if r.get("location_status") == "stated" else ""
        return ("your earlier grade on the open page: <b>" + E(r["doctor_answer"] or "not answered") + "</b>" + (f", {E(r['severity'])}" if r["severity"] else "")
                + (f", location {E(loc)}" if loc else "") + (f"; comment: &ldquo;{E(r['comment'])}&rdquo;" if r["comment"] else ""))

    secA, secB, secC = "", "", ""
    items, coverage, key_rows = [], [], []
    for cid in args.cases:
        print(f"[{cid}]", flush=True)
        c = Case(cid, runs[cid], ck)
        chk = {i["key"]: i for i in json.loads((runs[cid] / f"checklist_{shown_cfg[cid]}.json").read_text(encoding="utf-8"))["items"]}

        def tool(key):
            it = chk[key]
            return f"checklist answer shown to you earlier ({E(shown_cfg[cid])}): <b>{E(it.get('display', it['status']))}</b>" + (f" &mdash; {E(it.get('location_text') or it.get('location') or '')}" if (it.get('location_text') or it.get('location')) else "")
        hv = [v for v in ver["verdicts"] if v["case_id"] == cid]

        def hmask(v):
            f = hrun / cid / "findings_grounded" / f"mask_{int(v['finding_id'].split('|')[-1]):03d}.nii.gz"
            return _to_ras(np.asanyarray(nib.load(str(f)).dataobj) > 0, c.aff) if f.exists() else None
        ha = hrun / cid / "answers.json"
        hans = {a["ct_rate_label"]: a["answer"] for a in json.loads(ha.read_text(encoding="utf-8"))["answers"]} if ha.exists() else {}
        short = cid.replace("_a_2", "")

        # ============ A. retained dense regions ============
        st = c.stages("consolidation")
        s4 = st[3]
        near = c.near_eff if c.near_eff is not None else np.zeros(s4.shape, bool)
        masks = {"baseline_measured": s4 & ~near & ~c.dep, "fluid_adjacent": s4 & near & ~c.dep, "posterior": s4 & c.dep & ~near, "fluid_adjacent_and_posterior": s4 & near & c.dep}
        assert np.array_equal(masks["baseline_measured"], st[5]) and round(float(st[5].sum()) * c.vml) == c.rows["E85"]["value"], "baseline not reproduced"
        hcons = [(v["statement"], hmask(v)) for v in hv if set(v["parsed"]["types"]) & {"consolidation", "atelectasis"}]
        n_card = 0
        cov_case = []
        for tag in TAGS:
            m = masks[tag]
            lab, n = ndimage.label(m)
            sizes = np.bincount(lab.ravel())[1:] * c.vml if n else np.array([])
            order = list(np.argsort(-sizes)) if n else []
            tot, shown, sel = float(sizes.sum()) if n else 0.0, 0.0, []
            for idx in order:
                if len(sel) >= MAX_PER_TAG or (tot > 0 and shown / tot >= COVER):
                    break
                sel.append(int(idx))
                shown += float(sizes[idx])
            omitted = [float(sizes[i]) for i in order if int(i) not in sel]
            cv = {"case_id": cid, "tag": tag, "regions_total": int(n), "regions_with_a_card": len(sel), "regions_omitted": int(n) - len(sel),
                  "ml_total": round(tot, 2), "ml_on_cards": round(shown, 2), "volume_coverage_pct": round(100 * shown / tot, 1) if tot > 0 else None,
                  "ml_omitted": round(tot - shown, 2), "largest_omitted_region_ml": round(max(omitted), 2) if omitted else 0.0,
                  "stopped_because": "no region" if not n else "coverage target reached" if tot > 0 and shown / tot >= COVER else "maximum number of cards reached" if len(sel) >= MAX_PER_TAG else "all regions shown"}
            coverage.append(cv)
            cov_case.append(cv)
            for idx in sel:
                piece = lab == (idx + 1)
                cen = np.array(ndimage.center_of_mass(piece))
                lv = np.bincount(c.lobe_idx[piece], minlength=6)
                sl = ndimage.find_objects(piece.astype(np.uint8))[0]
                ext = float(max((s.stop - s.start) * z for s, z in zip(sl, c.zr)))
                n_card += 1
                pid = f"{cid}|dense_region|{n_card:02d}"
                pn = page_numbers(c, cen)
                half = max(40.0, ext * 0.8)
                imgs = [("plain, lung window", three_plane(c, cen, piece, half, (-1350, 150), False)), ("with the outline in red, lung window", three_plane(c, cen, piece, half, (-1350, 150), True)),
                        ("plain, soft-tissue window", three_plane(c, cen, piece, half, (-160, 240), False))]
                tl = c.short[int(lv[1:].argmax())] if lv[1:].sum() else "outside lobe outlines"
                hov = [f"HoloCT statement &ldquo;{E(s_)}&rdquo;: its outline overlaps this region by {float((piece & hm).sum()) * c.vml:.1f} mL" for s_, hm in hcons if hm is not None and (piece & hm).any()]
                reveal = ul([f"outline made by density thresholding (-300 to 100 HU) inside lung outlines from TotalSegmentator 2.18, after removing vessels and airways",
                             f"analyst tag: <b>{tag.replace('_', ' ')}</b> &mdash; {E(TAG_TEXT[tag])}",
                             f"region volume {float(sizes[idx]):.1f} mL; median {round(float(np.median(c.hu_r[piece])))} HU; lobe by the tool's lobe outlines: {tl}",
                             "consolidation: " + tool("consolidation") + "; " + prior(cid, "consolidation"),
                             "atelectasis: " + tool("atelectasis") + "; " + prior(cid, "atelectasis")] + (hov or ["HoloCT: no consolidation or atelectasis outline overlaps this region"]))
                it = {"id": pid, "case_id": cid, "section": "A_dense_regions", "kind": "dense_region", "title": f"Dense-lung region {n_card}", "reveal": reveal,
                      "q": [{"f": "represents", "label": "1. Identity: what does the outlined region represent?", "kind": "single", "group": "identity",
                             "opts": [["consolidation", "Consolidation"], ["atelectasis", "Atelectasis"], ["both", "Both"], ["other", "Another structure or process"], ["cannot_assess", "Cannot be assessed"]]},
                            {"f": "other_what", "label": "If another structure or process: what?", "kind": "text", "group": "identity"},
                            {"f": "mask", "label": "2. Outline adequacy (separate from question 1): is the red outline adequate for that region?", "kind": "single", "group": "mask",
                             "opts": [["adequate", "Adequate"], ["too_large", "Too large"], ["too_small", "Too small"], ["wrong_place", "Wrong place"], ["uncertain", "Uncertain"]]},
                            dict(LOBE_Q, label="3. Location: which lobe?"), COMMENT_Q]}
                items.append(it)
                secA += card(it, imgs, f"&middot; in the viewer page of this scan: axial image {pn['axial']}, coronal {pn['coronal']}, sagittal {pn['sagittal']}")
                key_rows.append({"item_id": pid, "tag": tag, "ml": round(float(sizes[idx]), 2), "tool_lobe": tl, "median_hu": round(float(np.median(c.hu_r[piece])))})
        tot_r, sel_r = sum(x["regions_total"] for x in cov_case), sum(x["regions_with_a_card"] for x in cov_case)
        tot_ml, sel_ml = sum(x["ml_total"] for x in cov_case), sum(x["ml_on_cards"] for x in cov_case)
        sid = f"{cid}|dense_scan_level"
        rows_cov = "".join(f"<li>{x['tag'].replace('_', ' ')}: {x['regions_with_a_card']} of {x['regions_total']} regions on cards, {x['ml_on_cards']} of {x['ml_total']} mL"
                           + (f" ({x['volume_coverage_pct']}%)" if x["volume_coverage_pct"] is not None else "") + f"; {x['regions_omitted']} not shown</li>" for x in cov_case)
        it = {"id": sid, "case_id": cid, "section": "A_dense_regions", "kind": "scan_level", "title": "This scan as a whole: dense lung",
              "reveal": ul(["consolidation: " + tool("consolidation") + "; " + prior(cid, "consolidation"), "atelectasis: " + tool("atelectasis") + "; " + prior(cid, "atelectasis"),
                            "pleural effusion: " + tool("pleural_effusion") + "; " + prior(cid, "pleural_effusion"),
                            f"HoloCT yes/no: consolidation {E(hans.get('Consolidation', 'not available'))}, atelectasis {E(hans.get('Atelectasis', 'not available'))}",
                            "regions on cards per analyst tag:" + "<ul>" + rows_cov + "</ul>"]),
              "q": [{"f": "consolidation", "label": "Presence: consolidation in this scan", "kind": "single", "group": "presence", "opts": [["present", "Present"], ["absent", "Absent"], ["cannot_tell", "Cannot tell"]]},
                    {"f": "atelectasis", "label": "Presence: atelectasis in this scan", "kind": "single", "group": "presence", "opts": [["present", "Present"], ["absent", "Absent"], ["cannot_tell", "Cannot tell"]]},
                    {"f": "uncovered", "label": "Dense lung that you see and that none of the cards above covers", "kind": "single", "group": "other", "opts": [["none", "None"], ["some", "Yes"], ["not_assessed", "Not assessed"]]},
                    dict(LOBE_Q, f="uncovered_lobe", label="If yes: where?"), COMMENT_Q]}
        items.append(it)
        secA += card(it, [], "(separate from the outlines above)",
                     f"For this scan the cards above show <b>{sel_r} of {tot_r}</b> automatically outlined dense regions, holding <b>{sel_ml:.1f} of {tot_ml:.1f} mL</b> of the outlined dense lung. "
                     "The other regions are not shown, so the cards do not represent every region of this scan." if tot_r else "No dense region was outlined on this scan.")

        # ============ B. nodule candidates ============
        g = reader.get((cid, "lung_nodule"), {})
        nf = c.seg / "lung_nodules" / "lung_nodules.nii.gz"
        hnod = [(v["statement"], hmask(v)) for v in hv if "nodule" in v["parsed"]["types"]]
        hol = np.zeros(c.lung.shape, bool)
        for _, hm in hnod:
            if hm is not None:
                hol |= hm
        cons0, cadj = st[0], st[3] & ~st[4]
        cands = []
        if nf.exists():
            nm = _to_ras(np.asanyarray(nib.load(str(nf)).dataobj) > 0, c.aff)
            lab, n = ndimage.label(nm, structure=np.ones((3, 3, 3)))
            for kk, sl in enumerate(ndimage.find_objects(lab), 1):
                m = lab[sl] == kk
                full = np.zeros(c.lung.shape, bool)
                full[sl] = m
                nv = int(m.sum())
                vol = nv * c.vml * 1000.0
                hu = c.hu_r[full]
                lv = np.bincount(c.lobe_idx[full], minlength=6)
                ext = [(s.stop - s.start) * z for s, z in zip(sl, c.zr)]
                hyp = []
                d_eq = (6 * vol / math.pi) ** (1 / 3)
                dense = 100 * float((full & cons0).sum()) / nv
                fl = 100 * float((full & cadj).sum()) / nv
                if dense >= 50 and d_eq >= 10:
                    hyp.append("may be part of a larger dense-lung region rather than a nodule")
                if fl >= 50:
                    hyp.append("lies mostly in lung next to outlined pleural fluid")
                if float(hu.max()) >= 300:
                    hyp.append("contains very dense voxels (possible calcification; not adjudicated)")
                if nv <= 10:
                    hyp.append("only a few voxels: may be too small to judge from the outline")
                cands.append({"mask": full, "centre": np.array(ndimage.center_of_mass(m)) + [s.start for s in sl], "vol": vol, "d_eq": d_eq, "ext": max(ext),
                              "lobe": c.short[int(lv[1:].argmax())] if lv[1:].sum() else "outside lobe outlines", "hu": round(float(np.median(hu))), "humax": round(float(hu.max())),
                              "hyp": hyp, "holoct": bool((full & hol).any())})
        cands.sort(key=lambda r: -r["vol"])
        for n_, r in enumerate(cands, 1):
            pid = f"{cid}|nodule_candidate|{n_:02d}"
            pn = page_numbers(c, r["centre"])
            half = max(30.0, r["ext"] * 0.9)
            imgs = [("plain, lung window", three_plane(c, r["centre"], r["mask"], half, (-1350, 150), False)), ("with the outline in red", three_plane(c, r["centre"], r["mask"], half, (-1350, 150), True))]
            reveal = ul(["outline from TotalSegmentator 2.18, task lung_nodules (one connected piece of its output)",
                         f"outline volume {r['vol']:.0f} mm&sup3; = equivalent-sphere diameter {r['d_eq']:.1f} mm (computed from the outline volume; <b>not</b> a caliper diameter); longest extent {r['ext']:.1f} mm",
                         f"median {r['hu']} HU, maximum {r['humax']} HU; lobe by the tool's lobe outlines: {r['lobe']}",
                         "analyst review hypotheses (prompts only, not adjudicated): " + (E("; ".join(r["hyp"])) if r["hyp"] else "none"),
                         "HoloCT: " + ("one of its nodule outlines overlaps this candidate" if r["holoct"] else "no nodule outline of HoloCT overlaps this candidate"),
                         "lung nodule: " + tool("lung_nodule") + "; " + prior(cid, "lung_nodule")])
            it = {"id": pid, "case_id": cid, "section": "B_nodule_candidates", "kind": "nodule_candidate", "title": f"Candidate {n_}", "reveal": reveal,
                  "q": [{"f": "finding", "label": "1. Identity: what is the outlined structure?", "kind": "single", "group": "identity", "opts": [["nodule", "Nodule"], ["non_nodule", "Not a nodule"], ["cannot_adjudicate", "Cannot adjudicate"]]},
                        {"f": "non_nodule_what", "label": "If not a nodule: what is it?", "kind": "text", "group": "identity"},
                        {"f": "mask", "label": "2. Outline usability (separate from question 1): is the red outline usable for measuring it?", "kind": "single", "group": "mask",
                         "opts": [["usable", "Usable"], ["unusable", "Unusable"], ["uncertain", "Uncertain"]]},
                        dict(LOBE_Q, label="3. Location: which lobe?"),
                        {"f": "caliper_long_mm", "label": "Your caliper measurement, if you take one: long axis (mm)", "kind": "mm", "group": "other", "placeholder": "mm"},
                        {"f": "caliper_short_mm", "label": "short axis (mm)", "kind": "mm", "group": "other", "placeholder": "mm"}, COMMENT_Q]}
            items.append(it)
            secB += card(it, imgs, f"&middot; in the viewer page of this scan: axial image {pn['axial']}, coronal {pn['coronal']}, sagittal {pn['sagittal']}")
        sid = f"{cid}|nodule_scan_level"
        it = {"id": sid, "case_id": cid, "section": "B_nodule_candidates", "kind": "scan_level", "title": "This scan as a whole: nodules",
              "reveal": ul(["lung nodule: " + tool("lung_nodule") + "; " + prior(cid, "lung_nodule"), f"TotalSegmentator lung_nodules outlined {len(cands)} candidate(s) on this scan",
                            f"HoloCT yes/no for lung nodule: {E(hans.get('Lung nodule', 'not available'))}; statements: " + (E(" | ".join(s_ for s_, _ in hnod)) if hnod else "none")]),
              "q": [{"f": "nodule_presence", "label": "Presence: lung nodule in this scan", "kind": "single", "group": "presence", "opts": [["present", "Present"], ["absent", "Absent"], ["cannot_tell", "Cannot tell"]]},
                    {"f": "missed", "label": "Missed nodules: nodules you see that are NOT among the candidates above", "kind": "single", "group": "other", "opts": [["none", "None"], ["some", "Yes, at least one"], ["not_assessed", "Not assessed"]]},
                    dict(LOBE_Q, f="missed_lobe", label="If yes: in which lobe(s)?"),
                    {"f": "missed_where", "label": "Image numbers / description of the missed nodules (optional)", "kind": "textarea", "group": "other"}]}
        items.append(it)
        secB += card(it, [], "", f"{len(cands)} candidate(s) were outlined on this scan." if cands else "No candidate was outlined on this scan.")

        # ============ C. pericardial outline ============
        pc = _load(c.seg, "pleural_pericard_effusion/pericardial_effusion", c.ct)
        if pc is not None and pc.any():
            pr = _to_ras(pc, c.aff)
            dist = ndimage.distance_transform_edt(pr, sampling=c.zr)
            at = np.unravel_index(int(dist.argmax()), dist.shape)
            pn = page_numbers(c, at)
            hu = c.hu_r[pr]
            imgs = [("plain, soft-tissue window", three_plane(c, at, pr, 75.0, (-160, 240), False)), ("with the outline in violet", three_plane(c, at, pr, 75.0, (-160, 240), True, "#c084fc"))]
            pid = f"{cid}|pericardial_outline"
            reveal = ul(["outline from TotalSegmentator 2.18, task pleural_pericard_effusion",
                         f"outline volume {float(pr.sum()) * c.vml:.1f} mL in {int(ndimage.label(pr)[1])} piece(s); median {round(float(np.median(hu)))} HU",
                         f"largest sphere that fits inside the outline: {greatest_thickness_mm(pc, c.aff):.1f} mm in diameter. This is a property of the automatic outline; it is not a "
                         "thickness measurement of fluid and has not been compared with a caliper",
                         "pericardial effusion: " + tool("pericardial_effusion") + "; " + prior(cid, "pericardial_effusion"),
                         f"HoloCT yes/no for pericardial effusion: {E(hans.get('Pericardial effusion', 'not available'))}"])
            it = {"id": pid, "case_id": cid, "section": "C_pericardial_outline", "kind": "pericardial_outline", "title": "Outlined pericardial region", "reveal": reveal,
                  "q": [{"f": "mask", "label": "1. Outline correctness: is what the violet outline covers pericardial fluid?", "kind": "single", "group": "mask",
                         "opts": [["fluid", "Yes, fluid"], ["partly", "Partly"], ["not_fluid", "No"], ["cannot_adjudicate", "Cannot adjudicate"]]},
                        {"f": "not_fluid_what", "label": "If no or partly: what is it?", "kind": "text", "group": "mask"},
                        {"f": "missed", "label": "2. Does the outline miss fluid that you can see?", "kind": "single", "group": "mask", "opts": [["no", "No"], ["yes", "Yes"], ["cannot_tell", "Cannot tell"]]},
                        {"f": "presence", "label": "3. Presence (separate from the outline): is a pericardial effusion present in this scan?", "kind": "single", "group": "presence",
                         "opts": [["present", "Present"], ["absent", "Absent"], ["cannot_tell", "Cannot tell"]]},
                        {"f": "size", "label": "4. If present: your own size word", "kind": "single", "group": "presence", "opts": [["trace", "Trace"], ["mild", "Mild"], ["moderate", "Moderate"], ["large", "Large"]]},
                        {"f": "location", "label": "5. Location of the fluid you see (optional)", "kind": "text", "group": "location", "placeholder": "for example: anterior, inferior, circumferential"}, COMMENT_Q]}
            items.append(it)
            secC += card(it, imgs, f"&middot; centred on one spot of the outline &middot; in the viewer page of this scan: axial image {pn['axial']}, coronal {pn['coronal']}, sagittal {pn['sagittal']}")
        del c
    fields = []
    for it in items:
        fields += [q["f"] for q in it["q"] if q["f"] not in fields]
    data = {"build": BUILD, "review_type": "development adjudication of previously inspected cases (open-page grades exist); staged; not blinded validation",
            "fields": fields, "items": [{k: v for k, v in it.items() if k != "title"} for it in items]}
    page = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Development adjudication package</title><style>{CSS}</style></head><body><div class='page'>"
            "<h1>Development adjudication: dense lung, nodule candidates, pericardial outline</h1>"
            "<div class='muted'>Four scans you have already graded. This is a development adjudication of specific automatic outlines, not a blinded evaluation: "
            "it helps us understand what the outlines are before anything is fixed. Patient's right is on the image left. Each card is centred on one outline. "
            "Identity, outline adequacy, location and presence are asked as separate questions on purpose.</div>"
            "<div class='top card'>Reviewer: <input id='rev' placeholder='your name'> <button class='lockbtn' id='lock'>&#128274; Save and lock initial assessment</button>"
            "<button class='save' id='save'>&#128190; Save answers (CSV)</button> <span id='cnt' class='muted'></span><div class='stagebar' id='stage'></div></div>"
            "<div class='sec'><h2 style='margin:0'>A. Dense-lung regions</h2><div class='muted'>Regions of dense lung outlined automatically from CT density. Only some of the outlined regions "
            "of each scan are shown (the largest ones); the note at the end of each scan says how many.</div></div>" + secA
            + "<div class='sec'><h2 style='margin:0'>B. Nodule candidates</h2><div class='muted'>Each card is one automatically outlined structure. It is a candidate: please say what it is. "
            "At the end of each scan: whether a nodule is present, and nodules you see that are not among the candidates.</div></div>" + secB
            + "<div class='sec'><h2 style='margin:0'>C. Pericardial outline</h2><div class='muted'>One card per scan: whether the outline is right, whether it misses fluid, and separately whether "
            "an effusion is present.</div></div>" + secC
            + "<div class='muted' style='margin-top:14px'>Contains CT-RATE images (CC BY-NC-SA, research use): keep within the research group. Research use only, not a diagnosis.</div>"
            f"</div><script>{JS.replace('__DATA__', json.dumps(data))}</script></body></html>")
    (out / "development_adjudication_package.html").write_text(page, encoding="utf-8")
    with open(out / "dense_region_selection_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage[0].keys()))
        w.writeheader()
        w.writerows(coverage)
    with open(out / "dense_region_key_ANALYST_ONLY.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        w.writeheader()
        w.writerows(key_rows)
    (out / "package_manifest.json").write_text(json.dumps({
        "build": BUILD, "review_type": data["review_type"], "cases": args.cases, "cards": len(items),
        "cards_by_section": {s: sum(1 for i in items if i["section"] == s) for s in ("A_dense_regions", "B_nodule_candidates", "C_pericardial_outline")},
        "cards_by_kind": {k: sum(1 for i in items if i["kind"] == k) for k in sorted({i["kind"] for i in items})},
        "hidden_until_lock": ["analyst hypotheses", "analyst tags", "which model drew an outline", "tool measurements (volume, HU, sizes, lobe by tool)", "checklist answers",
                              "HoloCT statements and yes/no answers", "the reviewer's earlier grades"],
        "questions_kept_separate": ["identity", "outline adequacy / correctness", "location", "condition presence"],
        "dense_region_selection_rule": SELECTION_RULE, "dense_region_selection_coverage": coverage,
        "note": "The hidden information is inside the HTML file; staging is a reading procedure, not a security barrier."}, indent=2), encoding="utf-8")
    print(json.dumps({"cards": len(items), "fields": fields}, indent=0)[:600])
    for x in coverage:
        print(x)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
