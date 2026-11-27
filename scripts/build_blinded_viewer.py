"""Blinded review page for the finding-mask audit (core env): one self-contained HTML per case.

The reader sees the CT, each statement the model generated and the outline the model drew for it. The page
is built ONLY from the finding inventory (scripts/audit_build_inventory.py), the CT and the TotalSegmentator
lobe outlines. It does not read or contain: audit flags, lung_ct checklist answers or measurements, agreement
groups, sampling strata, CT-RATE labels or the radiologist report.

Staged review (configs/audit_review_protocol_*.json):
  1. Per finding: text error / finding-mask error (can be combined), or no identified error, or cannot adjudicate;
     and, as separate fields, whether the described finding is there and where the reader sees it (lobes, not in a
     lobe, indeterminate, or not assessed = the explicit default).
     'Save and lock' freezes this initial judgment, including the location; it cannot be edited afterwards.
  2. Only when every finding of the case is locked can the lobe boundaries be revealed.
  3. Per finding: are the lobe boundaries acceptable here, an anatomical-mask error, or cannot adjudicate
     (default: not reviewed). An optional revised assessment is stored next to the locked initial one.
'Save review' downloads audit_review_<case>_<reader>_<time>.json and .csv; work is autosaved in the browser.
Nothing is sent over the network. The lobe outlines are inside the file from the start (hidden until step 2).

Usage: python scripts/build_blinded_viewer.py --inventory runs/A2_audit_inventory_<..>/inventory.json
           --case valid_301_a_2 --out runs/A2_audit_inventory_<..>/review_valid_301_a_2.html   (or --all)
"""
from __future__ import annotations

import argparse
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

from build_viewer_page import NIIVUE, nii_b64, target_mm  # noqa: E402
from lung_ct.agent.evidence import LOBES  # noqa: E402
from lung_ct.audit.schema import (ANATOMY_CHOICES, CHOICE_HELP, CHOICE_LABELS, ERROR_TYPES_STAGE1,  # noqa: E402
                                  EXCLUSIVE_STAGE1, FINDING_PRESENCE, LOBE_NAMES, LOCATION_LABELS, PRESENCE_LABELS,
                                  REVIEW_SCHEMA)

E = html.escape
VIEWER_BUILD = "blinded_viewer_v2"          # v2: stage 1 also records finding presence and the reader's own location
PROTOCOL = "audit_review_protocol_v0_1_draft"
# NiiVue colour maps whose top colour (mask value 1) is a distinct pure colour ('gold' and 'hot' end in white)
LOBE_STYLE = {"RUL": ("blue", "#2563eb"), "RML": ("green", "#22c55e"), "RLL": ("warm", "#facc15"),
              "LUL": ("violet", "#d946ef"), "LLL": ("cool", "#22d3ee")}

CSS = """
*{box-sizing:border-box}body{margin:0;font:13px/1.4 'Segoe UI',Arial,sans-serif;background:#0f1115;color:#e6e6e6}
.page{display:grid;grid-template-columns:1.2fr 1fr;height:100vh}
.viewer{display:flex;flex-direction:column;border-right:1px solid #2a2f3a;min-width:0}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:8px 10px;background:#161a22;border-bottom:1px solid #2a2f3a}
.bar b{font-size:12px;color:#9aa4b2;margin-right:2px}
button{background:#232937;color:#e6e6e6;border:1px solid #333b4d;border-radius:6px;padding:4px 9px;cursor:pointer;font:inherit}
button.on{background:#2f6feb;border-color:#2f6feb}button:disabled{opacity:.45;cursor:not-allowed}
.ov{display:inline-flex;align-items:center;gap:5px;padding:3px 8px;border-radius:6px;border:1px solid #333b4d;cursor:pointer;user-select:none}
.ov.off{opacity:.45}.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
canvas{flex:1;width:100%;min-height:0;display:block}
#loc{padding:5px 10px;font:12px Consolas,monospace;color:#9aa4b2;background:#161a22;border-top:1px solid #2a2f3a}
#err{display:none;padding:20px;color:#ffb4a8}
.side{overflow:auto;padding:10px 12px;background:#f4f5f7;color:#1f2328}
h1{font-size:18px;margin:0 0 2px}.muted{color:#66707a;font-size:11.5px}
.card{background:#fff;border:1px solid #d5d9de;border-radius:10px;padding:9px 10px;margin-top:9px}
.card.active{border-color:#2f6feb;box-shadow:0 0 0 2px #cfe0ff}
.side button{background:#fff;color:#1f2328;border:1px solid #c9d1d9}
.side button.pick.sel{background:#1f2328;color:#fff;border-color:#1f2328}
.side button.go{background:#2f6feb;border-color:#2f6feb;color:#fff;font-weight:600}
.side button.lock{background:#1a7f37;border-color:#1a7f37;color:#fff;font-weight:600}
.stmt{font-size:14px;font-weight:600;margin:2px 0 6px}
.num{display:inline-block;background:#1f2328;color:#fff;border-radius:10px;padding:0 8px;font-size:11.5px;margin-right:6px}
.row{display:flex;flex-wrap:wrap;gap:5px;align-items:center;margin-top:6px}
.row .lab{font-size:11.5px;color:#57606a;width:100%}
input.cm,input#reader{font:inherit;padding:3px 6px;border:1px solid #c9d1d9;border-radius:6px}
input.cm{width:100%;margin-top:5px}
.locked{background:#eef7f0;border:1px solid #b7dfc0;border-radius:8px;padding:5px 8px;margin-top:6px;font-size:12.5px}
.warn{background:#fff4cc;border:1px solid #f0d36b;border-radius:8px;padding:6px 9px;margin-top:6px;color:#3d3000}
.steps{background:#eef4ff;border:1px solid #b9cdf5;border-radius:10px;padding:8px 10px;margin-top:8px}
.s2{display:none;border-top:1px dashed #c9d1d9;margin-top:8px;padding-top:4px}
body.revealed .s2{display:block}
#count{font-size:12px;color:#3d4b66}
"""

JS = r"""
(async () => {
 const D = DATA, $ = s => document.querySelector(s), $$ = s => Array.from(document.querySelectorAll(s));
 const KEY = 'lungct_blind_' + D.inventory_sha256.slice(0, 12) + '_' + D.case_id;
 let st = {reader: '', revealed_utc: null, f: {}};
 try { const s = JSON.parse(localStorage.getItem(KEY) || 'null'); if (s && s.f) st = s; } catch (e) {}
 for (const f of D.findings) if (!st.f[f.finding_id]) st.f[f.finding_id] = {stage1: null, draft: {choices: [], comment: ''}, stage2: null, seconds: 0};
 for (const f of D.findings) { const d = st.f[f.finding_id].draft; if (d.finding === undefined) d.finding = ''; if (!d.loc) d.loc = {status: 'not_assessed', where: []}; }
 const locText = l => !l ? 'location not recorded (earlier form)' : l.status === 'stated' ? l.where.join(', ') : D.loc_labels[l.status];
 const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
 const now = () => new Date().toISOString();

 // ---------- viewer ----------
 const nv = new niivue.Niivue({isRadiologicalConvention: true, backColor: [0, 0, 0, 1], crosshairColor: [1, 0.85, 0.1, 1],
   show3Dcrosshair: false, isColorbar: false, multiplanarShowRender: niivue.SHOW_RENDER.NEVER});
 try { await nv.attachTo('gl'); } catch (e) { $('#err').style.display = 'block'; return; }
 nv.addVolume(await niivue.NVImage.loadFromBase64({base64: D.vols.ct, name: 'ct.nii.gz', colormap: 'gray', cal_min: -1350, cal_max: 150}));
 const idx = {};
 let busy = Promise.resolve();
 function ensure(key, cmap) {            // a layer is added to the viewer the first time it is needed
   busy = busy.then(async () => {
     if (idx[key] !== undefined || !D.vols[key]) return;
     nv.addVolume(await niivue.NVImage.loadFromBase64({base64: D.vols[key], name: key + '.nii.gz', colormap: cmap, opacity: 0, cal_min: 0.5, cal_max: 1}));
     idx[key] = nv.volumes.length - 1;
   });
   return busy;
 }
 nv.setSliceType(nv.sliceTypeAxial);
 nv.onLocationChange = d => { $('#loc').textContent = d.string || ''; };
 $$('[data-w]').forEach(b => b.onclick = () => {
   const [lo, hi] = b.dataset.w.split(',').map(Number);
   nv.volumes[0].cal_min = lo; nv.volumes[0].cal_max = hi; nv.updateGLVolume();
   $$('[data-w]').forEach(x => x.classList.toggle('on', x === b));
 });
 $$('[data-v]').forEach(b => b.onclick = () => { nv.setSliceType(Number(b.dataset.v)); $$('[data-v]').forEach(x => x.classList.toggle('on', x === b)); });
 let outlineOn = false, active = null, t0 = null;      // the scan opens without any outline (reviewer request)
 async function showMask() {
   const cur = D.findings.find(x => x.finding_id === active);
   if (cur && outlineOn) await ensure(cur.key, 'red');
   for (const f of D.findings) if (idx[f.key] !== undefined) nv.setOpacity(idx[f.key], (active === f.finding_id && outlineOn) ? 0.5 : 0);
   $('#outline').classList.toggle('on', outlineOn);
 }
 $('#outline').onclick = () => { outlineOn = !outlineOn; showMask(); };
 function tick() { if (active && t0 && !st.f[active].stage1) st.f[active].seconds += (Date.now() - t0) / 1000; t0 = Date.now(); }
 function activate(fid, jump) {
   tick(); active = fid; t0 = Date.now();
   $$('.card.finding').forEach(c => c.classList.toggle('active', c.dataset.fid === fid));
   showMask();
   const f = D.findings.find(x => x.finding_id === fid);
   if (jump && f.target) { nv.scene.crosshairPos = nv.mm2frac(f.target); nv.drawScene(); }
 }

 // ---------- lobes (step 2) ----------
 async function setLobe(k, on) { if (on) await ensure(k, D.lobes.find(l => l.key === k).cmap); if (idx[k] !== undefined) nv.setOpacity(idx[k], on ? 0.3 : 0); const el = document.querySelector('.ov[data-k="' + k + '"]'); if (el) el.classList.toggle('off', !on); }
 function reveal() {
   document.body.classList.add('revealed');
   const bar = $('#lobebar'); bar.style.display = 'flex';
   if (!bar.dataset.done) {
     for (const l of D.lobes) {
       const el = document.createElement('span'); el.className = 'ov'; el.dataset.k = l.key;
       el.innerHTML = '<span class="sw" style="background:' + l.swatch + '"></span>' + l.label;
       el.onclick = () => setLobe(l.key, el.classList.contains('off'));
       bar.appendChild(el);
     }
     bar.dataset.done = '1';
   }
   for (const l of D.lobes) setLobe(l.key, true);
 }

 // ---------- forms ----------
 const L = D.labels, EXCL = D.exclusive;
 function pickLogic(list, c) {            // exclusive choices clear the others, and the other way round
   const has = list.includes(c);
   if (has) return list.filter(x => x !== c);
   if (EXCL.includes(c)) return [c];
   return list.filter(x => !EXCL.includes(x)).concat([c]);
 }
 function render() {
   let locked = 0;
   for (const f of D.findings) {
     const card = document.querySelector('.card.finding[data-fid="' + f.finding_id + '"]'), r = st.f[f.finding_id];
     const s1 = card.querySelector('.s1');
     if (r.stage1) {
       locked++;
       s1.innerHTML = '<div class="locked"><b>Initial assessment (locked ' + r.stage1.saved_utc.slice(11, 16) + ' UTC):</b> '
         + r.stage1.choices.map(c => L[c]).join(' + ') + (r.stage1.comment ? ' &mdash; ' + esc(r.stage1.comment) : '')
         + '<br><b>Finding:</b> ' + (r.stage1.finding ? D.presence_labels[r.stage1.finding] : 'not recorded (earlier form)')
         + ' &middot; <b>Where (your reading):</b> ' + locText(r.stage1.location) + '</div>';
     } else {
       s1.innerHTML = '<div class="row"><span class="lab">Step 1 &mdash; your assessment of this statement and its outline (errors can be combined)</span>'
         + D.stage1.map(c => '<button class="pick' + (r.draft.choices.includes(c) ? ' sel' : '') + '" data-c="' + c + '" title="' + (D.help[c] || '') + '">' + L[c] + '</button>').join('')
         + '</div><div class="row"><span class="lab">Is the described finding there? (answered separately from where it is)</span>'
         + D.presence.map(c => '<button class="pick' + (r.draft.finding === c ? ' sel' : '') + '" data-p="' + c + '">' + D.presence_labels[c] + '</button>').join('')
         + '</div><div class="row"><span class="lab">Where do you see it? Your own reading, before any lobe outline is shown</span>'
         + D.lobe_names.map(c => '<button class="pick' + (r.draft.loc.status === 'stated' && r.draft.loc.where.includes(c) ? ' sel' : '') + '" data-l="' + c + '">' + c + '</button>').join('')
         + Object.keys(D.loc_labels).map(c => '<button class="pick' + (r.draft.loc.status === c ? ' sel' : '') + '" data-ls="' + c + '">' + D.loc_labels[c] + '</button>').join('')
         + '</div><input class="cm" placeholder="Comment (optional)" value="' + esc(r.draft.comment) + '">'
         + '<div class="row"><button class="lock"' + (r.draft.choices.length && r.draft.finding ? '' : ' disabled') + '>Save and lock initial assessment</button>'
         + '<span class="muted">Needs an assessment and an answer to "is the finding there". Cannot be changed after saving.</span></div>';
       s1.querySelectorAll('[data-p]').forEach(b => b.onclick = () => { r.draft.finding = r.draft.finding === b.dataset.p ? '' : b.dataset.p; save(); render(); });
       s1.querySelectorAll('[data-l]').forEach(b => b.onclick = () => {
         const w = r.draft.loc.status === 'stated' ? r.draft.loc.where.slice() : [], k = w.indexOf(b.dataset.l);
         if (k >= 0) w.splice(k, 1); else w.push(b.dataset.l);
         w.sort((x, y) => D.lobe_names.indexOf(x) - D.lobe_names.indexOf(y));
         r.draft.loc = w.length ? {status: 'stated', where: w} : {status: 'not_assessed', where: []}; save(); render(); });
       s1.querySelectorAll('[data-ls]').forEach(b => b.onclick = () => { r.draft.loc = {status: b.dataset.ls, where: []}; save(); render(); });
       s1.querySelectorAll('.pick[data-c]').forEach(b => b.onclick = () => { activate(f.finding_id, false); r.draft.choices = pickLogic(r.draft.choices, b.dataset.c); save(); render(); });
       s1.querySelector('.cm').oninput = e => { r.draft.comment = e.target.value; save(); };
       s1.querySelector('.lock').onclick = () => {
         if (!r.draft.choices.length || !r.draft.finding) return;
         tick();
         r.stage1 = {choices: r.draft.choices.slice(), comment: r.draft.comment.trim(), saved_utc: now(), seconds: Math.round(r.seconds),
                     finding: r.draft.finding, location: {status: r.draft.loc.status, where: r.draft.loc.where.slice()}};
         save(); render();
       };
     }
     const s2 = card.querySelector('.s2'), a = r.stage2 || {anatomy: '', revised: [], comment: ''};
     s2.innerHTML = '<div class="row"><span class="lab">Step 3 &mdash; lobe boundaries around this finding (leave empty = not reviewed)</span>'
       + D.anatomy.map(c => '<button class="pick' + (a.anatomy === c ? ' sel' : '') + '" data-a="' + c + '">' + L[c] + '</button>').join('') + '</div>'
       + '<div class="row"><span class="lab">Optional &mdash; revised assessment after seeing the lobes (your initial one stays stored)</span>'
       + D.stage1.map(c => '<button class="pick' + (a.revised.includes(c) ? ' sel' : '') + '" data-r="' + c + '">' + L[c] + '</button>').join('') + '</div>'
       + '<input class="cm" placeholder="Comment on anatomy or revision (optional)" value="' + esc(a.comment) + '">';
     const put = () => { a.saved_utc = now(); r.stage2 = a; save(); };
     s2.querySelectorAll('[data-a]').forEach(b => b.onclick = () => { a.anatomy = a.anatomy === b.dataset.a ? '' : b.dataset.a; put(); render(); });
     s2.querySelectorAll('[data-r]').forEach(b => b.onclick = () => { a.revised = pickLogic(a.revised, b.dataset.r); put(); render(); });
     s2.querySelector('.cm').oninput = e => { a.comment = e.target.value; put(); };
   }
   const all = locked === D.findings.length;
   $('#count').textContent = locked + ' of ' + D.findings.length + ' initial assessments locked';
   $('#reveal').disabled = !all || !!st.revealed_utc;
   $('#reveal').textContent = st.revealed_utc ? 'Lobe boundaries revealed' : 'Step 2: reveal lobe boundaries';
   if (st.revealed_utc) reveal();
 }
 function esc(s) { return String(s || '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c])); }
 $$('.card.finding').forEach(c => {
   c.querySelector('.go').onclick = () => { outlineOn = true; activate(c.dataset.fid, true); };
   c.addEventListener('focusin', () => { if (active !== c.dataset.fid) activate(c.dataset.fid, false); });
 });
 $('#reveal').onclick = () => { if (confirm('Reveal the lobe boundaries? Your initial assessments stay locked.')) { st.revealed_utc = now(); save(); render(); } };
 $('#reader').value = st.reader; $('#reader').oninput = e => { st.reader = e.target.value; save(); };

 // ---------- export ----------
 function record() {
   return {schema: D.review_schema, protocol: D.protocol, viewer_build: D.viewer_build, inventory_sha256: D.inventory_sha256,
     case_id: D.case_id, reader_id: st.reader.trim(), exported_utc: now(), lobes_revealed_utc: st.revealed_utc,
     reviews: D.findings.map(f => { const r = st.f[f.finding_id], a = r.stage2;
       return {finding_id: f.finding_id, statement: f.statement, mask_empty: !D.vols[f.key],
         status: !r.stage1 ? 'not_reviewed' : (a && a.anatomy ? 'complete' : 'initial_only'),
         stage1: r.stage1, stage2: a ? {anatomy_assessed: !!a.anatomy, anatomy: a.anatomy || null, revised_choices: a.revised, comment: (a.comment || '').trim(), saved_utc: a.saved_utc} : null}; })};
 }
 function download(name, text, type) {
   const u = URL.createObjectURL(new Blob([text], {type})), x = document.createElement('a');
   x.href = u; x.download = name; document.body.appendChild(x); x.click(); x.remove(); setTimeout(() => URL.revokeObjectURL(u), 1000);
 }
 const q = v => '"' + String(v === null || v === undefined ? '' : v).replace(/"/g, '""') + '"';
 $('#export').onclick = () => {
   if (!st.reader.trim()) { alert('Please enter your reader ID first.'); $('#reader').focus(); return; }
   const rec = record(), open = rec.reviews.filter(r => r.status === 'not_reviewed').length;
   if (open && !confirm(open + ' finding(s) have no locked initial assessment and will be exported as "not reviewed". Continue?')) return;
   const stamp = rec.exported_utc.replace(/[-:]/g, '').slice(0, 15), base = 'audit_review_' + D.case_id + '_' + st.reader.trim().replace(/\W+/g, '_') + '_' + stamp;
   download(base + '.json', JSON.stringify(rec, null, 2), 'application/json');
   const head = ['finding_id', 'reader_id', 'status', 'statement', 'mask_empty', 'initial_choices', 'initial_comment', 'initial_saved_utc', 'initial_seconds',
     'lobes_revealed_utc', 'anatomy_assessed', 'anatomy', 'revised_choices', 'stage2_comment', 'stage2_saved_utc', 'inventory_sha256', 'protocol', 'viewer_build',
     'review_schema', 'initial_finding', 'initial_location_status', 'initial_location_lobes'];
   const ls = r => !r.stage1 ? '' : r.stage1.location ? r.stage1.location.status : 'not_recorded_form_v1';
   const rows = rec.reviews.map(r => [r.finding_id, rec.reader_id, r.status, r.statement, r.mask_empty, r.stage1 ? r.stage1.choices.join('+') : '', r.stage1 ? r.stage1.comment : '',
     r.stage1 ? r.stage1.saved_utc : '', r.stage1 ? r.stage1.seconds : '', rec.lobes_revealed_utc, r.stage2 ? r.stage2.anatomy_assessed : false, r.stage2 ? r.stage2.anatomy : '',
     r.stage2 ? r.stage2.revised_choices.join('+') : '', r.stage2 ? r.stage2.comment : '', r.stage2 ? r.stage2.saved_utc : '', rec.inventory_sha256, rec.protocol, rec.viewer_build,
     rec.schema, r.stage1 ? (r.stage1.finding || '') : '', ls(r), r.stage1 && r.stage1.location ? r.stage1.location.where.join(' ') : ''].map(q).join(','));
   setTimeout(() => download(base + '.csv', '﻿' + head.join(',') + '\r\n' + rows.join('\r\n') + '\r\n', 'text/csv'), 400);
 };
 $('#clear').onclick = () => { if (confirm('Delete everything entered for this case on this computer?')) { localStorage.removeItem(KEY); location.reload(); } };
 render();
 if (D.findings.length) activate(D.findings[0].finding_id, false);
 window.__ready = true;
})();
"""


def build(inv: dict, case_id: str, out: Path, only: set | None = None) -> None:
    fs = sorted((f for f in inv["findings"] if f["case_id"] == case_id and (only is None or f["finding_id"] in only)),
                key=lambda f: f["index"])
    if not fs:
        raise SystemExit(f"no findings for {case_id} in the inventory")
    ct = nib.load(str(ROOT / fs[0]["ct"]["file"]))
    hu = np.asanyarray(ct.dataobj).astype(np.float32)

    def load(path: Path) -> np.ndarray:
        img = nib.load(str(path))
        if img.shape != ct.shape or not np.allclose(img.affine, ct.affine, atol=1e-3):
            raise SystemExit(f"{case_id}: geometry of {path.name} differs from the CT - not building the page")
        return np.asanyarray(img.dataobj) > 0

    seg = ROOT / "data" / "derived" / case_id / "segmentation" / "total"
    lobes = {short: load(seg / f"{name}.nii.gz") for name, (_, _, short) in LOBES.items()}
    lung = np.zeros(ct.shape, bool)
    for m in lobes.values():
        lung |= m
    body = ndimage.binary_opening(hu > -500, iterations=2) | lung
    idx = np.argwhere(body)
    lo, hi = np.maximum(idx.min(0) - 8, 0), np.minimum(idx.max(0) + 9, hu.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    aff = ct.affine.copy()
    aff[:3, 3] = (ct.affine @ np.append(lo, 1.0))[:3]
    vols = {"ct": nii_b64(np.clip(hu[sl], -1024, 3071).astype(np.int16), aff)}
    del hu, body
    findings, cards = [], ""
    for n, f in enumerate(fs, 1):
        key = f"f{f['index']:03d}"
        m = load(ROOT / f["mask"]["file"])
        outside = int(m.sum() - m[sl].sum())
        vol_ml = float(m.sum()) * abs(float(np.linalg.det(ct.affine[:3, :3]))) / 1000.0
        m = m[sl]
        target = target_mm(m, aff)
        if m.any():
            vols[key] = nii_b64(m.astype(np.uint8), aff)
        findings.append({"finding_id": f["finding_id"], "key": key, "statement": f["statement"], "target": target})
        note = (f" <span class='muted'>outline size {(f'{vol_ml:.0f}' if vol_ml >= 10 else f'{vol_ml:.2g}')} mL{' (very small: use the cross-hair to find it)' if vol_ml < 1 else ''}</span>"
                if m.any() else "<div class='warn'>The model drew no outline for this statement.</div>")
        if outside:
            note += f"<div class='warn'>Part of the outline ({outside} voxels) lies outside the displayed body region.</div>"
        stmt = E(f["statement"]) if f["statement"] else "<i>(the model gave no statement for this outline)</i>"
        cards += (f"<div class='card finding' data-fid='{E(f['finding_id'])}' tabindex='0'>"
                  f"<div class='stmt'><span class='num'>{n}</span>{stmt}</div>"
                  f"<button class='go'>&#128065; Show outline in scan</button>{note}"
                  f"<div class='s1'></div><div class='s2'></div></div>")
    lobe_list = []
    for short, m in lobes.items():
        key = f"lobe_{short}"
        vols[key] = nii_b64(m[sl].astype(np.uint8), aff)
        lobe_list.append({"key": key, "label": short, "cmap": LOBE_STYLE[short][0], "swatch": LOBE_STYLE[short][1]})
    data = {"case_id": case_id, "inventory_sha256": inv["inventory_sha256"], "protocol": PROTOCOL, "viewer_build": VIEWER_BUILD,
            "review_schema": REVIEW_SCHEMA, "vols": vols, "findings": findings, "lobes": lobe_list,
            "stage1": list(ERROR_TYPES_STAGE1) + list(EXCLUSIVE_STAGE1), "exclusive": list(EXCLUSIVE_STAGE1),
            "anatomy": list(ANATOMY_CHOICES), "labels": CHOICE_LABELS, "help": CHOICE_HELP,
            "presence": list(FINDING_PRESENCE), "presence_labels": PRESENCE_LABELS, "lobe_names": list(LOBE_NAMES),
            "loc_labels": LOCATION_LABELS}
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Blinded finding review - {E(case_id)}</title>
<style>{CSS}</style></head><body><div class="page">
<div class="viewer">
 <div class="bar"><b>Window</b>
  <button data-w="-1350,150" class="on">Lung</button><button data-w="-160,240">Soft tissue</button><button data-w="-500,1300">Bone</button>
  <b style="margin-left:10px">View</b>
  <button data-v="0" class="on">Axial</button><button data-v="1">Coronal</button><button data-v="2">Sagittal</button><button data-v="3">3 planes</button>
  <b style="margin-left:10px">Finding</b><button id="outline">Outline on/off</button>
 </div>
 <div class="bar" id="lobebar" style="display:none"><b>Lobe boundaries</b></div>
 <canvas id="gl"></canvas><div id="err">This viewer needs WebGL2 (current Chrome or Edge).</div>
 <div id="loc">Scroll: next/previous slice &middot; click: move cross-hair &middot; right-drag: brightness/contrast</div>
</div>
<div class="side">
 <h1>{E(case_id)} &mdash; review of AI-generated findings</h1>
 <div class="muted">Research use only, not a diagnosis &middot; patient's right is on the image left &middot; {len(fs)} generated finding(s) &middot; {PROTOCOL}</div>
 <div class="steps"><b>How to review</b><br>
  1. The scan opens without any outline. For each statement, press <i>Show outline in scan</i>, judge the statement and its red outline, say whether the finding is there and where you see it, then <i>Save and lock</i>.<br>
  2. When all are locked, reveal the lobe boundaries.<br>
  3. Say whether the lobe boundaries around each finding are acceptable. You may add a revised assessment; the first one is kept.<br>
  <span class="muted">Text error = {E(CHOICE_HELP['text_error'])}. Finding-mask error = {E(CHOICE_HELP['finding_mask_error'])}.
  The list may be incomplete: findings the AI did not mention are not part of this review.</span></div>
 <div class="card"><div class="row"><b>Reader ID</b> <input id="reader" placeholder="e.g. initials"> <span id="count"></span></div></div>
 {cards}
 <div class="card"><div class="row"><button id="reveal" class="go" disabled>Step 2: reveal lobe boundaries</button>
  <span class="muted">Available when every initial assessment is locked. Lobe boundaries are automatic (TotalSegmentator) and can be wrong.</span></div></div>
 <div class="card"><div class="row"><button id="export" class="lock">Save review (JSON + CSV)</button><button id="clear">Clear this case</button>
  <span class="muted">Files download to your computer; nothing is sent anywhere. Please send both files back.</span></div>
  <div class="muted" style="margin-top:6px">Contains CT-RATE images (CC BY-NC-SA, research use): keep within the research group.</div></div>
</div></div>
<script>
/* NiiVue 0.69.0, BSD-2-Clause, (c) NiiVue contributors, https://github.com/niivue/niivue (see scripts/vendor/NIIVUE_LICENSE.txt) */
{NIIVUE.read_text(encoding='utf-8')}
</script>
<script>
const DATA = {json.dumps(data)};
{JS}
</script></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"-> {out}  ({out.stat().st_size / 1e6:.1f} MB; {len(fs)} findings)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--case")
    ap.add_argument("--all", action="store_true", help="one page per case in the inventory, next to it")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selection", type=Path, help="selection.json from audit_sampling_manifest.py (finding ids only); "
                    "only these findings are shown, in generated order")
    args = ap.parse_args()
    ip = args.inventory if args.inventory.is_absolute() else ROOT / args.inventory
    inv = json.loads(ip.read_text(encoding="utf-8"))
    only = None
    if args.selection:
        sel = json.loads((args.selection if args.selection.is_absolute() else ROOT / args.selection).read_text(encoding="utf-8"))
        if sel["inventory_sha256"] != inv["inventory_sha256"]:
            raise SystemExit("selection file was made for a different inventory")
        only = set(sel["finding_ids"])
    pool = [f for f in inv["findings"] if only is None or f["finding_id"] in only]
    cases = sorted({f["case_id"] for f in pool}) if args.all else [args.case]
    if not cases or cases == [None]:
        raise SystemExit("use --case ID or --all")
    for c in cases:
        out = (args.out if args.out.is_absolute() else ROOT / args.out) if (args.out and not args.all) else ip.parent / f"review_{c}.html"
        build(inv, c, out, only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
