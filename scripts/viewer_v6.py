"""2D canvas reviewer page for nodule review, build "checklist_viewer_v6_canvas2d_axial25_mip10"
(build_viewer_page.py --viewer v6). Display only.

Requested by the clinician reviewer: a regular axial view about 2.5 mm thick and an axial MIP about 10 mm thick, both in
the lung window, with simple controls. Viewer v4 and v5 are not changed; this module takes the v4 page script and adds
the two slab views by exact, asserted text insertions. Geometry, decoding, outlines, grading and CSV are v4's.

Three ways of showing the axial images
  "Axial 2.5 mm"   MEAN slab (opening view): every screen point is the average CT value (HU) of neighbouring native
                   images, calculated from the stored values BEFORE brightness / contrast, then rounded to a whole HU
                   (halves up: floor(mean + 0.5)). It is derived for display. It is not a scanner reconstruction.
  "MIP 10 mm"      MAXIMUM slab: the highest CT value through the slab, also taken before brightness / contrast.
  native slices    the images as stored, one at a time (under "Details"). Outlines exist only here.

Slice spacing and slice thickness are different quantities
  spacing   = distance between the centres of neighbouring images. Read from the scan's affine. Always known.
  thickness = how much tissue one image averages. NOT in the NIfTI header and NOT in the CT-RATE metadata table (that
              table has XYSpacing and ZSpacing only). For some scans the radiology report text states a section
              thickness; the page builder passes that figure on, labelled as "stated in the report", never as measured.
  The page therefore counts slabs from SPACING and says so.

Slab rule (same arithmetic as viewer v5: `slab_plan`, re-used here and in the page script; tested against each other)
  dz        = slice spacing in mm
  n         = max(1, floor(requested_mm / dz + 0.5))        whole number of images, halves up, at least 1
  coverage  = n * dz                                        the figure used for "about 2.5 mm" / "about 10 mm"
  centres   = (n - 1) * dz                                  first to last image centre
  If a slice thickness t is known, the tissue really covered is (n - 1) * dz + t. The Details panel shows that figure
  when the report states t, and says that it is unknown otherwise.
  Examples: 0.75 mm spacing -> 2.5 mm: 3 images, 2.25 mm coverage (3.0 mm if images are 1.5 mm thick);
            10 mm: 13 images, 9.75 mm.   1.25 mm spacing -> 2.5 mm: 2 images, 2.5 mm; 10 mm: 8 images, 10.0 mm.
  placement = centred on the image the reader is on; with an even n the extra image is on the superior (head) side.
  scan edge = images outside the scan are left out: the slab is shortened, never shifted or padded, and the label says so.
  scrolling = one native image per step in every view, so neighbouring slabs overlap by n - 1 images: no gaps.

When 2.5 mm cannot be made
  If the images are so far apart that n = 1 (spacing of 1.67 mm or more), nothing is averaged: the native image is
  shown at its own resolution, the button reads "Axial <spacing> mm", and the label states that a 2.5 mm view is not
  available for this scan. Images are never skipped and never relabelled as 2.5 mm. The one exception to the wording:
  if the stored images are themselves 2.25 to 2.75 mm apart, the native image IS the regular view and keeps the name
  "Axial 2.5 mm", with the real spacing in the label.
An existing 2.5 mm reconstruction would be preferred; the page builder records whether the data set holds one.

What it does not do
  * No resampling in-plane, no interpolation between images, no WebGL, no full-volume texture. One slab is computed
    when needed from the decoded scan already in memory.
  * Cache: finished slabs (one int16 per pixel of an axial image) in a least-recently-used list of at most
    SLAB_CACHE_BYTES, shared by both slab views; one int32 scratch image for the sums. Emptied when native slices are chosen.
  * Slabs are axial only. Coronal, sagittal and 3 planes show native images.
  * No outline is ever drawn on a mean slab or a MIP: an outline belongs to ONE image. The outline bar exists only with
    native slices. "Show in scan" first switches to native slices, then shows the outline there.
  * The reader's zoom is kept when the outline bar appears or disappears (same screen pixels per image pixel, same
    point at the centre; exact value restored on return; Fit stays Fit), as in viewer v5.
  * The original CT, model inputs, masks, measurements and checklist answers are not read or changed by these views.
The page opens on "Axial 2.5 mm", fitted to the scan area, lung window, no outline.
"""
from __future__ import annotations

import json

import numpy as np

import viewer_v4 as v4
from viewer_v4 import (CLIP_NOTE, MAX_OBLIQUE_DEG, ObliqueOrientationError, canonical_transform, ct_payload,  # noqa: F401  (same geometry code)
                       obliquity_deg, overlay_payload, to_canonical)
from viewer_v5 import slab_max, slab_plan  # noqa: F401  (the same slab rule and reference maximum)

VIEWER_BUILD = "checklist_viewer_v6_canvas2d_axial25_mip10"
AXIAL_MM = 2.5
MIP_MM = 10
SLAB_CACHE_BYTES = 24 * 1024 * 1024


def slab_mean(arr_can: np.ndarray, k0: int, k1: int) -> np.ndarray:
    """Reference mean over canonical images k0..k1 (inclusive), rounded to whole HU with halves up, as the page does."""
    s = np.asarray(arr_can)[:, :, k0:k1 + 1].astype(np.int64).sum(axis=2)
    return np.floor(s / float(k1 - k0 + 1) + 0.5).astype(np.int16)


def reported_thickness_mm(technique: str | None) -> float | None:
    """'... with a section thickness of 1.5 mm.' -> 1.5. The report's own statement; None when it gives no figure."""
    import re
    m = re.search(r"(?:section|slice)\s+thickness\s+of\s+(\d+(?:[.,]\d+)?)\s*mm", technique or "", re.I)
    return float(m.group(1).replace(",", ".")) if m else None


CSS = v4.CSS + """
#slbar{position:relative}
#slbar button:disabled{opacity:.4;cursor:not-allowed}
#sl25,#slmip{font-weight:600;padding-left:14px;padding-right:14px}
#slbreak{flex-basis:100%;height:0}
#slinfo{flex:1 1 0;min-width:0;height:22px;line-height:22px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#dbe7ff;font-size:14px}
#slnote{display:none;color:#ffd9a8;font-size:14px;margin-left:6px;white-space:nowrap}
body.slabmode #slnote{display:inline}
#sldet{position:absolute;left:10px;top:100%;z-index:30;width:min(660px,94%);max-height:70vh;overflow:auto;background:#1b2433;color:#e6edf7;border:1px solid #35507f;border-radius:10px;
 padding:10px 14px;font-size:14px;line-height:1.5;box-shadow:0 8px 24px rgba(0,0,0,.5)}
#sldet[hidden]{display:none}
#sldet p{margin:0 0 7px}
#sldet b{color:#ffd9a8;font-size:14px !important;margin:0 !important}
#sldet25,#sldetmip,#sldetsrc{color:#bfe3c9}
#slnat{background:#1f6f43;border-color:#1f6f43}
/* the read-out under the scan keeps room for three lines, so the scan area does not change size while scrolling */
#loc{min-height:calc(4.5em + 14px)}
#winval{display:inline-block;min-width:150px;color:#c9d3e0;font-size:14px;margin-left:4px;white-space:nowrap}
/* slab views: no outline bar; a short notice sits on the label line */
body.slabmode #ovbar{display:none}
"""

_DETAILS = (
    "<p><b>What is shown now</b></p><p id='sldetsrc'></p><p id='sldet25'></p><p id='sldetmip'></p>"
    "<p><button id='slnat' title='the images as stored, one at a time; outlines can be shown here'>Show native thin slices</button> "
    "The images as stored, one at a time. Outlines can be shown only here.</p>"
    "<p><b>Axial 2.5 mm.</b> Each point is the average CT value of neighbouring native slices. It is calculated from the stored values, before brightness / contrast, "
    "and rounded to a whole HU. It is made on this page for viewing; it is not a reconstruction from the scanner.</p>"
    "<p><b>MIP 10 mm.</b> Each point is the brightest CT value through the slab, also taken before brightness / contrast. "
    "A bright structure anywhere in the slab hides darker tissue in front of or behind it.</p>"
    "<p><b>Why &ldquo;&asymp;&rdquo;.</b> A slab is a whole number of slices, so its size is close to 2.5 or 10 mm but usually not equal. "
    "It is counted as slices &times; spacing. Spacing is the distance between neighbouring slices in this file; it is not the slice thickness.</p>"
    "<p><b>Scrolling</b> moves one native slice at a time in every view, so neighbouring slabs overlap and nothing is skipped. "
    "At the top and bottom of the scan a slab has fewer slices, and the label says so.</p>"
    "<p><b>Outlines</b> belong to one slice and would be misleading on an average or a MIP of several. &ldquo;Show in scan&rdquo; switches to native slices first.</p>")

_SLAB_BAR = """ <div class="bar" id="slbar"><b>View</b>
  <button id="sl25" class="on" title="regular axial view: the average of neighbouring native slices, about 2.5 mm">Axial 2.5 mm</button><button id="slmip" title="MIP = maximum intensity projection: the brightest value through about 10 mm">MIP 10 mm</button>
  <button id="sldetbtn" title="exact slab thickness, how it is calculated, and the native thin slices">Details</button>
  <div id="slbreak"></div><div id="slinfo"></div><span id="slnote">Outlines: native slices only (Details)</span>
  <div id="sldet" hidden>""" + _DETAILS + """<button id="sldetclose">Close</button></div>
 </div>
"""
_WIN_READOUT = '<span id="winval" title="window width (W) and level (L = centre of the window), in HU"></span>'


def _once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"viewer v4 text changed: expected exactly one {old[:60]!r}, found {text.count(old)}"
    return text.replace(old, new)


VIEWER_HTML = _once(v4.VIEWER_HTML, ' <div class="bar__AO__" id="ovbar">', _SLAB_BAR + ' <div class="bar__AO__" id="ovbar">')
VIEWER_HTML = _once(VIEWER_HTML, '<button data-w="-500,1300">Bone</button>', '<button data-w="-500,1300">Bone</button>' + _WIN_READOUT)
EARLY_JS = v4.EARLY_JS

_JS_STATE = r"""
 // ---------- slab views (viewer v6): axial mean (about 2.5 mm) and axial MIP (about 10 mm), from CT values before windowing ----------
 // slabPlan is the same arithmetic as slab_plan() in scripts/viewer_v5.py (index k: larger = more superior).
 const SL = {mode: 'mean', cache: new Map(), sum: null, computed: 0, hits: 0, reused: 0, zmem: null};
 const SL_MM = {mean: __AXIAL_MM__, mip: __MIP_MM__}, SL_CACHE_BYTES = __SLAB_CACHE_BYTES__, SLINFO = META.slab || {};
 const SL_CACHE_MAX = Math.max(2, Math.floor(SL_CACHE_BYTES / (nxy * 2)));            // slabs kept: one Int16 per pixel of an axial image each
 function slabPlan(k, mm) {
   const dz = sp[2], n = Math.max(1, Math.floor(mm / dz + 0.5)), below = (n - 1) >> 1, a = k - below, b = a + n - 1, k0 = Math.max(0, a), k1 = Math.min(nz - 1, b);
   return {requested_mm: mm, planned_images: n, planned_mm: n * dz, k0: k0, k1: k1, images: k1 - k0 + 1, actual_mm: (k1 - k0 + 1) * dz, centre_span_mm: (k1 - k0) * dz, truncated: k0 !== a || k1 !== b, centre: k};
 }
 const slabOn = () => SL.mode !== 'native' && S.mode === 'single' && S.view === 'axial';
 const slabKind = view => (slabOn() && view === 'axial') ? SL.mode : null;
 function slabData(kind, k0, k1) {                   // 'mip': maxima, 'mean': rounded means, of canonical images k0..k1 in display order (row from the top, column from the left)
   const key = kind + ':' + k0 + ':' + k1; let m = SL.cache.get(key);
   if (m) { SL.cache.delete(key); SL.cache.set(key, m); SL.hits++; return m; }         // most recently used goes last
   if (SL.cache.size >= SL_CACHE_MAX) { const old = SL.cache.keys().next().value; m = SL.cache.get(old); SL.cache.delete(old); SL.reused++; }
   else m = new Int16Array(nxy);
   const g = G.axial, w = g.w, h = g.h, cs = g.cs, rs = g.rs, n = k1 - k0 + 1;
   if (kind === 'mip') {
     for (let z = k0; z <= k1; z++) {
       const b0 = g.base(z);
       if (z === k0) { for (let r = 0, p = 0; r < h; r++) { let q = b0 + r * rs; for (let c = 0; c < w; c++, p++, q += cs) m[p] = ct[q]; } }
       else { for (let r = 0, p = 0; r < h; r++) { let q = b0 + r * rs; for (let c = 0; c < w; c++, p++, q += cs) { const v = ct[q]; if (v > m[p]) m[p] = v; } } }
     }
   } else {                                          // mean: whole-number sums first (exact), one division per pixel, halves rounded up
     const a = SL.sum || (SL.sum = new Int32Array(nxy));
     for (let z = k0; z <= k1; z++) {
       const b0 = g.base(z);
       if (z === k0) { for (let r = 0, p = 0; r < h; r++) { let q = b0 + r * rs; for (let c = 0; c < w; c++, p++, q += cs) a[p] = ct[q]; } }
       else { for (let r = 0, p = 0; r < h; r++) { let q = b0 + r * rs; for (let c = 0; c < w; c++, p++, q += cs) a[p] += ct[q]; } }
     }
     if (n === 1) { for (let p = 0; p < nxy; p++) m[p] = a[p]; } else { for (let p = 0; p < nxy; p++) m[p] = Math.floor(a[p] / n + 0.5); }
   }
   SL.cache.set(key, m); SL.computed++; return m;
 }
 function fillSlab(kind, s, u32) {                   // window (lut) is applied to the slab values, never before calculating them; no outline is drawn here
   const pl = slabPlan(s, SL_MM[kind]), m = slabData(kind, pl.k0, pl.k1), min = VMIN;
   for (let p = 0, n = m.length; p < n; p++) u32[p] = grayPx[lut[m[p] - min]];
   return pl;
 }
 const fmtMm = x => (Math.round(x * 100) / 100).toString();
 const slices = n => n + ' slice' + (n === 1 ? '' : 's');
 const nativeIs25 = () => Math.abs(sp[2] - SL_MM.mean) <= 0.25;          // the stored slices are themselves about 2.5 mm apart
 function slabSentence(kind, k) {                    // exact figures of one slab, for the Details panel
   const pl = slabPlan(k, SL_MM[kind]), dz = sp[2], t = SLINFO.reported_thickness_mm;
   if (kind === 'mean' && pl.planned_images === 1) return nativeIs25() ? 'Axial 2.5 mm here: the native slice, as stored. The slices of this scan are ' + fmtMm(dz) + ' mm apart, so nothing is averaged.'
     : 'Axial: the native slice is shown as it is. A 2.5 mm view cannot be made for this scan: its slices are ' + fmtMm(dz) + ' mm apart' + (dz < SL_MM.mean ? ', and averaging two would give ' + fmtMm(2 * dz) + ' mm.' : ', which is more than 2.5 mm.');
   return (kind === 'mip' ? 'MIP 10 mm here: brightest of ' : 'Axial 2.5 mm here: mean of ') + slices(pl.images) + ' × ' + fmtMm(dz) + ' mm spacing = ' + fmtMm(pl.actual_mm) + ' mm (asked for ' + pl.requested_mm +
     ' mm). Centres of the first and last slice: ' + fmtMm(pl.centre_span_mm) + ' mm apart.' +
     (t ? ' If each slice is ' + fmtMm(t) + ' mm thick, as the radiology report states, the tissue covered is ' + fmtMm(pl.centre_span_mm + t) + ' mm.' : ' The slice thickness is not recorded, so the tissue really covered is not known exactly.') +
     (pl.truncated ? ' Shortened at the scan edge; a full slab is ' + slices(pl.planned_images) + ', ' + fmtMm(pl.planned_mm) + ' mm.' : '');
 }
 function syncSlabUi() {
   const can = S.mode === 'single' && S.view === 'axial', on = slabOn(), dz = sp[2], k = S.cur[2];
   document.body.classList.toggle('slabmode', on);
   const wv = $('#winval'); if (wv) wv.textContent = 'W ' + Math.round(S.win[1] - S.win[0]) + '  ·  L ' + Math.round((S.win[0] + S.win[1]) / 2);
   const b25 = $('#sl25'), bm = $('#slmip'), bn = $('#slnat'); if (!b25) return;
   const one = slabPlan(k, SL_MM.mean).planned_images === 1;
   b25.textContent = one && !nativeIs25() ? 'Axial ' + fmtMm(dz) + ' mm' : 'Axial 2.5 mm';
   b25.classList.toggle('on', can && SL.mode === 'mean'); bm.classList.toggle('on', can && SL.mode === 'mip'); b25.disabled = bm.disabled = !can;
   b25.title = can ? (one ? 'native axial slices: a 2.5 mm view cannot be made for this scan' : 'regular axial view: the average of neighbouring native slices, about 2.5 mm') : 'available in the Axial view only';
   bm.title = can ? 'MIP = maximum intensity projection: the brightest value through about 10 mm' : 'available in the Axial view only';
   bn.textContent = (on ? 'Show native thin slices' : 'Native thin slices are shown') + ' (' + fmtMm(dz) + ' mm apart)'; bn.classList.toggle('on', !on);
   $('#sldetsrc').textContent = 'This scan: slices ' + fmtMm(dz) + ' mm apart (from the file). Slice thickness: ' +
     (SLINFO.reported_thickness_mm ? fmtMm(SLINFO.reported_thickness_mm) + ' mm, as stated in the radiology report (not in the image file)' : 'not recorded') + '. ' + (SLINFO.reconstruction_note || '');
   $('#sldet25').textContent = slabSentence('mean', k); $('#sldetmip').textContent = slabSentence('mip', k);
   const info = $('#slinfo'), no = imageNo('axial', k); let txt;
   if (!can) txt = 'Coronal, sagittal and 3 planes show native slices  ·  Axial 2.5 mm and MIP 10 mm are for the Axial view';
   else if (!on) txt = 'Native slice  ·  ' + fmtMm(dz) + ' mm apart  ·  Image ' + no + '  ·  outlines can be shown';
   else {
     const kind = SL.mode, pl = slabPlan(k, SL_MM[kind]), first = imageNo('axial', pl.k1), last = imageNo('axial', pl.k0);
     if (pl.planned_images === 1) txt = (kind === 'mip' ? 'MIP' : 'Axial') + ' ' + fmtMm(dz) + ' mm  ·  native slice  ·  Image ' + no + '  ·  ' +
       (kind === 'mip' ? 'same as the native slice' : nativeIs25() ? 'slices of this scan are ' + fmtMm(dz) + ' mm apart' : 'a 2.5 mm view is not available for this scan (slices ' + fmtMm(dz) + ' mm apart)');
     else {
       const approx = pl.truncated ? Math.round(pl.actual_mm * 10) / 10 : pl.requested_mm;
       txt = (kind === 'mip' ? 'MIP ≈' : 'Axial ≈') + approx + ' mm  ·  ' + (kind === 'mip' ? '' : 'mean of ') + slices(pl.images) + '  ·  Images ' + first + '–' + last + '  ·  Centre ' + no + (pl.truncated ? '  ·  shortened at scan edge' : '');
     }
   }
   info.textContent = txt; info.title = txt;
 }
 function sizeCanvas() {                             // the same sizing as paint(), done at once so that layout() sees the new scan area
   const dpr = window.devicePixelRatio || 1, cw = Math.max(1, Math.round(canvas.clientWidth * dpr)), ch = Math.max(1, Math.round(canvas.clientHeight * dpr));
   if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
 }
 function setSlab(mode) {                            // 'mean' (Axial 2.5 mm) / 'mip' (MIP 10 mm) / 'native'
   if (['mean', 'mip', 'native'].indexOf(mode) < 0) return false;
   const can = S.mode === 'single' && S.view === 'axial';
   if (mode !== 'native' && !can) return false;
   const was = slabOn(), will = mode !== 'native' && can;
   if (will && V.active) setOverlay(null);                             // no outline over an average or a projection: the active one is switched off and released
   const z = S.zp.axial, keep = was !== will && can && !(z.zoom === 1 && z.pan[0] === 0 && z.pan[1] === 0);
   let sx0 = 0; if (keep) { sizeCanvas(); sx0 = layout('axial').sx; }
   SL.mode = mode;
   if (mode === 'native') SL.cache.clear();                            // slab buffers are released when native slices are chosen
   syncSlabUi();                                                       // the outline bar appears / disappears here, so the scan area changes height
   if (keep) {                                                         // zoom is relative to Fit: rescale it so one image pixel keeps its size on screen
     sizeCanvas();
     if (SL.zmem && z.zoom === SL.zmem.to) { const m = SL.zmem; z.zoom = m.from; SL.zmem = {from: m.to, to: m.from}; }      // going back: the earlier value, exactly
     else { const sx1 = layout('axial').sx, z0 = z.zoom; if (sx1 > 0) z.zoom = Math.min(ZMAX, Math.max(ZMIN, z.zoom * sx0 / sx1)); SL.zmem = {from: z0, to: z.zoom}; }
   }
   draw(); return SL.mode;
 }
"""

_JS_HOOKS = r"""
   setSlab: setSlab, slabPlan: slabPlan,
   slab: () => { const kind = SL.mode === 'native' ? 'mean' : SL.mode, pl = slabPlan(S.cur[2], SL_MM[kind]); return Object.assign({mode: SL.mode, shown: slabKind('axial'), mm: SL_MM, spacing_mm: sp[2], images_total: nz,
     first_image_no: imageNo('axial', pl.k1), last_image_no: imageNo('axial', pl.k0), centre_image_no: imageNo('axial', S.cur[2]),
     cache_entries: SL.cache.size, cache_max: SL_CACHE_MAX, cache_bytes: SL.cache.size * nxy * 2, cache_limit_bytes: SL_CACHE_BYTES, scratch_bytes: SL.sum ? SL.sum.byteLength : 0,
     computed: SL.computed, hits: SL.hits, reused: SL.reused, info: ($('#slinfo') || {}).textContent, details: ($('#sldet') || {}).textContent, detailsOpen: !!($('#sldet') && !$('#sldet').hidden),
     btn25: ($('#sl25') || {}).textContent, winText: ($('#winval') || {}).textContent, reported_thickness_mm: SLINFO.reported_thickness_mm || null}, pl); },
   slabHU: (kind, k) => { const pl = slabPlan(k, SL_MM[kind]), m = slabData(kind, pl.k0, pl.k1); return Object.assign({w: G.axial.w, h: G.axial.h, b64: b64(new Uint8Array(m.buffer.slice(0)))}, pl); },
   shownGray: view => { view = view || AV(); const g = G[view], u = new Uint32Array(g.w * g.h), s = S.cur[g.ax], kd = slabKind(view); if (kd) fillSlab(kd, s, u); else fillSlice(view, s, u, true);
     const o = new Uint8Array(u.length), rgb = new Uint8Array(u.length); for (let i = 0; i < o.length; i++) { o[i] = u[i] & 255; rgb[i] = ((u[i] & 255) === ((u[i] >> 8) & 255) && (u[i] & 255) === ((u[i] >> 16) & 255)) ? 0 : 1; }
     return {w: g.w, h: g.h, b64: b64(o), coloured_pixels: rgb.reduce((a, x) => a + x, 0), slab: kd}; },
"""

_JS_WIRING = r"""
   $('#sl25').onclick = () => setSlab('mean'); $('#slmip').onclick = () => setSlab('mip');
   $('#slnat').onclick = () => { setSlab('native'); $('#sldet').hidden = true; };
   $('#sldetbtn').onclick = () => { $('#sldet').hidden = !$('#sldet').hidden; }; $('#sldetclose').onclick = () => { $('#sldet').hidden = true; };
   syncSlabUi(); draw();
"""


def _build_js() -> str:
    js = v4.JS
    js = _once(js, " const ZMIN = 0.1, ZMAX = 16;\n", " const ZMIN = 0.1, ZMAX = 16;\n" + _JS_STATE
               .replace("__AXIAL_MM__", str(AXIAL_MM)).replace("__MIP_MM__", str(MIP_MM)).replace("__SLAB_CACHE_BYTES__", str(SLAB_CACHE_BYTES)))
    # the one place a pane is filled before drawing: a slab instead of the native image, without an outline
    js = _once(js, "     fillSlice(view, s, b.u, true);\n", "     { const kd = slabKind(view); if (kd) fillSlab(kd, s, b.u); else fillSlice(view, s, b.u, true); }\n")
    # read-out under the scan, and the slab description, refreshed on every paint
    js = _once(js, "(ov ? '  ·  outline: ' + ov.label : '');\n }\n",
               "(ov ? '  ·  outline: ' + ov.label : '');\n"
               "   { const kd = slabKind(av); if (kd) { const pl = slabPlan(sa, SL_MM[kd]), m = slabData(kd, pl.k0, pl.k1), cr = ga.cr(S.cur);\n"
               "     $('#loc').textContent = $('#loc').textContent.replace(hu + ' HU at the cross-hair', (kd === 'mip' ? 'MIP: brightest in slab ' : 'mean of slab ') + m[cr[0] + ga.w * cr[1]] + ' HU at the cross-hair (centre slice ' + hu + ' HU)'); } }\n"
               "   syncSlabUi();\n }\n")
    # coming back to the axial view while a slab view is chosen: no outline may stay on; the bar is set before the scan is drawn
    js = _once(js, " function setView(v) {\n", " function setView(v) {\n   if (v === 'axial' && SL.mode !== 'native' && V.active) setOverlay(null);\n")
    js = _once(js, "x.classList.toggle('on', x.dataset.v === v)); draw();\n", "x.classList.toggle('on', x.dataset.v === v)); syncSlabUi(); draw();\n")
    # no outline can be switched on while a slab is shown
    js = _once(js, "     if (!V.ready || key === V.active) return;\n", "     if (!V.ready || key === V.active) return;\n     if (key && slabOn()) return;\n")
    # "Show in scan" goes to the native slices first, then shows the outline there
    js = _once(js, "     const k = tr.dataset.jump, t = META.targets[k];\n     if (!t) return;\n",
               "     const k = tr.dataset.jump, t = META.targets[k];\n     if (!t) return;\n     if (slabOn()) setSlab('native');\n")
    js = _once(js, " Object.assign(V, {\n", " Object.assign(V, {\n" + _JS_HOOKS.lstrip("\n"))
    js = _once(js, "   $('#z11').onclick = () => oneToOne();\n", "   $('#z11').onclick = () => oneToOne();\n" + _JS_WIRING.lstrip("\n"))
    js = _once(js, "V = window.__viewer = {kind: 'canvas2d',", "V = window.__viewer = {kind: 'canvas2d', slabBuild: true,")
    return js


JS = _build_js()


def viewer_html(agentonly: bool) -> str:
    return VIEWER_HTML.replace("__AO__", " agentonly" if agentonly else "")


def viewer_script(meta: dict) -> str:
    return "<script>\n" + JS.replace("__META__", json.dumps(meta)) + "\n</script>"


def standalone_page(title: str, arr: np.ndarray, affine: np.ndarray, overlays: dict | None = None, targets: dict | None = None,
                    slab: dict | None = None) -> str:
    """A minimal page with ONLY the v6 viewer (no checklist): used for the slab tests on synthetic volumes."""
    page = v4.standalone_page(title, arr, affine, overlays, targets)
    a = page.index('<div class="viewer">')
    b = page.index("</div>\n<script>\n", a)
    page = page[:a] + viewer_html(False).rstrip("\n") + page[b:]
    page = _once(page, v4.CSS + "</style>", CSS + "</style>")
    a = page.rindex("<script>")
    m0 = page.index("const META = ", a) + len("const META = ")
    meta = json.loads(page[m0:page.index("\n", m0)].rstrip().rstrip(";"))
    meta["build"] = VIEWER_BUILD
    if slab:
        meta["slab"] = slab
    return page[:a] + viewer_script(meta) + "\n</body></html>"
