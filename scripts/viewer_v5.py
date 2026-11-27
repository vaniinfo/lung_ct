"""2D canvas reviewer page with an optional axial sliding-slab MIP, build "checklist_viewer_v5_2_canvas2d_mip"
(build_viewer_page.py --viewer v5). Display only.

Viewer v4 (scripts/viewer_v4.py) is not changed and its pages are rebuilt byte for byte. This module takes the v4 page
script and adds the MIP mode by exact, asserted text insertions; geometry, decoding, outlines, grading and CSV are v4's.

What MIP is here
  A maximum intensity projection through a slab of neighbouring AXIAL images: each screen point shows the highest CT
  value found along the head-to-foot line through the slab. The maximum is taken from the stored CT values (HU) BEFORE
  brightness / contrast (the window) is applied. Changing the window re-colours the same maxima.

Slab definition (the same arithmetic in `slab_plan` below and in the page script; tested against each other)
  dz        = slice spacing along the head-foot axis, in mm, from the scan's affine (never a fixed number of slices)
  n         = max(1, floor(requested_mm / dz + 0.5))      nearest whole number of images, halves rounded up, at least 1
  thickness = n * dz                                      COVERAGE convention: each image stands for one spacing of tissue.
              This figure is shown, next to the requested one. The distance from the centre of the first image to the
              centre of the last is (n - 1) * dz and is shown as well (13 images at 0.75 mm: 9.75 mm coverage, 9.0 mm
              between the outer image centres). Both describe the same 13 images; the projection does not depend on
              which figure is quoted.
  spacing is not slice thickness: dz is the distance between neighbouring image centres, read from the affine. The
              acquisition / reconstruction slice thickness (how much tissue one image averages) is not stored in the
              NIfTI affine and is not in the CT-RATE metadata we hold (it lists z spacing only). If images are thicker
              than their spacing (overlapping reconstruction) the tissue really covered is (n - 1) * dz + thickness,
              which is more than n * dz; if thinner, less. The page therefore says "spacing", never "slice thickness".
  placement = centred on the image the reader is on (the "centre image"). With an even n the slab cannot be symmetric:
              the extra image is on the superior (head) side.
  scan edge = images that would lie outside the scan are left out; the slab is shortened, never shifted or padded.
              The page then shows the shortened image count and thickness and says "shortened at the edge of the scan".
  If the spacing is larger than the request, n = 1 and the "MIP" equals the single image; the page says so.
  Scrolling moves the centre image by one image, so neighbouring slabs overlap by n - 1 images.

What it does not do
  * No resampling, no interpolation between images, no full-volume graphics texture, no WebGL. One slab is computed
    when it is needed, from the decoded scan already in memory.
  * Cache: finished slabs (one int16 value per pixel of one axial image) are kept in a least-recently-used list of at
    most MIP_CACHE_BYTES; the oldest buffer is reused for the next slab. Leaving MIP mode empties the list.
  * Axial only. Choosing another view, or 3 planes, leaves MIP mode.
  * Outlines are hidden in MIP mode: an outline is drawn for ONE image, and placing it over a projection of several
    images would mark tissue that is not at that depth. "Show in scan" first returns to the single image.
    The outline bar is removed while MIP is on; a short notice next to the MIP buttons says so. The scan area is
    therefore a little taller in MIP.
  * Zoom chosen by the reader is kept across that change of size: zoom is stored relative to Fit, so when the scan
    area changes height the stored zoom is rescaled to keep the same number of screen pixels per image pixel and the
    same image point at the centre of the scan area. Going back restores the earlier zoom value exactly.
    If the reader is at Fit (zoom 100 %, no pan), the image stays at Fit and uses the taller area.
    The page opens at Fit, as viewer v4 does.
  * Label: "MIP ~10 mm . 13 slices . Images 183-195 . Centre 189" (the page uses the approximately-equal sign). The
    millimetre figure is the thickness asked for; it is approximate because the slab is a whole number of slices.
    If the slab is shortened at the scan edge, or is a single slice, the figure is slices x spacing instead and the
    label says why. Spacing, exact coverage and centre distance of the current slab are in "? MIP help".
  * The read-out next to the window presets shows width (W) and level (L, the centre) in HU for any window.
  * Model inputs, predictions, measurements and checklist answers are not read or changed by this mode.
The page opens exactly as v4 does: one axial image, no outline, 10 mm selected but MIP off.
"""
from __future__ import annotations

import json
import math

import numpy as np

import viewer_v4 as v4
from viewer_v4 import (CLIP_NOTE, MAX_OBLIQUE_DEG, ObliqueOrientationError, canonical_transform, ct_payload,  # noqa: F401  (same geometry code)
                       obliquity_deg, overlay_payload, to_canonical)

VIEWER_BUILD = "checklist_viewer_v5_2_canvas2d_mip"
MIP_OPTIONS_MM = (5, 8, 10)
MIP_DEFAULT_MM = 10
MIP_CACHE_BYTES = 24 * 1024 * 1024


def slab_plan(k: int, nz: int, dz: float, requested_mm: float) -> dict:
    """Images of the slab centred on image index k (canonical index: larger k = more superior). See the module text."""
    n = max(1, int(math.floor(requested_mm / dz + 0.5)))
    below = (n - 1) // 2
    a, b = k - below, k - below + n - 1
    k0, k1 = max(0, a), min(nz - 1, b)
    return {"requested_mm": requested_mm, "planned_images": n, "planned_mm": n * dz, "k0": k0, "k1": k1, "images": k1 - k0 + 1,
            "actual_mm": (k1 - k0 + 1) * dz, "centre_span_mm": (k1 - k0) * dz, "truncated": (k0 != a) or (k1 != b), "centre": k}


def slab_max(arr_can: np.ndarray, k0: int, k1: int) -> np.ndarray:
    """Reference maximum over canonical images k0..k1 (inclusive) of a RAS+ volume, in canonical [i, j] order."""
    return np.asarray(arr_can)[:, :, k0:k1 + 1].max(axis=2)


CSS = v4.CSS + """
#mipbar{position:relative}
#mipbar button:disabled{opacity:.4;cursor:not-allowed}
#mipcentre{visibility:hidden;background:#1f6f43;border-color:#1f6f43}
body.mipmode #mipcentre{visibility:visible}
#mipbreak{flex-basis:100%;height:0}
#mipinfo{flex:1 1 0;min-width:0;height:22px;line-height:22px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#7f8a99;font-size:14px}
body.mipmode #mipinfo{color:#dbe7ff}
#miphelp{position:absolute;left:10px;top:100%;z-index:30;width:min(620px,92%);background:#1b2433;color:#e6edf7;border:1px solid #35507f;border-radius:10px;
 padding:10px 14px;font-size:14px;line-height:1.5;box-shadow:0 8px 24px rgba(0,0,0,.5)}
#miphelp[hidden]{display:none}
#miphelp p{margin:0 0 7px}
#miphelp b{color:#ffd9a8;font-size:14px !important;margin:0 !important}
/* the read-out under the scan keeps room for three lines, so the scan area has the same size in single-image and MIP mode */
#loc{min-height:calc(4.5em + 14px)}
#winval{display:inline-block;min-width:150px;color:#c9d3e0;font-size:14px;margin-left:4px;white-space:nowrap}
/* MIP: the outline bar is removed; a short notice sits next to the MIP buttons */
body.mipmode #ovbar{display:none}
#mipnoout{display:none;color:#ffd9a8;font-size:14px;margin-left:6px;white-space:nowrap}
body.mipmode #mipnoout{display:inline}
#miphelpnow{color:#bfe3c9}
"""

_MIP_HELP = (
    "<p><b>MIP (maximum intensity projection).</b> Every point shows the brightest CT value found through a slab of neighbouring axial slices. "
    "The values are taken before brightness / contrast, so changing the window only re-colours the same picture.</p>"
    "<p id='miphelpnow'></p>"
    "<p><b>Why &ldquo;&asymp;&rdquo;.</b> The slab is a whole number of slices, so its size is close to the thickness you choose but usually not equal to it. "
    "Coverage is counted as slices &times; spacing (13 slices &times; 0.75 mm = 9.75 mm for &ldquo;10 mm&rdquo;). "
    "The centres of the first and last slice are one spacing less apart (9 mm in that example).</p>"
    "<p><b>Spacing is not slice thickness.</b> Spacing is the distance between neighbouring slices in this file. "
    "The scanner&rsquo;s slice thickness is not stored in the file, and this page does not know it.</p>"
    "<p><b>At the top and bottom of the scan</b> the slab has fewer slices, and the label says so.</p>"
    "<p><b>Outlines are not shown in MIP.</b> An outline belongs to one slice and would be misleading on top of several. Choose Single slice to see outlines again.</p>"
    "<p><b>Keep in mind:</b> a bright structure anywhere in the slab hides darker tissue in front of or behind it. Check a finding on the single slice.</p>")

_MIP_BAR = """ <div class="bar" id="mipbar"><b>Image</b>
  <button id="mipoff" class="on" title="one axial image at a time, as the scan was stored">Single slice</button><button id="mipon" title="MIP = maximum intensity projection: the brightest value through a slab of neighbouring axial images">MIP</button>
  <b style="margin-left:10px">Slab</b><button data-mip="5">5 mm</button><button data-mip="8">8 mm</button><button data-mip="10" class="on">10 mm</button>
  <button id="mipcentre" title="leave MIP and show the single image at the centre of this slab">&#8617; Single image 0000</button>
  <button id="miphelpbtn" title="what the MIP view shows and how the slab thickness is counted">? MIP help</button>
  <div id="mipbreak"></div><div id="mipinfo"></div><span id="mipnoout">Outlines are not shown in MIP view</span>
  <div id="miphelp" hidden>""" + _MIP_HELP + """<button id="miphelpclose">Close</button></div>
 </div>
"""
_WIN_READOUT = '<span id="winval" title="window width (W) and level (L = centre of the window), in HU"></span>'


def _once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"viewer v4 text changed: expected exactly one {old[:60]!r}, found {text.count(old)}"
    return text.replace(old, new)


VIEWER_HTML = _once(v4.VIEWER_HTML, ' <div class="bar__AO__" id="ovbar">', _MIP_BAR + ' <div class="bar__AO__" id="ovbar">')
VIEWER_HTML = _once(VIEWER_HTML, '<button data-w="-500,1300">Bone</button>', '<button data-w="-500,1300">Bone</button>' + _WIN_READOUT)
EARLY_JS = v4.EARLY_JS

_JS_STATE = r"""
 // ---------- MIP (viewer v5): axial sliding slab, maxima taken from CT values before windowing ----------
 // slabPlan is the same arithmetic as slab_plan() in scripts/viewer_v5.py (index k: larger = more superior).
 const MIP = {on: false, mm: __MIP_DEFAULT__, cache: new Map(), computed: 0, hits: 0, reused: 0};
 const MIP_OPTIONS = __MIP_OPTIONS__, MIP_CACHE_BYTES = __MIP_CACHE_BYTES__;
 const MIP_CACHE_MAX = Math.max(2, Math.floor(MIP_CACHE_BYTES / (nxy * 2)));          // slabs kept: one Int16 per pixel of an axial image each
 function slabPlan(k, mm) {
   const dz = sp[2], n = Math.max(1, Math.floor(mm / dz + 0.5)), below = (n - 1) >> 1, a = k - below, b = a + n - 1, k0 = Math.max(0, a), k1 = Math.min(nz - 1, b);
   return {requested_mm: mm, planned_images: n, planned_mm: n * dz, k0: k0, k1: k1, images: k1 - k0 + 1, actual_mm: (k1 - k0 + 1) * dz, centre_span_mm: (k1 - k0) * dz, truncated: k0 !== a || k1 !== b, centre: k};
 }
 const mipShown = view => MIP.on && S.mode === 'single' && S.view === 'axial' && view === 'axial';
 function slabMax(k0, k1) {                          // maxima of canonical images k0..k1 in display order (row from the top, column from the left)
   const key = k0 + ':' + k1; let m = MIP.cache.get(key);
   if (m) { MIP.cache.delete(key); MIP.cache.set(key, m); MIP.hits++; return m; }      // most recently used goes last
   if (MIP.cache.size >= MIP_CACHE_MAX) { const old = MIP.cache.keys().next().value; m = MIP.cache.get(old); MIP.cache.delete(old); MIP.reused++; }
   else m = new Int16Array(nxy);
   const g = G.axial, w = g.w, h = g.h, cs = g.cs, rs = g.rs;
   for (let z = k0; z <= k1; z++) {
     const b0 = g.base(z);
     if (z === k0) { for (let r = 0, p = 0; r < h; r++) { let q = b0 + r * rs; for (let c = 0; c < w; c++, p++, q += cs) m[p] = ct[q]; } }
     else { for (let r = 0, p = 0; r < h; r++) { let q = b0 + r * rs; for (let c = 0; c < w; c++, p++, q += cs) { const v = ct[q]; if (v > m[p]) m[p] = v; } } }
   }
   MIP.cache.set(key, m); MIP.computed++; return m;
 }
 function fillMip(s, u32) {                          // window (lut) is applied to the maxima, never before taking them; no outline is drawn here
   const pl = slabPlan(s, MIP.mm), m = slabMax(pl.k0, pl.k1), min = VMIN;
   for (let p = 0, n = m.length; p < n; p++) u32[p] = grayPx[lut[m[p] - min]];
   return pl;
 }
 const fmtMm = x => (Math.round(x * 100) / 100).toString();
 function syncMipUi() {
   const can = S.mode === 'single' && S.view === 'axial', on = MIP.on && can;
   document.body.classList.toggle('mipmode', on);
   const wv = $('#winval'); if (wv) wv.textContent = 'W ' + Math.round(S.win[1] - S.win[0]) + '  ·  L ' + Math.round((S.win[0] + S.win[1]) / 2);
   const bOn = $('#mipon'), bOff = $('#mipoff'); if (!bOn) return;
   bOn.classList.toggle('on', on); bOff.classList.toggle('on', !on); bOn.disabled = !can;
   bOn.title = can ? 'MIP = maximum intensity projection: the brightest value through a slab of neighbouring axial images' : 'MIP is available in the Axial view only';
   $$('[data-mip]').forEach(x => x.classList.toggle('on', Number(x.dataset.mip) === MIP.mm));
   const info = $('#mipinfo'), cen = $('#mipcentre'), now = $('#miphelpnow');
   if (now) { const q = slabPlan(S.cur[2], MIP.mm); now.textContent = (on ? 'This slab: ' : 'With ' + MIP.mm + ' mm selected, the slab here would be: ') + q.images + ' slice' + (q.images === 1 ? '' : 's') + ' × ' + fmtMm(sp[2]) +
     ' mm spacing = ' + fmtMm(q.actual_mm) + ' mm coverage (asked for ' + q.requested_mm + ' mm). Centres of the first and last slice: ' + fmtMm(q.centre_span_mm) + ' mm apart.' +
     (q.truncated ? ' Shortened at the scan edge; a full slab is ' + q.planned_images + ' slices, ' + fmtMm(q.planned_mm) + ' mm.' : ''); }
   if (!on) { info.textContent = can ? 'Single axial image  ·  MIP is off' : 'MIP is available in the Axial view only'; info.title = ''; return; }
   const k = S.cur[2], pl = slabPlan(k, MIP.mm), first = imageNo('axial', pl.k1), last = imageNo('axial', pl.k0);
   // one short line; spacing, exact coverage and centre distance are under "? MIP help"
   const full = !pl.truncated && pl.planned_images > 1, approx = full ? pl.requested_mm : Math.round(pl.actual_mm * 10) / 10;
   const txt = 'MIP ≈' + approx + ' mm  ·  ' + pl.images + ' slice' + (pl.images === 1 ? '' : 's') + '  ·  Images ' + first + '–' + last + '  ·  Centre ' + imageNo('axial', k) +
     (pl.truncated ? '  ·  shortened at scan edge' : '') + (pl.planned_images === 1 ? '  ·  same as single image' : '');
   info.textContent = txt; info.title = txt;
   cen.textContent = '↩ Single image ' + imageNo('axial', k);
 }
 function sizeCanvas() {                             // the same sizing as paint(), done at once so that layout() sees the new scan area
   const dpr = window.devicePixelRatio || 1, cw = Math.max(1, Math.round(canvas.clientWidth * dpr)), ch = Math.max(1, Math.round(canvas.clientHeight * dpr));
   if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
 }
 function setMip(on, mm) {                           // on: true / false / undefined (keep); mm: one of MIP_OPTIONS / undefined (keep)
   if (mm !== undefined && mm !== null) { mm = Number(mm); if (MIP_OPTIONS.indexOf(mm) < 0) return false; MIP.mm = mm; }
   if (on !== undefined && on !== null) {
     on = !!on && S.mode === 'single' && S.view === 'axial';
     if (on && !MIP.on && V.active) setOverlay(null);                  // outlines are not drawn over a projection: the active one is switched off and released
     const z = S.zp.axial, keep = on !== MIP.on && S.mode === 'single' && S.view === 'axial' && !(z.zoom === 1 && z.pan[0] === 0 && z.pan[1] === 0);
     let sx0 = 0; if (keep) { sizeCanvas(); sx0 = layout('axial').sx; }
     MIP.on = on;
     if (!on) MIP.cache.clear();                                       // slab buffers are released when MIP is left
     syncMipUi();                                                      // the outline bar appears / disappears here, so the scan area changes height
     if (keep) {                                                       // zoom is relative to Fit: rescale it so one image pixel keeps its size on screen
       sizeCanvas();
       if (MIP.zmem && z.zoom === MIP.zmem.to) { const m = MIP.zmem; z.zoom = m.from; MIP.zmem = {from: m.to, to: m.from}; }      // going back: the earlier value, exactly
       else { const sx1 = layout('axial').sx, z0 = z.zoom; if (sx1 > 0) z.zoom = Math.min(ZMAX, Math.max(ZMIN, z.zoom * sx0 / sx1)); MIP.zmem = {from: z0, to: z.zoom}; }
     }
     draw(); return MIP.on;
   }
   syncMipUi(); draw(); return MIP.on;
 }
"""

_JS_HOOKS = r"""
   setMip: setMip, slabPlan: slabPlan,
   mip: () => { const pl = slabPlan(S.cur[2], MIP.mm); return Object.assign({on: MIP.on, shown: mipShown('axial'), options: MIP_OPTIONS.slice(), spacing_mm: sp[2], images_total: nz,
     first_image_no: imageNo('axial', pl.k1), last_image_no: imageNo('axial', pl.k0), centre_image_no: imageNo('axial', S.cur[2]),
     cache_entries: MIP.cache.size, cache_max: MIP_CACHE_MAX, cache_bytes: MIP.cache.size * nxy * 2, cache_limit_bytes: MIP_CACHE_BYTES,
     computed: MIP.computed, hits: MIP.hits, reused: MIP.reused, info: ($('#mipinfo') || {}).textContent, help: ($('#miphelp') || {}).textContent, helpOpen: !!($('#miphelp') && !$('#miphelp').hidden),
     winText: ($('#winval') || {}).textContent}, pl); },
   mipHU: (k, mm) => { const pl = slabPlan(k, mm), m = slabMax(pl.k0, pl.k1); return Object.assign({w: G.axial.w, h: G.axial.h, b64: b64(new Uint8Array(m.buffer.slice(0)))}, pl); },
   shownGray: view => { view = view || AV(); const g = G[view], u = new Uint32Array(g.w * g.h), s = S.cur[g.ax]; if (mipShown(view)) fillMip(s, u); else fillSlice(view, s, u, true);
     const o = new Uint8Array(u.length), rgb = new Uint8Array(u.length); for (let i = 0; i < o.length; i++) { o[i] = u[i] & 255; rgb[i] = ((u[i] & 255) === ((u[i] >> 8) & 255) && (u[i] & 255) === ((u[i] >> 16) & 255)) ? 0 : 1; }
     return {w: g.w, h: g.h, b64: b64(o), coloured_pixels: rgb.reduce((a, x) => a + x, 0), mip: mipShown(view)}; },
"""

_JS_WIRING = r"""
   $('#mipon').onclick = () => setMip(true); $('#mipoff').onclick = () => setMip(false); $('#mipcentre').onclick = () => setMip(false);
   $$('[data-mip]').forEach(b => b.onclick = () => setMip(undefined, Number(b.dataset.mip)));
   $('#miphelpbtn').onclick = () => { $('#miphelp').hidden = !$('#miphelp').hidden; }; $('#miphelpclose').onclick = () => { $('#miphelp').hidden = true; };
   syncMipUi();
"""


def _build_js() -> str:
    js = v4.JS
    js = _once(js, " const ZMIN = 0.1, ZMAX = 16;\n", " const ZMIN = 0.1, ZMAX = 16;\n" + _JS_STATE
               .replace("__MIP_DEFAULT__", str(MIP_DEFAULT_MM)).replace("__MIP_OPTIONS__", json.dumps(list(MIP_OPTIONS_MM)))
               .replace("__MIP_CACHE_BYTES__", str(MIP_CACHE_BYTES)))
    # the one place a pane is filled before drawing: a MIP slab instead of the single image, without an outline
    js = _once(js, "     fillSlice(view, s, b.u, true);\n", "     if (mipShown(view)) fillMip(s, b.u); else fillSlice(view, s, b.u, true);\n")
    # read-out under the scan, and the slab description, refreshed on every paint
    js = _once(js, "(ov ? '  ·  outline: ' + ov.label : '');\n }\n",
               "(ov ? '  ·  outline: ' + ov.label : '');\n"
               "   if (mipShown(av)) { const pl = slabPlan(sa, MIP.mm), m = slabMax(pl.k0, pl.k1), cr = ga.cr(S.cur);\n"
               "     $('#loc').textContent = $('#loc').textContent.replace(hu + ' HU at the cross-hair', 'MIP: brightest in slab ' + m[cr[0] + ga.w * cr[1]] + ' HU at the cross-hair (centre image ' + hu + ' HU)'); }\n"
               "   syncMipUi();\n }\n")
    # another view or 3 planes: MIP is axial only, so it is left first
    js = _once(js, " function setView(v) {\n", " function setView(v) {\n   if (v !== 'axial' && MIP.on) { MIP.on = false; MIP.cache.clear(); }\n")
    # no outline can be switched on while the projection is shown
    js = _once(js, "     if (!V.ready || key === V.active) return;\n", "     if (!V.ready || key === V.active) return;\n     if (key && MIP.on) return;\n")
    # "Show in scan" goes back to the single image first, then shows the outline there
    js = _once(js, "     const k = tr.dataset.jump, t = META.targets[k];\n     if (!t) return;\n",
               "     const k = tr.dataset.jump, t = META.targets[k];\n     if (!t) return;\n     if (MIP.on) setMip(false);\n")
    js = _once(js, " Object.assign(V, {\n", " Object.assign(V, {\n" + _JS_HOOKS.lstrip("\n"))
    js = _once(js, "   $('#z11').onclick = () => oneToOne();\n", "   $('#z11').onclick = () => oneToOne();\n" + _JS_WIRING.lstrip("\n"))
    js = _once(js, "V = window.__viewer = {kind: 'canvas2d',", "V = window.__viewer = {kind: 'canvas2d', mipBuild: true,")
    return js


JS = _build_js()


def viewer_html(agentonly: bool) -> str:
    return VIEWER_HTML.replace("__AO__", " agentonly" if agentonly else "")


def viewer_script(meta: dict) -> str:
    return "<script>\n" + JS.replace("__META__", json.dumps(meta)) + "\n</script>"


def standalone_page(title: str, arr: np.ndarray, affine: np.ndarray, overlays: dict | None = None, targets: dict | None = None) -> str:
    """A minimal page with ONLY the v5 viewer (no checklist): used for the MIP tests on synthetic volumes."""
    page = v4.standalone_page(title, arr, affine, overlays, targets)
    a = page.index('<div class="viewer">')
    b = page.index("</div>\n<script>\n", a)
    page = page[:a] + viewer_html(False).rstrip("\n") + page[b:]
    page = _once(page, v4.CSS + "</style>", CSS + "</style>")
    a = page.rindex("<script>")
    m0 = page.index("const META = ", a) + len("const META = ")
    meta = json.loads(page[m0:page.index("\n", m0)].rstrip().rstrip(";"))
    meta["build"] = VIEWER_BUILD
    return page[:a] + viewer_script(meta) + "\n</body></html>"
