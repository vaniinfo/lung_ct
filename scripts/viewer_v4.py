"""2D canvas reviewer page, build "checklist_viewer_v4_canvas2d" (build_viewer_page.py --viewer v4).

Why: the WebGL viewer (NiiVue 0.69.0) rebuilds full-size 3D textures on every refresh; on an integrated graphics
adapter the graphics process grew by gigabytes per window change and died (docs/STATUS.md, 2026-11-22). This page
uses no WebGL and no viewer library: it draws ONE 2D slice at a time on an ordinary canvas from the decoded scan.

Unchanged: one standalone HTML for file://, the v2 layout, answers, measurements, grading form grades_v2, staging, CSV.
No cropping beyond the existing body crop, no resampling. The embedded scan is the same prepared display array as in
every earlier page (int16, values clipped to -1024..3071 HU before embedding: a pre-existing step, see CLIP_NOTE).

Geometry (the main risk), decided here and nowhere else:
  * At build time the voxel array is reordered to canonical RAS+ index order (axis 0 -> patient Right, axis 1 ->
    Anterior, axis 2 -> Superior) with the orientation transform derived from the affine by nibabel. This is a
    permutation / flip of voxels: every value is kept, none is interpolated. Overlays get the same transform.
  * Voxel spacing comes from the affine's column norms. The canonical affine and its inverse are embedded and used for
    physical position read-out and for "Show in scan" (physical mm -> slice).
  * Oblique acquisitions are NOT resliced (that would resample). `obliquity_deg` is measured; above MAX_OBLIQUE_DEG the
    build stops with an explicit error. Nothing is ever assumed to be axis-aligned without this check.
  * "3 planes" shows the three views side by side on ONE cross-hair position; each pane uses the same convention.
  * Display convention (radiological), fixed in the page and printed on every edge of the image:
      axial     screen left = patient Right, top = Anterior
      coronal   screen left = patient Right, top = Superior
      sagittal  screen left = Anterior,      top = Superior
    Pixels are drawn with the physical aspect ratio (column spacing : row spacing).
"""
from __future__ import annotations

import base64
import hashlib
import json
import zlib

import nibabel as nib
import numpy as np

import viewer_v3 as v3

VIEWER_BUILD = "checklist_viewer_v4_canvas2d"
MAX_OBLIQUE_DEG = 0.5
HASH_CHUNK = 8 * 1024 * 1024
CLIP_NOTE = ("Voxel values are the prepared display array used by every viewer version: the original HU clipped to "
             "-1024..3071 and stored as int16. Values outside that range (for example dense metal) are shown at the limit.")


class ObliqueOrientationError(ValueError):
    pass


def obliquity_deg(affine: np.ndarray) -> float:
    """Largest angle (degrees) between a voxel axis and its nearest patient axis. 0 = axis-aligned acquisition."""
    r = np.asarray(affine, float)[:3, :3]
    r = r / np.linalg.norm(r, axis=0)
    return float(np.degrees(np.arccos(np.clip(np.abs(r).max(axis=0), -1.0, 1.0))).max())


def canonical_transform(affine: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray, str, float]:
    """(orientation transform to RAS+, canonical affine, original axis codes, obliquity in degrees).

    Raises ObliqueOrientationError above MAX_OBLIQUE_DEG: an oblique volume would need reslicing (resampling), which this
    viewer does not do, so it is refused instead of being shown as if it were axis-aligned."""
    affine = np.asarray(affine, float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ObliqueOrientationError("the affine is not a usable 4 x 4 orientation matrix (not finite, or singular)")
    ang = obliquity_deg(affine)
    if ang > MAX_OBLIQUE_DEG:
        raise ObliqueOrientationError(f"oblique orientation ({ang:.2f} degrees from the patient axes) is not supported by the "
                                      f"2D viewer: it would have to be resliced. Limit {MAX_OBLIQUE_DEG} degrees.")
    ornt = nib.orientations.io_orientation(affine)
    if not np.isfinite(ornt).all() or sorted(int(a) for a in ornt[:, 0]) != [0, 1, 2]:
        raise ObliqueOrientationError("the affine does not map the three voxel axes to three different patient axes")
    can = affine @ nib.orientations.inv_ornt_aff(ornt, shape)
    codes = "".join(nib.aff2axcodes(affine))
    if nib.aff2axcodes(can) != ("R", "A", "S"):
        raise ObliqueOrientationError("canonical affine is not RAS+")
    return ornt, can, codes, ang


def to_canonical(arr: np.ndarray, ornt: np.ndarray) -> np.ndarray:
    """Reorder a voxel array to RAS+ index order. A permutation / flip only: same values, same count."""
    out = nib.orientations.apply_orientation(arr, ornt)
    assert out.size == arr.size
    return out


def ct_payload(arr_can: np.ndarray, can_affine: np.ndarray, orig_codes: str, ang: float) -> tuple[str, dict]:
    """Canonical int16 volume -> embedded blocks + meta (lossless coding of viewer v3, per-chunk SHA-256)."""
    stream, meta = v3.encode_ct_stream(np.ascontiguousarray(arr_can))
    gz = zlib.compress(stream, 6, wbits=31)
    html, n_parts = v3._blocks("ct", gz)
    raw = np.ascontiguousarray(arr_can).astype("<i2").tobytes(order="F")
    sp = np.linalg.norm(can_affine[:3, :3], axis=0)
    meta.update(gz_bytes=len(gz), parts=n_parts, raw_bytes=len(raw), min=int(arr_can.min()), max=int(arr_can.max()),
                hash_chunk=HASH_CHUNK,
                chunk_sha256=[hashlib.sha256(raw[i:i + HASH_CHUNK]).hexdigest() for i in range(0, len(raw), HASH_CHUNK)],
                spacing=[float(x) for x in sp], affine=[[float(x) for x in row] for row in can_affine],
                inv_affine=[[float(x) for x in row] for row in np.linalg.inv(can_affine)],
                index_order="RAS+ (axis 0 to patient Right, axis 1 to Anterior, axis 2 to Superior)",
                original_axis_codes=orig_codes, obliquity_deg=round(ang, 4), clip_note=CLIP_NOTE)
    return html, meta


def overlay_payload(key: str, mask_can: np.ndarray) -> tuple[str, dict]:
    """Canonical 0/1 mask -> its bounding box only (same grid, integer offset): exact, and small once decoded."""
    m = np.asarray(mask_can) > 0
    idx = np.argwhere(m)
    lo, hi = idx.min(0), idx.max(0) + 1
    sub = np.ascontiguousarray(m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.uint8))
    back = np.zeros(m.shape, bool)
    back[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = sub > 0
    assert np.array_equal(back, m), "bounding-box crop of the overlay is not exact"
    raw = sub.tobytes(order="F")
    gz = zlib.compress(raw, 9, wbits=31)
    html, n_parts = v3._blocks(key, gz)
    return html, {"codec": "bbox-raw-gzip-v1", "offset": [int(x) for x in lo], "dims": [int(x) for x in sub.shape],
                  "raw_bytes": len(raw), "gz_bytes": len(gz), "parts": n_parts, "voxels_on": int(m.sum()),
                  "sha256": hashlib.sha256(raw).hexdigest()}


CSS = """
.cwrap{flex:1;min-height:0;position:relative;background:#000;overflow:hidden}
#gl{position:absolute;left:0;top:0;width:100%;height:100%;display:block;flex:none;touch-action:none;cursor:crosshair}
#vstat{position:absolute;left:50%;top:46%;transform:translate(-50%,-50%);z-index:20;background:rgba(22,26,34,.94);border:1px solid #3b4558;
 border-radius:10px;padding:14px 18px;min-width:300px;max-width:70%;text-align:center;color:#e6e6e6;font-size:14px}
#vstat.err{border-color:#b4523f;color:#ffd9d1;text-align:left}
#vstat .pbar{height:6px;background:#2a3140;border-radius:3px;margin-top:9px;overflow:hidden}
#vstat .pbar i{display:block;height:100%;width:0;background:#2f6feb}
#vbanner{display:none;padding:7px 10px;background:#4a2a22;color:#ffd9d1;border-bottom:1px solid #7a3b2e;font-size:12.5px}
#vbanner button{margin-left:8px}
.ov.busy{opacity:1;border-color:#2f6feb}.ov.busy::after{content:' loading...';color:#9fc2ff;font-size:11px}
.ov.fail{border-color:#b4523f}
#loc{white-space:normal;font-size:14px !important;line-height:1.5;padding:7px 10px !important}
/* larger text for readers (reviewer request 2026-11-22): the right-hand panel is scaled as a whole, default 120 %, adjustable with A- / A+ */
/* adjustable split between the scan and the panel: drag the bar between them (25-75 %), remembered in this browser */
.page{grid-template-columns:var(--split,50%) 1fr !important}
#splitbar{position:fixed;top:0;bottom:0;left:calc(var(--split,50%) - 5px);width:10px;cursor:col-resize;z-index:50;background:transparent}
#splitbar::after{content:'';position:absolute;top:0;bottom:0;left:4px;width:2px;background:#3b4558}
#splitbar:hover::after,#splitbar.drag::after{background:#2f6feb;width:4px;left:3px}
body.splitting{user-select:none;cursor:col-resize}
.side{zoom:var(--tz,1.2)}
.bar{font-size:15px}.bar b{font-size:14px !important}.bar button{font-size:15px;padding:6px 12px}.ov{font-size:15px;padding:5px 10px !important}
.sw{width:13px !important;height:13px !important}
#tzval{color:#9aa4b2;font-size:13px;min-width:44px;text-align:center}
#measdet>summary{cursor:pointer;padding:2px 0}#measdet[open]>summary{margin-bottom:6px}
"""

VIEWER_HTML = """<div class="viewer">
 <div class="bar"><b>Window</b>
  <button data-w="-1350,150" class="on">Lung</button><button data-w="-160,240">Soft tissue</button><button data-w="-500,1300">Bone</button>
  <b style="margin-left:10px">View</b>
  <button data-v="axial" class="on">Axial</button><button data-v="coronal">Coronal</button><button data-v="sagittal">Sagittal</button><button data-v="three" title="axial, coronal and sagittal side by side, linked by the cross-hair">3 planes</button>
  <b style="margin-left:10px">Zoom</b><button id="zout" title="zoom out">&minus;</button><button id="zin" title="zoom in">+</button><button id="zfit" title="fit the image to the window">Fit</button><button id="z11" title="one image pixel = one screen pixel (no scaling across the image)">1:1 pixels</button>
  <button id="smooth" class="on" title="on: scaled image is smoothed. off: every image pixel is drawn as a sharp square, for pixel inspection">Smooth</button>
  <button id="xhair" class="on" style="margin-left:10px" title="show or hide the yellow cross-hair">Cross-hair</button>
  <b style="margin-left:10px">Text size</b><button id="tzout" title="smaller text in the panel on the right">A&minus;</button><span id="tzval"></span><button id="tzin" title="larger text in the panel on the right">A+</button>
 </div>
 <div class="bar__AO__" id="ovbar"><b>Outlines (off to start; one at a time)</b></div>
 <div id="vbanner"></div>
 <div class="cwrap"><canvas id="gl"></canvas><div id="vstat">Loading the scan&hellip;<div class="pbar"><i></i></div>
  <div class="muted" style="margin-top:6px;color:#9aa4b2">You can already use the grading form on the right.</div></div></div>
 <div id="loc">Wheel or arrow keys: next / previous image &middot; click: move cross-hair &middot; drag: move the image &middot; right-drag or Shift+drag: brightness / contrast &middot; Ctrl+wheel: zoom</div>
</div>
"""

EARLY_JS = v3.EARLY_JS + r"""
(function () {   // text size of the right-hand panel: default 120 %, remembered in this browser
  const KEY = 'lungct_text_zoom'; let z = 1.2;
  try { const s = parseFloat(localStorage.getItem(KEY)); if (s >= 1 && s <= 1.8) z = s; } catch (e) {}
  function apply() { document.documentElement.style.setProperty('--tz', z); const v = document.getElementById('tzval'); if (v) v.textContent = Math.round(z * 100) + '%'; }
  function bump(d) { z = Math.min(1.8, Math.max(1, Math.round((z + d) * 10) / 10)); try { localStorage.setItem(KEY, z); } catch (e) {} apply(); }
  apply();
  const a = document.getElementById('tzin'), b = document.getElementById('tzout');
  if (a) a.onclick = () => bump(0.1); if (b) b.onclick = () => bump(-0.1);
  window.__textZoom = () => z;
})();
(function () {   // adjustable split: a bar between the scan and the right-hand panel
  const page = document.querySelector('.page'); if (!page || !document.querySelector('.side')) return;
  const KEY = 'lungct_split_pct'; let pct = 50;
  try { const s = parseFloat(localStorage.getItem(KEY)); if (s >= 25 && s <= 75) pct = s; } catch (e) {}
  const bar = document.createElement('div'); bar.id = 'splitbar'; bar.title = 'Drag to make the scan or the panel wider. Double-click for half and half.';
  document.body.appendChild(bar);
  function apply(save) { pct = Math.min(75, Math.max(25, pct)); document.documentElement.style.setProperty('--split', pct + '%'); if (save) { try { localStorage.setItem(KEY, pct.toFixed(1)); } catch (e) {} } }
  apply(false);
  let on = false;
  bar.addEventListener('pointerdown', e => { on = true; bar.setPointerCapture(e.pointerId); bar.classList.add('drag'); document.body.classList.add('splitting'); e.preventDefault(); });
  bar.addEventListener('pointermove', e => { if (!on) return; pct = 100 * e.clientX / Math.max(1, window.innerWidth); apply(false); });
  const end = () => { if (!on) return; on = false; bar.classList.remove('drag'); document.body.classList.remove('splitting'); apply(true); };
  bar.addEventListener('pointerup', end); bar.addEventListener('pointercancel', end);
  bar.addEventListener('dblclick', () => { pct = 50; apply(true); });
  window.__split = () => pct;
})();
"""

JS = r"""
const META = __META__;
(function () {
 const T0 = performance.now();
 const $ = s => document.querySelector(s), $$ = s => Array.from(document.querySelectorAll(s));
 const V = window.__viewer = {kind: 'canvas2d', build: META.build, ready: false, active: null, timings: {}, verified: null, log: []};
 const stat = $('#vstat'), banner = $('#vbanner');
 function status(html, frac, isErr) {
   if (html === null) { stat.style.display = 'none'; return; }
   stat.style.display = ''; stat.className = isErr ? 'err' : '';
   stat.innerHTML = html + (frac === undefined || isErr ? '' : '<div class="pbar"><i style="width:' + Math.round(100 * frac) + '%"></i></div>');
 }
 function warn(msg, withReload) {
   V.log.push(msg); window.__viewerErrors.push(msg);
   banner.style.display = 'block'; banner.innerHTML = '';
   banner.appendChild(document.createTextNode(msg + ' '));
   if (withReload) { const r = document.createElement('button'); r.textContent = 'Reload the page'; r.onclick = () => location.reload(); banner.appendChild(r); }
   const x = document.createElement('button'); x.textContent = 'Dismiss'; x.onclick = () => { banner.style.display = 'none'; }; banner.appendChild(x);
 }
 const SAFE = ' Your grades are saved in this browser and are not affected.';
 window.addEventListener('error', e => warn('The viewer reported an error: ' + (e.message || 'unknown') + '.' + SAFE, false));
 window.addEventListener('unhandledrejection', e => { warn('The viewer reported an error: ' + ((e.reason && e.reason.message) || e.reason || 'unknown') + '.' + SAFE, false); });
 const tick = () => new Promise(r => setTimeout(r, 0));
 const HASH = (new URLSearchParams(location.search)).get('hash') || 'chunks';     // chunks (default) | whole | none: for memory measurements

 // ---------- decoding (lossless; nothing is resampled) ----------
 function b64ToBytes(txt) { if (typeof Uint8Array.fromBase64 === 'function') return Uint8Array.fromBase64(txt);
   const bin = atob(txt), n = bin.length, u = new Uint8Array(n); for (let i = 0; i < n; i++) u[i] = bin.charCodeAt(i); return u; }
 function gzStream(vol, consume, onPart) {
   if (typeof DecompressionStream === 'undefined') throw new Error('this browser is too old (no DecompressionStream); use a current Chrome or Edge');
   const els = $$('script.vp[data-vol="' + vol + '"]').sort((a, b) => a.dataset.part - b.dataset.part);
   if (!els.length) throw new Error('the data for "' + vol + '" is not in this file');
   let k = 0;
   const src = new ReadableStream({ pull(c) {
     if (k >= els.length) { c.close(); return; }
     const el = els[k++]; let bytes;
     try { bytes = b64ToBytes(el.textContent); } catch (e) { c.error(new Error('the embedded data is damaged (base64)')); return; }
     if (consume) el.remove();
     if (onPart) onPart(k / els.length);
     c.enqueue(bytes);
   } }, {highWaterMark: 1});
   return src.pipeThrough(new DecompressionStream('gzip')).getReader();
 }
 const hex = buf => Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('');
 const canHash = () => !!(window.crypto && crypto.subtle);
 const C = META.ct, nx = C.dims[0], ny = C.dims[1], nz = C.dims[2], nxy = nx * ny, NV = C.voxels;
 const sp = C.spacing, VMIN = C.min, VMAX = C.max;
 let ct = null;

 async function decodeCT(onProg) {
   const n = NV, off = C.offset;
   const buf = new ArrayBuffer(n * 2), out = new Int16Array(buf);
   const rd = gzStream('ct', true, onProg);
   let i = 0, st = 0, lo = 0, acc = 0, w = 1, got = 0, lastYield = performance.now();
   for (;;) {
     const {done, value} = await rd.read();
     if (done) break;
     got += value.length;
     for (let j = 0, L = value.length; j < L; j++) {
       const b = value[j]; let zz = -1;
       if (st === 0) { if (b < 255) zz = b; else st = 1; } else if (st === 1) { lo = b; st = 2; } else { zz = lo | (b << 8); st = 0; }
       if (zz < 0) continue;
       if (i >= n) throw new Error('the embedded scan holds more voxels than its header says');
       let p;
       if (i >= nxy) p = ((out[i - 1] + out[i - nx] + out[i - nxy] + 3 * off) / 3) | 0;
       else p = (((i >= 1 ? out[i - 1] + off : 0) + (i >= nx ? out[i - nx] + off : 0)) / 3) | 0;
       const u = p + ((zz >>> 1) ^ -(zz & 1));
       out[i++] = u - off;
       acc = (acc + Math.imul(u + 1, w)) >>> 0; if (++w > 65521) w = 1;
     }
     if (performance.now() - lastYield > 60) { await tick(); lastYield = performance.now(); }
   }
   if (i !== n || got !== C.stream_bytes) throw new Error('the embedded scan is incomplete (' + i + ' of ' + n + ' voxels)');
   if (acc !== C.checksum) throw new Error('the embedded scan failed its checksum; the file may be damaged');
   let sha = 'not checked';
   if (HASH !== 'none' && canHash()) {
     if (HASH === 'whole') {                 // one digest over the whole volume: the browser may copy the buffer for this
       if (hex(await crypto.subtle.digest('SHA-256', new Uint8Array(buf))) !== C.sha256) throw new Error('the embedded scan failed its SHA-256 check; the file may be damaged');
     } else {                                // default: SHA-256 per 8 MB piece, so no second copy of the volume is needed
       const cs = C.hash_chunk;
       for (let o = 0, k = 0; o < buf.byteLength; o += cs, k++) {
         const h = hex(await crypto.subtle.digest('SHA-256', new Uint8Array(buf, o, Math.min(cs, buf.byteLength - o))));
         if (h !== C.chunk_sha256[k]) throw new Error('the embedded scan failed its SHA-256 check (piece ' + (k + 1) + '); the file may be damaged');
       }
     }
     sha = true;
   } else if (HASH !== 'none') sha = 'not available in this browser';
   V.verified = {checksum: true, sha256: sha, mode: HASH};
   return out;
 }
 async function decodeOverlay(key) {
   const m = META.vols[key], out = new Uint8Array(m.raw_bytes);
   const rd = gzStream(key, false, null);
   let o = 0;
   for (;;) {
     const {done, value} = await rd.read();
     if (done) break;
     if (o + value.length > out.length) throw new Error('outline data is longer than its header says');
     out.set(value, o); o += value.length;
   }
   if (o !== m.raw_bytes) throw new Error('outline data is incomplete');
   if (canHash() && hex(await crypto.subtle.digest('SHA-256', out)) !== m.sha256) throw new Error('outline data failed its SHA-256 check');
   return out;
 }

 // ---------- geometry: canonical RAS+ voxel index (i to Right, j to Anterior, k to Superior) ----------
 // Display convention (radiological). c = column from the left, r = row from the top, s = image index along the view axis.
 //   axial:    i = nx-1-c, j = ny-1-r, k = s      left = Right, top = Anterior
 //   coronal:  i = nx-1-c, j = s,      k = nz-1-r left = Right, top = Superior
 //   sagittal: i = s,      j = ny-1-c, k = nz-1-r left = Anterior, top = Superior
 const G = {
   axial:    {w: nx, h: ny, pw: sp[0], ph: sp[1], n: nz, ax: 2, edges: ['R', 'L', 'A', 'P'], base: s => (nx - 1) + nx * ((ny - 1) + ny * s), cs: -1, rs: -nx,
              vox: (s, c, r) => [nx - 1 - c, ny - 1 - r, s], cr: v => [nx - 1 - v[0], ny - 1 - v[1]]},
   coronal:  {w: nx, h: nz, pw: sp[0], ph: sp[2], n: ny, ax: 1, edges: ['R', 'L', 'S', 'I'], base: s => (nx - 1) + nx * (s + ny * (nz - 1)), cs: -1, rs: -nxy,
              vox: (s, c, r) => [nx - 1 - c, s, nz - 1 - r], cr: v => [nx - 1 - v[0], nz - 1 - v[2]]},
   sagittal: {w: ny, h: nz, pw: sp[1], ph: sp[2], n: nx, ax: 0, edges: ['A', 'P', 'S', 'I'], base: s => s + nx * ((ny - 1) + ny * (nz - 1)), cs: -nx, rs: -nxy,
              vox: (s, c, r) => [s, ny - 1 - c, nz - 1 - r], cr: v => [ny - 1 - v[1], nz - 1 - v[2]]}
 };
 const A = C.affine, AI = C.inv_affine;
 const voxToMm = v => [0, 1, 2].map(r => A[r][0] * v[0] + A[r][1] * v[1] + A[r][2] * v[2] + A[r][3]);
 const mmToVoxF = m => [0, 1, 2].map(r => AI[r][0] * m[0] + AI[r][1] * m[1] + AI[r][2] * m[2] + AI[r][3]);
 const clampVox = v => [Math.min(nx - 1, Math.max(0, Math.round(v[0]))), Math.min(ny - 1, Math.max(0, Math.round(v[1]))), Math.min(nz - 1, Math.max(0, Math.round(v[2])))];
 // image number shown to the reader: 1 = most superior (axial), most anterior (coronal), patient's right (sagittal)
 const imageNo = (view, s) => view === 'sagittal' ? G[view].n - s : G[view].n - s;

 // ---------- state and reusable buffers (nothing of volume size is allocated after loading) ----------
 // zoom is RELATIVE TO FIT (1 = the image fills the window). pxScale() gives screen pixels per image pixel; "1:1 pixels" sets it to 1.
 // mode 'single' shows S.view; mode 'three' shows axial (left), coronal (top right) and sagittal (bottom right), all on the same
 // cross-hair position S.cur. Each view keeps its own zoom and pan. S.focus = the pane last used in three-plane mode.
 const S = {mode: 'single', view: 'axial', focus: 'axial', cur: [nx >> 1, ny >> 1, nz >> 1], win: [-1350, 150], xhair: true, smooth: true,
            zp: {axial: {zoom: 1, pan: [0, 0]}, coronal: {zoom: 1, pan: [0, 0]}, sagittal: {zoom: 1, pan: [0, 0]}}};
 const AV = () => S.mode === 'three' ? S.focus : S.view;          // the view that buttons, keys and the read-out refer to
 const ZMIN = 0.1, ZMAX = 16;
 const lut = new Uint8Array(VMAX - VMIN + 1);        // value -> grey level, refilled on a window change
 const grayPx = new Uint32Array(256), blendPx = new Uint32Array(256);
 for (let g = 0; g < 256; g++) grayPx[g] = (0xFF000000 | (g << 16) | (g << 8) | g) >>> 0;
 function setLut() { const lo = S.win[0], k = 255 / Math.max(1, S.win[1] - S.win[0]); for (let v = VMIN; v <= VMAX; v++) { const g = (v - lo) * k; lut[v - VMIN] = g <= 0 ? 0 : g >= 255 ? 255 : g + 0.5 | 0; } }
 const ALPHA = 0.55;
 function setBlend(rgb) { for (let g = 0; g < 256; g++) { const r = g * (1 - ALPHA) + rgb[0] * ALPHA + 0.5 | 0, gg = g * (1 - ALPHA) + rgb[1] * ALPHA + 0.5 | 0, b = g * (1 - ALPHA) + rgb[2] * ALPHA + 0.5 | 0; blendPx[g] = (0xFF000000 | (b << 16) | (gg << 8) | r) >>> 0; } }
 let ov = null;                                      // the ONE decoded outline: {key, data, o:[i,j,k], d:[mx,my,mz]}
 const canvas = $('#gl'), ctx = canvas.getContext('2d');
 const ocs = {};                                     // one small off-screen canvas per view, created once and reused
 function offscreen(view) { const g = G[view]; if (!ocs[view]) { const c = document.createElement('canvas'); c.width = g.w; c.height = g.h; ocs[view] = {c: c, x: c.getContext('2d')}; } return ocs[view]; }
 const octx = document.createElement('canvas').getContext('2d');
 const img = {};                                     // one ImageData per view, created once and reused
 function sliceBuf(view) { const g = G[view]; if (!img[view]) { const d = new ImageData(g.w, g.h); img[view] = {d: d, u: new Uint32Array(d.data.buffer)}; } return img[view]; }
 function fillSlice(view, s, u32, withOverlay) {     // the only place voxels become pixels
   const g = G[view], w = g.w, h = g.h, cs = g.cs, rs = g.rs, b0 = g.base(s), min = VMIN;
   for (let r = 0, p = 0; r < h; r++) { let q = b0 + r * rs; for (let c = 0; c < w; c++, p++, q += cs) u32[p] = grayPx[lut[ct[q] - min]]; }
   if (withOverlay && ov) {
     const o = ov.o, d = ov.d, m = ov.data, mx = d[0], mxy = d[0] * d[1];
     for (let r = 0, p = 0; r < h; r++) for (let c = 0; c < w; c++, p++) {
       const v = g.vox(s, c, r), a = v[0] - o[0], b = v[1] - o[1], e = v[2] - o[2];
       if (a >= 0 && b >= 0 && e >= 0 && a < d[0] && b < d[1] && e < d[2] && m[a + mx * b + mxy * e]) u32[p] = blendPx[u32[p] & 255];
     }
   }
 }
 function panes() {                                  // rectangles (canvas pixels) of the views on screen
   const W = canvas.width, H = canvas.height;
   if (S.mode !== 'three') return [{view: S.view, x: 0, y: 0, w: W, h: H}];
   const half = Math.floor(W / 2), hh = Math.floor(H / 2), gap = 2;
   return [{view: 'axial', x: 0, y: 0, w: half - gap, h: H}, {view: 'coronal', x: half, y: 0, w: W - half, h: hh - gap}, {view: 'sagittal', x: half, y: hh, w: W - half, h: H - hh}];
 }
 function rectOf(view) { return panes().find(p => p.view === view) || {view: view, x: 0, y: 0, w: canvas.width, h: canvas.height}; }
 function paneAt(cx, cy) { return panes().find(p => cx >= p.x && cx < p.x + p.w && cy >= p.y && cy < p.y + p.h) || null; }
 function layout(view, R) {                          // where a slice sits inside its pane (canvas pixels), physical aspect ratio
   view = view || AV(); R = R || rectOf(view);
   const g = G[view], z = S.zp[view], pw = g.w * g.pw, ph = g.h * g.ph;
   const fit = Math.min(R.w / pw, R.h / ph) * 0.94, sc = fit * z.zoom;
   const dw = pw * sc, dh = ph * sc;
   return {x: R.x + (R.w - dw) / 2 + z.pan[0], y: R.y + (R.h - dh) / 2 + z.pan[1], w: dw, h: dh, sx: dw / g.w, sy: dh / g.h, g: g, R: R, view: view};
 }
 let raf = 0;
 function draw() { if (!raf) raf = requestAnimationFrame(() => { raf = 0; try { paint(); } catch (e) { warn('The scan could not be drawn: ' + (e && e.message || e) + '.' + SAFE, true); } }); }
 function paint() {
   if (!ct) return;
   const dpr = window.devicePixelRatio || 1, cw = Math.max(1, Math.round(canvas.clientWidth * dpr)), ch = Math.max(1, Math.round(canvas.clientHeight * dpr));
   if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
   ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = S.mode === 'three' ? '#2a2f3a' : '#000'; ctx.fillRect(0, 0, cw, ch);
   const three = S.mode === 'three';
   for (const R of panes()) {
     const view = R.view, g = G[view], s = S.cur[g.ax], b = sliceBuf(view), o = offscreen(view);
     fillSlice(view, s, b.u, true);
     o.x.putImageData(b.d, 0, 0);
     const L = layout(view, R);
     ctx.save(); ctx.beginPath(); ctx.rect(R.x, R.y, R.w, R.h); ctx.clip();
     ctx.fillStyle = '#000'; ctx.fillRect(R.x, R.y, R.w, R.h);
     ctx.imageSmoothingEnabled = S.smooth; if (S.smooth) ctx.imageSmoothingQuality = 'high';
     ctx.drawImage(o.c, 0, 0, g.w, g.h, L.x, L.y, L.w, L.h);
     const cr = g.cr(S.cur), px = L.x + (cr[0] + 0.5) * L.sx, py = L.y + (cr[1] + 0.5) * L.sy;
     if (S.xhair) { ctx.strokeStyle = 'rgba(255,217,26,0.9)'; ctx.lineWidth = Math.max(1, dpr); ctx.beginPath(); ctx.moveTo(L.x, py); ctx.lineTo(L.x + L.w, py); ctx.moveTo(px, L.y); ctx.lineTo(px, L.y + L.h); ctx.stroke(); }
     ctx.fillStyle = '#9aa4b2'; ctx.font = 'bold ' + ((three ? 15 : 18) * dpr) + 'px Segoe UI, Arial, sans-serif'; ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
     const e = g.edges, m = (three ? 12 : 15) * dpr;
     ctx.fillText(e[0], R.x + m, R.y + R.h / 2); ctx.fillText(e[1], R.x + R.w - m, R.y + R.h / 2); ctx.fillText(e[2], R.x + R.w / 2, R.y + m); ctx.fillText(e[3], R.x + R.w / 2, R.y + R.h - m);
     if (three) {
       ctx.textAlign = 'left'; ctx.fillStyle = view === S.focus ? '#9fc2ff' : '#9aa4b2'; ctx.font = (13 * dpr) + 'px Segoe UI, Arial, sans-serif';
       ctx.fillText(view[0].toUpperCase() + view.slice(1) + ' ' + imageNo(view, s) + ' / ' + g.n, R.x + 8 * dpr, R.y + 13 * dpr);
       if (view === S.focus) { ctx.strokeStyle = '#2f6feb'; ctx.lineWidth = 2 * dpr; ctx.strokeRect(R.x + dpr, R.y + dpr, R.w - 2 * dpr, R.h - 2 * dpr); }
     }
     ctx.restore();
   }
   const av = AV(), ga = G[av], sa = S.cur[ga.ax], La = layout(av);
   const mm = voxToMm(S.cur), hu = ct[S.cur[0] + nx * (S.cur[1] + ny * S.cur[2])];
   $('#loc').textContent = (three ? '3 planes  ·  ' : '') + av[0].toUpperCase() + av.slice(1) + ' image ' + imageNo(av, sa) + ' of ' + ga.n + '  ·  ' + hu + ' HU at the cross-hair  ·  window ' +
     Math.round(S.win[0]) + ' to ' + Math.round(S.win[1]) + '  ·  zoom ' + Math.round(S.zp[av].zoom * 100) + '% of fit (1 image pixel = ' + pxText(La) + ' screen pixels' + (S.smooth ? '' : ', not smoothed') + ')  ·  position ' + mm.map(x => x.toFixed(1)).join(', ') + ' mm' + (ov ? '  ·  outline: ' + ov.label : '');
 }
 function pxText(L) { const a = L.sx.toFixed(2), b = L.sy.toFixed(2); return a === b ? a : a + ' x ' + b; }
 function resetZoom(view) { S.zp[view].zoom = 1; S.zp[view].pan = [0, 0]; }
 function oneToOne(view) {                           // one image column = one screen (device) pixel; rows follow the physical aspect ratio
   view = view || AV(); const L = layout(view), z = S.zp[view]; z.zoom = Math.min(ZMAX, Math.max(ZMIN, z.zoom / L.sx)); z.pan = [0, 0]; draw();
 }
 function setView(v) {
   if (v === 'three') { S.mode = 'three'; ['axial', 'coronal', 'sagittal'].forEach(resetZoom); }
   else if (G[v]) { S.mode = 'single'; S.view = v; S.focus = v; resetZoom(v); }
   else return;
   $$('[data-v]').forEach(x => x.classList.toggle('on', x.dataset.v === v)); draw();
 }
 function step(dir, view) { view = view || AV(); const ax = G[view].ax, n = G[view].n; S.cur[ax] = Math.min(n - 1, Math.max(0, S.cur[ax] + dir)); draw(); }
 function setWindow(lo, hi, fromPreset) { if (hi - lo < 1) hi = lo + 1; S.win = [lo, hi]; setLut(); if (!fromPreset) $$('[data-w]').forEach(x => x.classList.remove('on')); draw(); }
 function zoomBy(f, cx, cy, view) {                  // keep the point under the pointer in place
   view = view || AV();
   const L0 = layout(view), zs = S.zp[view], z = Math.min(ZMAX, Math.max(ZMIN, zs.zoom * f));
   if (cx === undefined) { cx = L0.R.x + L0.R.w / 2; cy = L0.R.y + L0.R.h / 2; }
   const fx = (cx - L0.x) / L0.w, fy = (cy - L0.y) / L0.h;
   zs.zoom = z; const L1 = layout(view);
   zs.pan[0] += cx - (L1.x + fx * L1.w); zs.pan[1] += cy - (L1.y + fy * L1.h); draw();
 }
 function goToMm(mm) { S.cur = clampVox(mmToVoxF(mm)); draw(); return S.cur.slice(); }
 function canvasXY(ev) { const rect = canvas.getBoundingClientRect(), k = canvas.width / Math.max(1, rect.width); return [(ev.clientX - rect.left) * k, (ev.clientY - rect.top) * k]; }
 function pick(ev) {                                 // pointer position -> {view of the pane, voxel on its shown slice or null}
   const xy = canvasXY(ev), R = paneAt(xy[0], xy[1]);
   if (!R) return null;
   const L = layout(R.view, R), g = L.g, c = Math.floor((xy[0] - L.x) / L.sx), r = Math.floor((xy[1] - L.y) / L.sy);
   return {view: R.view, vox: (c < 0 || r < 0 || c >= g.w || r >= g.h) ? null : g.vox(S.cur[g.ax], c, r)};
 }

 // ---------- outlines: one at a time, decoded on request, previous one released ----------
 let chain = Promise.resolve();
 function chip(k) { return $('.ov[data-k="' + k + '"]'); }
 function paintChips() { for (const o of META.overlays) { const c = chip(o.key); if (c) { c.classList.toggle('off', V.active !== o.key); c.classList.remove('busy'); } } }
 function hexRgb(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
 function setOverlay(key) {
   chain = chain.then(async () => {
     if (!V.ready || key === V.active) return;
     if (ov) { ov.data = null; ov = null; V.active = null; paintChips(); draw(); }      // release the previous decoded outline
     if (!key) return;
     const o = META.overlays.find(x => x.key === key), c = chip(key);
     if (c) { c.classList.add('busy'); c.classList.remove('fail'); }
     try {
       const m = META.vols[key], data = await decodeOverlay(key);
       setBlend(hexRgb(o.swatch));
       ov = {key: key, label: o.label, data: data, o: m.offset, d: m.dims}; V.active = key;
     } catch (e) {
       if (c) c.classList.add('fail');
       warn('The "' + o.label + '" outline could not be shown (' + (e && e.message || e) + '). The scan and the other outlines still work.' + SAFE, false);
     }
     paintChips(); draw();
   }).catch(e => { V.log.push('outline queue: ' + e); paintChips(); });
   return chain;
 }

 // ---------- test hooks (read-only views of exactly what is drawn) ----------
 function grayOf(view, s, withOverlay) { const g = G[view], u = new Uint32Array(g.w * g.h); fillSlice(view, s, u, !!withOverlay); return {g: g, u: u}; }
 const b64 = u8 => { let t = ''; for (let i = 0; i < u8.length; i += 32768) t += String.fromCharCode.apply(null, u8.subarray(i, i + 32768)); return btoa(t); };
 Object.assign(V, {
   setOverlay: setOverlay, idle: () => chain, setView: setView, step: step, setWindow: setWindow, zoomBy: zoomBy, goToMm: goToMm, draw: draw,
   state: () => { const av = AV(), rect = canvas.getBoundingClientRect(), k = rect.width / Math.max(1, canvas.width);
     return {view: av, mode: S.mode, focus: S.focus, cur: S.cur.slice(), win: S.win.slice(), zoom: S.zp[av].zoom, pan: S.zp[av].pan.slice(), xhair: S.xhair, smooth: S.smooth,
             pxScale: (() => { const L = layout(av); return [L.sx, L.sy]; })(), active: V.active, dims: [nx, ny, nz], spacing: sp.slice(),
             overlayBytes: ov ? ov.data.length : 0, imageNo: imageNo(av, S.cur[G[av].ax]), images: G[av].n,
             panes: panes().map(p => ({view: p.view, x: rect.left + p.x * k, y: rect.top + p.y * k, w: p.w * k, h: p.h * k}))}; },
   setCur: v => { S.cur = clampVox(v); draw(); },
   voxToMm: voxToMm, mmToVox: m => clampVox(mmToVoxF(m)), edges: v => G[v].edges.slice(), pixelMm: (view, s, c, r) => voxToMm(G[view].vox(s, c, r)),
   images: v => G[v].n, canvasRect: () => { const r = canvas.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; },
   sliceGray: (view, s) => { const r = grayOf(view, s, false), o = new Uint8Array(r.u.length); for (let i = 0; i < o.length; i++) o[i] = r.u[i] & 255; return {w: r.g.w, h: r.g.h, pw: r.g.pw, ph: r.g.ph, b64: b64(o)}; },
   sliceMask: (view, s) => { const a = grayOf(view, s, false), b = grayOf(view, s, true), o = new Uint8Array(a.u.length); for (let i = 0; i < o.length; i++) o[i] = a.u[i] !== b.u[i] ? 1 : 0; return {w: a.g.w, h: a.g.h, b64: b64(o)}; },
   sliceHU: (view, s) => { const g = G[view], o = new Int16Array(g.w * g.h), b0 = g.base(s); for (let r = 0, p = 0; r < g.h; r++) { let q = b0 + r * g.rs; for (let c = 0; c < g.w; c++, p++, q += g.cs) o[p] = ct[q]; } return {w: g.w, h: g.h, b64: b64(new Uint8Array(o.buffer))}; },
   screenOf: (v, view) => { const L = layout(view), cr = L.g.cr(v), rect = canvas.getBoundingClientRect(), k = rect.width / canvas.width; return [rect.left + (L.x + (cr[0] + 0.5) * L.sx) * k, rect.top + (L.y + (cr[1] + 0.5) * L.sy) * k]; },
   layout: view => { const L = layout(view), rect = canvas.getBoundingClientRect(), k = rect.width / canvas.width; return {x: rect.left + L.x * k, y: rect.top + L.y * k, w: L.w * k, h: L.h * k, cols: L.g.w, rows: L.g.h, pw: L.g.pw, ph: L.g.ph}; }
 });

 async function start() {
   if (!ctx || !octx) { status('<b>The scan viewer cannot start in this browser</b> (no 2D canvas).' + SAFE, undefined, true); return; }
   try {
     status('Loading the scan&hellip; decoding', 0);
     let last = 0;
     ct = await decodeCT(f => { if (f - last >= 0.02 || f === 1) { last = f; status('Loading the scan&hellip; ' + Math.round(100 * f) + '%', f); } });
     V.timings.decode_ms = Math.round(performance.now() - T0);
   } catch (e) {
     ct = null;
     status('<b>The scan could not be loaded:</b> ' + (e && e.message || e) + '.' + SAFE + '<br><br><button onclick="location.reload()">Reload the page</button>', undefined, true);
     V.log.push('ct: ' + e); return;
   }
   setLut();
   for (const o of META.overlays) {
     const el = document.createElement('span'); el.className = 'ov off'; el.dataset.k = o.key;
     el.innerHTML = '<span class="sw" style="background:' + o.swatch + '"></span>' + o.label;
     el.onclick = () => setOverlay(V.active === o.key ? null : o.key);
     $('#ovbar').appendChild(el);
   }
   $$('[data-w]').forEach(b => b.onclick = () => { const p = b.dataset.w.split(',').map(Number); setWindow(p[0], p[1], true); $$('[data-w]').forEach(x => x.classList.toggle('on', x === b)); });
   $$('[data-v]').forEach(b => b.onclick = () => setView(b.dataset.v));
   $('#zin').onclick = () => zoomBy(1.25); $('#zout').onclick = () => zoomBy(0.8);
   $('#zfit').onclick = () => { (S.mode === 'three' ? ['axial', 'coronal', 'sagittal'] : [S.view]).forEach(resetZoom); draw(); };
   $('#z11').onclick = () => oneToOne();
   $('#smooth').onclick = () => { S.smooth = !S.smooth; $('#smooth').classList.toggle('on', S.smooth); draw(); };
   $('#xhair').onclick = () => { S.xhair = !S.xhair; $('#xhair').classList.toggle('on', S.xhair); draw(); };
   // mouse / touchpad
   let wacc = 0, drag = null;
   canvas.addEventListener('wheel', e => {
     e.preventDefault();
     const xy = canvasXY(e), R = paneAt(xy[0], xy[1]), view = R ? R.view : AV();
     if (S.mode === 'three' && R) S.focus = view;   // the pane under the pointer is the one that scrolls or zooms
     if (e.ctrlKey) { zoomBy(Math.exp(-e.deltaY * 0.0025), xy[0], xy[1], view); return; }
     const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaMode === 2 ? e.deltaY * 100 : e.deltaY;
     let n = 0;
     if (Math.abs(dy) >= 50) { n = dy > 0 ? 1 : -1; wacc = 0; }        // a mouse-wheel notch: exactly one image
     else { wacc += dy; n = wacc > 0 ? Math.floor(wacc / 50) : Math.ceil(wacc / 50); wacc -= n * 50; }   // touchpad: small steps add up
     if (n) step(-n, view); else draw();              // downwards = next image (towards the feet in the axial view)
   }, {passive: false});
   canvas.addEventListener('contextmenu', e => e.preventDefault());
   canvas.addEventListener('pointerdown', e => {
     canvas.setPointerCapture(e.pointerId);
     const xy = canvasXY(e), R = paneAt(xy[0], xy[1]), view = R ? R.view : AV();
     if (S.mode === 'three' && R) S.focus = view;
     drag = {x: e.clientX, y: e.clientY, moved: false, mode: (e.button === 2 || e.shiftKey) ? 'window' : 'pan', win: S.win.slice(), view: view, pan: S.zp[view].pan.slice()};
     draw();
   });
   canvas.addEventListener('pointermove', e => {
     if (!drag) return;
     const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
     if (!drag.moved && Math.abs(dx) + Math.abs(dy) < 4) return;
     drag.moved = true;
     if (drag.mode === 'window') {                   // right = wider (less contrast), down = darker
       const w0 = drag.win[1] - drag.win[0], c0 = (drag.win[0] + drag.win[1]) / 2, w = Math.max(1, w0 + dx * 4), c = c0 + dy * 3;
       setWindow(c - w / 2, c + w / 2, false);
     } else { const k = canvas.width / Math.max(1, canvas.getBoundingClientRect().width); S.zp[drag.view].pan = [drag.pan[0] + dx * k, drag.pan[1] + dy * k]; draw(); }
   });
   const end = e => {
     if (!drag) return;
     if (!drag.moved && drag.mode === 'pan') { const p = pick(e); if (p && p.vox) { S.cur = p.vox; draw(); } }   // a click moves the cross-hair in every pane
     drag = null;
   };
   canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', () => { drag = null; });
   canvas.addEventListener('dblclick', e => { const xy = canvasXY(e), R = paneAt(xy[0], xy[1]); resetZoom(R ? R.view : AV()); draw(); });
   canvas.tabIndex = 0;
   canvas.addEventListener('keydown', e => { const d = {ArrowDown: 1, ArrowUp: -1, PageDown: 10, PageUp: -10}[e.key]; if (d) { e.preventDefault(); step(-d); } });
   if (window.ResizeObserver) new ResizeObserver(draw).observe(canvas.parentElement); else window.addEventListener('resize', draw);
   $$('tr.jump').forEach(tr => tr.onclick = async () => {
     if (document.body.classList.contains('stage1')) return;          // staged page: no outline before step 1 is locked
     const k = tr.dataset.jump, t = META.targets[k];
     if (!t) return;
     await setOverlay(k);
     if (V.active !== k) return;
     goToMm(t);                                      // physical position (mm) -> image in every shown view
     $$('tr.sel').forEach(x => x.classList.remove('sel')); tr.classList.add('sel');
   });
   V.ready = true; V.timings.ready_ms = Math.round(performance.now() - T0);
   status(null); draw();
 }
 start().catch(e => { status('<b>The scan viewer stopped:</b> ' + (e && e.message || e) + '.' + SAFE + '<br><br><button onclick="location.reload()">Reload the page</button>', undefined, true); });
})();
"""


def viewer_html(agentonly: bool) -> str:
    return VIEWER_HTML.replace("__AO__", " agentonly" if agentonly else "")


def viewer_script(meta: dict) -> str:
    return "<script>\n" + JS.replace("__META__", json.dumps(meta)) + "\n</script>"


def standalone_page(title: str, arr: np.ndarray, affine: np.ndarray, overlays: dict | None = None,
                    targets: dict | None = None) -> str:
    """A minimal page with ONLY the viewer (no checklist): used for the geometry tests on synthetic volumes.
    arr: int16 volume in ANY orientation with its affine; overlays: {key: (label, '#rrggbb', mask in arr's orientation)}."""
    ornt, can, codes, ang = canonical_transform(affine, arr.shape)
    html, ct_meta = ct_payload(to_canonical(arr, ornt), can, codes, ang)
    vols, ovs = {}, []
    for key, (label, colour, mask) in (overlays or {}).items():
        h, vols[key] = overlay_payload(key, to_canonical(mask, ornt))
        html += h
        ovs.append({"key": key, "label": label, "swatch": colour})
    meta = {"build": VIEWER_BUILD, "case_id": title, "ct": ct_meta, "vols": vols, "overlays": ovs, "targets": targets or {}}
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>" + title + "</title><style>"
            "*{box-sizing:border-box}body{margin:0;font:13px/1.35 'Segoe UI',Arial,sans-serif;background:#0f1115;color:#e6e6e6}"
            ".page{height:100vh}.viewer{display:flex;flex-direction:column;height:100%}"
            ".bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:8px 10px;background:#161a22}"
            "button{background:#232937;color:#e6e6e6;border:1px solid #333b4d;border-radius:6px;padding:4px 9px}button.on{background:#2f6feb}"
            ".ov{display:inline-flex;gap:5px;padding:3px 8px;border:1px solid #333b4d;border-radius:6px;cursor:pointer}.ov.off{opacity:.45}"
            ".sw{width:11px;height:11px;display:inline-block}#loc{padding:5px 10px;font:12px Consolas,monospace;color:#9aa4b2;background:#161a22}"
            + CSS + "</style></head><body><div class='page'>" + viewer_html(False) + "</div>\n<script>\n" + EARLY_JS + "\n</script>\n"
            + html + viewer_script(meta) + "\n</body></html>")
