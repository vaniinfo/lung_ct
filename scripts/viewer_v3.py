"""Loader for the standalone viewer page, build "checklist_viewer_v3_lazy" (build_viewer_page.py --viewer v3).

Same page, same answers, same grading form as viewer v2. What changes is HOW the scan gets into the browser:

  * Still ONE html file opened from file://. No server, no install, no browser setting.
  * The CT is stored losslessly: every voxel is predicted from three already-decoded neighbours (previous voxel,
    previous row, previous slice), the difference is written as one byte (three bytes when it is 255 or more), and
    the byte stream is gzipped. The browser reverses this exactly and checks a checksum and, where the browser allows,
    a SHA-256 of the voxel bytes against the values computed here from the original array. No resampling, no change of
    HU values, geometry or orientation: the NIfTI header is embedded unchanged.
  * Payloads sit in <script type="application/octet-stream"> blocks, not in a JavaScript literal. The CT blocks are
    decoded piece by piece and removed from the page as they are consumed.
  * Overlays stay compressed until one is asked for. Only ONE overlay is loaded at a time; the previous one is removed
    from NiiVue (removeVolume) and its references dropped before the next is decoded.
  * Visible loading and error states; an overlay that fails does not block later clicks; WebGL context loss is caught.

Workaround for the bundled NiiVue 0.69.0 (found by reading scripts/vendor/niivue-0.69.0.umd.js): when one overlay is
present, every updateGLVolume() call (overlay on/off, window change, contrast drag) creates a new full-size RGBA 3D
texture for the overlay layer without deleting the previous one ("rgbaTex(null, TEXTURE2, dims)"). The page hands the
previous overlay texture to that call so it is deleted first, and shrinks the overlay texture again when no overlay is
shown. This uses the instance's own rgbaTex method; it changes nothing that is drawn.
"""
from __future__ import annotations

import base64
import hashlib
import json
import zlib

import nibabel as nib
import numpy as np

VIEWER_BUILD = "checklist_viewer_v3_lazy"
CODEC = "avg3-zigzag-escape-gzip-v1"
PART_BYTES = 6 * 1024 * 1024  # multiple of 3: every part is a complete base64 string
CHECK_MOD = 65521


def _nifti_header(arr_dtype, shape, affine) -> bytes:
    """The first 352 bytes of a single-file .nii for this grid (348-byte header + 4-byte extension flag)."""
    img = nib.Nifti1Image(np.zeros((1, 1, 1), arr_dtype), affine)
    img.header.set_data_shape(shape)
    img.header.set_xyzt_units("mm")
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header["vox_offset"] = 352
    raw = img.header.binaryblock + b"\x00\x00\x00\x00"
    assert len(raw) == 352
    return raw


def checksum(values: np.ndarray) -> int:
    """sum((v + 1) * ((i mod 65521) + 1)) mod 2^32 over raster-ordered non-negative values; the page computes the same."""
    acc = 0
    step = CHECK_MOD * 4096
    for s in range(0, values.size, step):
        v = values[s:s + step].astype(np.uint64) + 1
        w = (np.arange(s, s + v.size, dtype=np.uint64) % CHECK_MOD) + 1
        acc = (acc + int((v * w).sum() % (1 << 32))) % (1 << 32)
    return acc


def encode_ct_stream(arr: np.ndarray) -> tuple[bytes, dict]:
    """Lossless residual byte stream of an int16 volume (Fortran / NIfTI raster order). Not yet gzipped."""
    if arr.dtype != np.int16 or arr.ndim != 3:
        raise ValueError("CT must be a 3-D int16 array")
    nx, ny, nz = arr.shape
    off = -int(arr.min())
    u = arr.astype(np.int32).ravel(order="F") + off          # >= 0

    def shifted(k):
        p = np.zeros_like(u)
        if k < u.size:
            p[k:] = u[:-k]
        return p
    pred = (shifted(1) + shifted(nx) + shifted(nx * ny)) // 3
    res = u - pred
    zz = ((res << 1) ^ (res >> 31)).astype(np.uint32)
    if int(zz.max()) > 0xFFFF:
        raise ValueError("value range too wide for this coding")
    big = zz >= 255
    n = zz.size
    out = np.empty(n + 2 * int(big.sum()), np.uint8)
    pos = np.arange(n, dtype=np.int64)
    pos[1:] += 2 * np.cumsum(big[:-1], dtype=np.int64)
    out[pos] = np.where(big, 255, zz).astype(np.uint8)
    pb = pos[big]
    out[pb + 1] = (zz[big] & 255).astype(np.uint8)
    out[pb + 2] = (zz[big] >> 8).astype(np.uint8)
    meta = {"codec": CODEC, "dims": [nx, ny, nz], "offset": off, "voxels": n, "stream_bytes": int(out.size),
            "checksum": checksum(u), "sha256": hashlib.sha256(arr.astype("<i2").tobytes(order="F")).hexdigest()}
    return out.tobytes(), meta


def decode_ct_stream(stream: bytes, meta: dict) -> np.ndarray:
    """Reference decoder (plain Python, slow): used by the unit test on small volumes. The page has the same logic."""
    nx, ny, nz = meta["dims"]
    n, off, nxy = nx * ny * nz, meta["offset"], nx * ny
    u = [0] * n
    k = 0
    for i in range(n):
        b = stream[k]
        k += 1
        if b == 255:
            b = stream[k] | (stream[k + 1] << 8)
            k += 2
        r = (b >> 1) ^ -(b & 1)
        p = ((u[i - 1] if i >= 1 else 0) + (u[i - nx] if i >= nx else 0) + (u[i - nxy] if i >= nxy else 0)) // 3
        u[i] = p + r
    if k != len(stream):
        raise ValueError("stream length mismatch")
    return (np.array(u, np.int32) - off).astype(np.int16).reshape((nx, ny, nz), order="F")


def _blocks(vol: str, gz: bytes) -> tuple[str, int]:
    parts = [gz[i:i + PART_BYTES] for i in range(0, len(gz), PART_BYTES)] or [b""]
    html = "".join(f'<script type="application/octet-stream" class="vp" data-vol="{vol}" data-part="{k}">'
                   f"{base64.b64encode(p).decode()}</script>\n" for k, p in enumerate(parts))
    return html, len(parts)


def ct_payload(arr: np.ndarray, affine: np.ndarray) -> tuple[str, dict]:
    stream, meta = encode_ct_stream(arr)
    gz = zlib.compress(stream, 6, wbits=31)                  # wbits 31 = gzip container (DecompressionStream('gzip'))
    html, n_parts = _blocks("ct", gz)
    meta.update(header_b64=base64.b64encode(_nifti_header(np.int16, arr.shape, affine)).decode(), gz_bytes=len(gz),
                parts=n_parts, raw_bytes=int(arr.size * 2))
    return html, meta


def overlay_payload(key: str, mask: np.ndarray, affine: np.ndarray) -> tuple[str, dict]:
    """A 0/1 uint8 mask on the CT grid: raw bytes gzipped, decoded only when the overlay is requested."""
    raw = np.ascontiguousarray(mask.astype(np.uint8)).tobytes(order="F")
    gz = zlib.compress(raw, 9, wbits=31)
    html, n_parts = _blocks(key, gz)
    return html, {"codec": "raw-gzip-v1", "dims": list(mask.shape), "voxels": int(mask.size), "raw_bytes": len(raw),
                  "gz_bytes": len(gz), "parts": n_parts, "sha256": hashlib.sha256(raw).hexdigest(),
                  "header_b64": base64.b64encode(_nifti_header(np.uint8, mask.shape, affine)).decode()}


CSS = """
.viewer{position:relative}
#vstat{position:absolute;left:50%;top:46%;transform:translate(-50%,-50%);z-index:20;background:rgba(22,26,34,.94);border:1px solid #3b4558;
 border-radius:10px;padding:14px 18px;min-width:300px;max-width:70%;text-align:center;color:#e6e6e6;font-size:14px}
#vstat.err{border-color:#b4523f;color:#ffd9d1;text-align:left}
#vstat .pbar{height:6px;background:#2a3140;border-radius:3px;margin-top:9px;overflow:hidden}
#vstat .pbar i{display:block;height:100%;width:0;background:#2f6feb}
#vbanner{display:none;padding:7px 10px;background:#4a2a22;color:#ffd9d1;border-bottom:1px solid #7a3b2e;font-size:12.5px}
#vbanner button{margin-left:8px}
.ov.busy{opacity:1;border-color:#2f6feb}.ov.busy::after{content:' loading...';color:#9fc2ff;font-size:11px}
.ov.fail{border-color:#b4523f}
"""

STATUS_HTML = ('<div id="vbanner"></div><div id="vstat">Loading the scan&hellip;<div class="pbar"><i></i></div>'
               '<div class="muted" style="margin-top:6px;color:#9aa4b2">You can already use the grading form on the right.</div></div>')

# UI that must work even if the viewer cannot start (placed before the payload blocks)
EARLY_JS = r"""
document.querySelectorAll('.mbtn').forEach(b => b.onclick = e => {
  e.stopPropagation(); const d = b.nextElementSibling; d.hidden = !d.hidden; b.textContent = d.hidden ? 'Numbers' : 'Hide numbers';
});
document.querySelectorAll('.mdet').forEach(d => d.onclick = e => e.stopPropagation());
window.__viewerErrors = [];
"""

JS = r"""
const META = __META__;
(function () {
 const T0 = performance.now();
 const $ = s => document.querySelector(s), $$ = s => Array.from(document.querySelectorAll(s));
 const V = window.__viewer = {build: META.build, ready: false, lost: false, active: null, timings: {}, verified: null, log: []};
 const stat = $('#vstat'), banner = $('#vbanner');
 function status(html, frac, isErr) {
   if (html === null) { stat.style.display = 'none'; return; }
   stat.style.display = ''; stat.className = isErr ? 'err' : '';
   stat.innerHTML = html + (frac === undefined || isErr ? '' : '<div class="pbar"><i style="width:' + Math.round(100 * frac) + '%"></i></div>');
 }
 function warn(msg, withReload) {            // non-blocking message above the scan
   V.log.push(msg); window.__viewerErrors.push(msg);
   banner.style.display = 'block';
   banner.innerHTML = '';
   banner.appendChild(document.createTextNode(msg + ' '));
   if (withReload) { const r = document.createElement('button'); r.textContent = 'Reload the page'; r.onclick = () => location.reload(); banner.appendChild(r); }
   const x = document.createElement('button'); x.textContent = 'Dismiss'; x.onclick = () => { banner.style.display = 'none'; }; banner.appendChild(x);
 }
 const SAFE = ' Your grades are saved in this browser and are not affected.';
 window.addEventListener('error', e => warn('The viewer reported an error: ' + (e.message || 'unknown') + '.' + SAFE, false));
 window.addEventListener('unhandledrejection', e => { warn('The viewer reported an error: ' + ((e.reason && e.reason.message) || e.reason || 'unknown') + '.' + SAFE, false); });
 const tick = () => new Promise(r => setTimeout(r, 0));

 // ---------- decoding (lossless; nothing is resampled) ----------
 function b64ToBytes(txt) { if (typeof Uint8Array.fromBase64 === 'function') return Uint8Array.fromBase64(txt);
   const bin = atob(txt), n = bin.length, u = new Uint8Array(n); for (let i = 0; i < n; i++) u[i] = bin.charCodeAt(i); return u; }
 function gzStream(vol, consume, onPart) {  // compressed bytes -> decompressed chunks, one base64 block at a time
   if (typeof DecompressionStream === 'undefined') throw new Error('this browser is too old (no DecompressionStream); use a current Chrome or Edge');
   const els = $$('script.vp[data-vol="' + vol + '"]').sort((a, b) => a.dataset.part - b.dataset.part);
   if (!els.length) throw new Error('the data for "' + vol + '" is not in this file');
   let k = 0;
   const src = new ReadableStream({ pull(c) {
     if (k >= els.length) { c.close(); return; }
     const el = els[k++]; let bytes;
     try { bytes = b64ToBytes(el.textContent); } catch (e) { c.error(new Error('the embedded data is damaged (base64)')); return; }
     if (consume) el.remove();              // the compressed text is no longer needed
     if (onPart) onPart(k / els.length);
     c.enqueue(bytes);
   } }, {highWaterMark: 1});
   return src.pipeThrough(new DecompressionStream('gzip')).getReader();
 }
 const hex = buf => Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('');
 async function sha256(view) { return (window.crypto && crypto.subtle) ? hex(await crypto.subtle.digest('SHA-256', view)) : null; }
 function header(m, nbytes) { const buf = new ArrayBuffer(352 + nbytes); new Uint8Array(buf, 0, 352).set(b64ToBytes(m.header_b64)); return buf; }

 async function decodeCT(onProg) {
   const m = META.ct, nx = m.dims[0], nxy = m.dims[0] * m.dims[1], n = m.voxels, off = m.offset;
   const buf = header(m, n * 2), out = new Int16Array(buf, 352, n);
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
     if (performance.now() - lastYield > 60) { await tick(); lastYield = performance.now(); }   // keep the page responsive
   }
   if (i !== n || got !== m.stream_bytes) throw new Error('the embedded scan is incomplete (' + i + ' of ' + n + ' voxels)');
   if (acc !== m.checksum) throw new Error('the embedded scan failed its checksum; the file may be damaged');
   const sha = await sha256(new Uint8Array(buf, 352));
   if (sha !== null && sha !== m.sha256) throw new Error('the embedded scan failed its SHA-256 check; the file may be damaged');
   V.verified = {checksum: true, sha256: sha === null ? 'not available in this browser' : true};
   return buf;
 }
 async function decodeOverlay(key) {
   const m = META.vols[key], buf = header(m, m.raw_bytes), out = new Uint8Array(buf, 352);
   const rd = gzStream(key, false, null);
   let o = 0;
   for (;;) {
     const {done, value} = await rd.read();
     if (done) break;
     if (o + value.length > out.length) throw new Error('overlay data is longer than its header says');
     out.set(value, o); o += value.length;
   }
   if (o !== m.raw_bytes) throw new Error('overlay data is incomplete');
   const sha = await sha256(out);
   if (sha !== null && sha !== m.sha256) throw new Error('overlay data failed its SHA-256 check');
   return buf;
 }

 // ---------- viewer ----------
 let nv = null, chain = Promise.resolve(), activeVol = null;
 function overlayTexSmall() {                // no overlay shown: do not keep a full-size RGBA overlay texture
   try { const gl = nv.gl; nv.overlayTexture = nv.rgbaTex(nv.overlayTexture, gl.TEXTURE2, [2, 2, 2, 2], true); nv.overlayTextureID = nv.overlayTexture; } catch (e) { V.log.push('overlay texture reset: ' + e); }
 }
 function chip(k) { return $('.ov[data-k="' + k + '"]'); }
 function paintChips() { for (const o of META.overlays) { const c = chip(o.key); if (c) { c.classList.toggle('off', V.active !== o.key); c.classList.remove('busy'); } } }
 // One overlay at a time. Calls are queued; a failure is reported and the queue goes on.
 function setOverlay(key) {
   chain = chain.then(async () => {
     if (!V.ready || V.lost) return;
     if (key === V.active) return;
     if (activeVol) {                        // remove the previous overlay from NiiVue and drop every reference to it
       const old = activeVol; activeVol = null; V.active = null;
       try { nv.removeVolume(old); } catch (e) { V.log.push('removeVolume: ' + e); }
       old.img = null;
       overlayTexSmall(); nv.drawScene(); paintChips();
     }
     if (!key) return;
     const o = META.overlays.find(x => x.key === key), c = chip(key);
     if (c) { c.classList.add('busy'); c.classList.remove('fail'); }
     try {
       const t = performance.now();
       let buf = await decodeOverlay(key);
       const img = await niivue.NVImage.new(buf, key + '.nii', o.cmap, 0.55, null, 0.5, 1);
       buf = null;
       nv.addVolume(img); activeVol = img; V.active = key;
       V.timings['overlay_' + key + '_ms'] = Math.round(performance.now() - t);
     } catch (e) {
       if (c) c.classList.add('fail');
       warn('The "' + o.label + '" outline could not be shown (' + (e && e.message || e) + '). The scan and the other outlines still work.' + SAFE, false);
     }
     paintChips();
   }).catch(e => { V.log.push('overlay queue: ' + e); paintChips(); });
   return chain;
 }
 V.setOverlay = setOverlay; V.idle = () => chain;

 async function start() {
   const canvas = $('#gl');
   canvas.addEventListener('webglcontextlost', e => {
     e.preventDefault(); V.lost = true;
     status('<b>The graphics driver stopped the scan viewer</b> (WebGL context lost). This usually means the computer ran short of graphics memory.' + SAFE +
            '<br><br><button onclick="location.reload()">Reload the page</button>', undefined, true);
   });
   canvas.addEventListener('webglcontextrestored', () => warn('The graphics driver is available again. Reload the page to show the scan.', true));
   try {
     nv = new niivue.Niivue({isRadiologicalConvention: true, backColor: [0, 0, 0, 1], crosshairColor: [1, 0.85, 0.1, 1],
       show3Dcrosshair: false, isColorbar: false, multiplanarShowRender: niivue.SHOW_RENDER.NEVER});
     await nv.attachTo('gl');
     if (!nv.gl) throw new Error('no WebGL2');
   } catch (e) {
     status('<b>The scan viewer cannot start on this computer</b> (WebGL2 is not available: ' + (e && e.message || e) + '). Use a current Chrome or Edge.' + SAFE, undefined, true);
     return;
   }
   V.nv = nv;
   // NiiVue 0.69.0 creates a new full-size overlay texture on every refresh without deleting the old one: hand it the old one.
   const rgbaTex0 = nv.rgbaTex.bind(nv), TEX_OVERLAY = nv.gl.TEXTURE2;
   nv.rgbaTex = function (tex, unit, dims, init) {
     if (!tex && unit === TEX_OVERLAY && nv.overlayTexture) { tex = nv.overlayTexture; nv.overlayTexture = null; V.texReused = (V.texReused || 0) + 1; }
     return rgbaTex0(tex, unit, dims, init);
   };
   let buf;
   try {
     status('Loading the scan&hellip; decoding', 0);
     let last = 0;
     buf = await decodeCT(f => { if (f - last >= 0.02 || f === 1) { last = f; status('Loading the scan&hellip; ' + Math.round(100 * f) + '%', f); } });
     V.timings.decode_ms = Math.round(performance.now() - T0);
     status('Loading the scan&hellip; preparing the image', 1);
     await tick();
     const ct = await niivue.NVImage.new(buf, 'ct.nii', 'gray', 1, null, -1350, 150);
     buf = null;
     nv.addVolume(ct);
   } catch (e) {
     status('<b>The scan could not be loaded:</b> ' + (e && e.message || e) + '.' + SAFE + '<br><br><button onclick="location.reload()">Reload the page</button>', undefined, true);
     V.log.push('ct: ' + e);
     return;
   }
   nv.setSliceType(nv.sliceTypeAxial);
   nv.onLocationChange = d => { $('#loc').textContent = d.string || ''; };
   for (const o of META.overlays) {
     const el = document.createElement('span'); el.className = 'ov off'; el.dataset.k = o.key;
     el.innerHTML = '<span class="sw" style="background:' + o.swatch + '"></span>' + o.label;
     el.onclick = () => setOverlay(V.active === o.key ? null : o.key);
     $('#ovbar').appendChild(el);
   }
   $$('[data-w]').forEach(b => b.onclick = () => {
     if (V.lost) return;
     const [lo, hi] = b.dataset.w.split(',').map(Number);
     nv.volumes[0].cal_min = lo; nv.volumes[0].cal_max = hi; nv.updateGLVolume();
     $$('[data-w]').forEach(x => x.classList.toggle('on', x === b));
   });
   $$('[data-v]').forEach(b => b.onclick = () => {
     if (V.lost) return;
     nv.setSliceType(Number(b.dataset.v));
     $$('[data-v]').forEach(x => x.classList.toggle('on', x === b));
   });
   $$('tr.jump').forEach(tr => tr.onclick = async () => {
     if (document.body.classList.contains('stage1')) return;          // staged page: no outline before step 1 is locked
     const k = tr.dataset.jump, t = META.targets[k];
     if (!t || V.lost) return;
     await setOverlay(k);
     if (V.active !== k) return;
     nv.scene.crosshairPos = nv.mm2frac(t); nv.drawScene();
     $$('tr.sel').forEach(x => x.classList.remove('sel')); tr.classList.add('sel');
   });
   V.ready = true; V.timings.ready_ms = Math.round(performance.now() - T0);
   status(null);
 }
 start().catch(e => { status('<b>The scan viewer stopped:</b> ' + (e && e.message || e) + '.' + SAFE + '<br><br><button onclick="location.reload()">Reload the page</button>', undefined, true); });
})();
"""


def viewer_script(meta: dict) -> str:
    return "<script>\n" + JS.replace("__META__", json.dumps(meta)) + "\n</script>"
