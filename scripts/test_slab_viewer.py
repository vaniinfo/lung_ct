"""Technical checks of viewer v6 ("Axial 2.5 mm" mean slab, "MIP 10 mm", native slices under Details), run against the
UNMODIFIED html from file:// (core env).

Technical checks only. Whether the display is useful for reading is a question for the clinician reviewer and is not
answered here.

  --synthetic   small volumes with KNOWN voxel values, stored in several axis orders / flips, with slice spacings
                0.75, 1.25, 2.0, 2.5 and 6 mm. For every spacing, both slab views and centre images including both scan
                edges, the page's slab must equal the mean / maximum computed here with numpy from the patient-oriented
                volume, and the stated image range and thickness must equal the arithmetic written out in this file.
                Also: opening state, window independence, orientation, switching between the three views, other planes,
                outlines, zoom, cache bound.
  --page FILE   a real case page: opening state, both slabs compared with SimpleITK reading the ORIGINAL file, labels
                and Details, native slices, zoom, real wheel scrolling, a stress block with memory measured as in
                test_canvas_viewer.py, the AI-reader lines, then grading and CSV export.

Usage: python scripts/test_slab_viewer.py --synthetic --browser edge
       python scripts/test_slab_viewer.py --page runs/M12_.../viewer_valid_301_M12_v3_10.html --browser edge --gpu-mode igpu
Writes runs/V11_slab_test_<label>_<time>/{result.json,result.md,*.png}.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import viewer_v6 as v6  # noqa: E402
from test_canvas_viewer import GPU_FLAGS, Browser, arr8, arr16, grid_from_sitk, variant, wait_ready  # noqa: E402
from test_mip_viewer import VARIANTS, axial_display, lut, phantom, zoom_kept  # noqa: E402
from test_viewer_page import BROWSERS, PREP, gpu_mb  # noqa: E402

MM = {"mean": 2.5, "mip": 10}


def plan(k: int, nz: int, dz: float, mm: float) -> dict:
    """The slab rule, written out here independently of the viewer code (k: larger = more superior)."""
    n = int(mm / dz + 0.5)                      # nearest whole number of images, halves up
    if n < 1:
        n = 1
    lo = k - (n - 1) // 2                       # with an even count the extra image is on the superior side
    hi = lo + n - 1
    k0, k1 = max(lo, 0), min(hi, nz - 1)        # shortened at the scan edge, never shifted
    return {"n": n, "k0": k0, "k1": k1, "images": k1 - k0 + 1, "actual_mm": (k1 - k0 + 1) * dz, "truncated": (k0, k1) != (lo, hi)}


def ref_slab(kind: str, vol: np.ndarray, k0: int, k1: int) -> np.ndarray:
    """vol: [i, j, k] patient-oriented. Mean: exact integer sum, one division, halves up. Max: plain maximum."""
    part = vol[:, :, k0:k1 + 1]
    if kind == "mip":
        return part.max(axis=2)
    return np.floor(part.astype(np.int64).sum(axis=2) / float(k1 - k0 + 1) + 0.5).astype(np.int16)


def g(x: float) -> str:
    return f"{round(x, 2):g}"


def label(kind: str, p: dict, dz: float, nz: int, k: int) -> str:
    """The one-line label the page must show for a slab view."""
    name = "MIP" if kind == "mip" else "Axial"
    no = nz - k
    if p["n"] == 1:
        tail = ("same as the native slice" if kind == "mip" else f"slices of this scan are {g(dz)} mm apart" if abs(dz - 2.5) <= 0.25
                else f"a 2.5 mm view is not available for this scan (slices {g(dz)} mm apart)")
        return f"{name} {g(dz)} mm  ·  native slice  ·  Image {no}  ·  {tail}"
    approx = f"{int(p['actual_mm'] * 10 + 0.5) / 10:g}" if p["truncated"] else f"{MM[kind]:g}"
    return (f"{name} ≈{approx} mm  ·  {'' if kind == 'mip' else 'mean of '}{p['images']} slice{'' if p['images'] == 1 else 's'}  ·  Images {nz - p['k1']}–{nz - p['k0']}  ·  Centre {no}"
            + ("  ·  shortened at scan edge" if p["truncated"] else ""))


SNAP = ("const snap = () => { const s = v.state(), L = v.layout('axial'), r = v.canvasRect(); return {zoom: s.zoom, pan: s.pan, px: s.pxScale, L: [L.x, L.y, L.w, L.h], cv: [gl.width, gl.height], cr: r,"
        " inside: L.x >= r[0] - 0.5 && L.y >= r[1] - 0.5 && L.x + L.w <= r[0] + r[2] + 0.5 && L.y + L.h <= r[1] + r[3] + 0.5}; };")
OPEN = ("(() => { const v = window.__viewer, st = v.state(), sl = v.slab(); const L = v.layout('axial'), r = v.canvasRect(); return {st, sl, build: v.build, kind: v.kind, proto: location.protocol, errors: window.__viewerErrors,"
        " body: document.body.classList.contains('slabmode'), on25: document.getElementById('sl25').classList.contains('on'), onMip: document.getElementById('slmip').classList.contains('on'),"
        " ovbar: getComputedStyle(document.getElementById('ovbar')).display, note: getComputedStyle(document.getElementById('slnote')).display, noteText: document.getElementById('slnote').textContent,"
        " detHidden: document.getElementById('sldet').hidden, winOn: (document.querySelector('[data-w].on') || {}).textContent, viewOn: document.querySelector('[data-v].on').dataset.v,"
        " chipsOn: document.querySelectorAll('#ovbar .ov:not(.off)').length, shown: v.shownGray('axial'), native: v.sliceGray('axial', st.cur[2]).b64, btns: Array.from(document.querySelectorAll('#slbar > button')).map(b => b.textContent),"
        " fit: {zoom: st.zoom, pan: st.pan, inside: L.x >= r[0] - 0.5 && L.y >= r[1] - 0.5 && L.x + L.w <= r[0] + r[2] + 0.5 && L.y + L.h <= r[1] + r[3] + 0.5, fills: Math.max(L.w / r[2], L.h / r[3])},"
        " saved: Object.keys(localStorage).filter(k => /zoom|pan/i.test(k) && k !== 'lungct_text_zoom')}; })()")


def run_synthetic(args, out: Path, R: dict, ok) -> None:
    tmp = out / "pages"
    tmp.mkdir()
    port = args.port
    for dz in (0.75, 1.25, 2.0, 2.5, 6.0):
        base, mask, aff, marks = phantom(dz)
        nz = base.shape[2]
        n25, n10 = plan(12, nz, dz, 2.5)["n"], plan(12, nz, dz, 10)["n"]
        R.setdefault("rounding", {})[str(dz)] = {"axial_2_5": {"images": n25, "mm": n25 * dz}, "mip_10": {"images": n10, "mm": n10 * dz}}
        for vi, (perm, flips) in enumerate(VARIANTS if dz == 0.75 else VARIANTS[:2]):
            arr, a2 = variant(base, aff, perm, flips)
            m2, _ = variant(mask, aff, perm, flips)
            page = tmp / f"slab_dz{dz}_v{vi}.html"
            thick = 1.5 if vi == 0 else None
            page.write_text(v6.standalone_page(f"slab dz {dz} variant {vi}", arr, a2, {"m": ("test outline", "#ef4444", m2)},
                                               slab={"reported_thickness_mm": thick, "reconstruction_note": "TEST NOTE."}), encoding="utf-8")
            b = Browser(args.browser, "sw", page.as_uri(), port)
            port += 1
            c = b.c
            tag = f"dz {dz} mm, stored as perm {perm} flips {flips}"
            try:
                _, st = wait_ready(b, 120)
                if not st["ready"]:
                    ok(f"{tag}: page loads", False, st["err"])
                    continue
                R.setdefault("browser_version", b.version)
                time.sleep(0.3)
                s0 = c.js(OPEN)
                k0 = s0["st"]["cur"][2]
                p0 = plan(k0, nz, dz, 2.5)
                want0 = lut(axial_display(ref_slab("mean", base, p0["k0"], p0["k1"])), -1350, 150)
                btn = "Axial 2.5 mm" if (n25 > 1 or abs(dz - 2.5) <= 0.25) else f"Axial {g(dz)} mm"
                ok(f"{tag}: opens on the axial mean slab ('{btn}'), fitted, lung window, no outline, outline bar absent, Details closed, no error",
                   s0["sl"]["mode"] == "mean" and s0["sl"]["shown"] == "mean" and s0["body"] and s0["on25"] and not s0["onMip"] and s0["st"]["view"] == "axial" and s0["st"]["mode"] == "single"
                   and s0["fit"]["zoom"] == 1 and s0["fit"]["pan"] == [0, 0] and s0["fit"]["inside"] and 0.93 < s0["fit"]["fills"] <= 0.95 and s0["sl"]["winText"] == "W 1500  ·  L -600"
                   and s0["st"]["active"] is None and s0["ovbar"] == "none" and s0["note"] != "none" and s0["detHidden"] and not s0["errors"] and s0["build"] == v6.VIEWER_BUILD
                   and s0["btns"][:3] == [btn, "MIP 10 mm", "Details"] and s0["shown"]["coloured_pixels"] == 0 and np.array_equal(arr8(s0["shown"]), want0)
                   and ((arr8(s0["shown"]).tobytes() != base64.b64decode(s0["native"])) or p0["images"] == 1),
                   f"image fills {s0['fit']['fills']:.2f} of the scan area; buttons {s0['btns'][:3]}; {s0['sl']['info']}")
                ok(f"{tag}: slice spacing read from the affine; only 2.5 mm and 10 mm exist (no 5 / 8 mm option)",
                   abs(s0["sl"]["spacing_mm"] - dz) < 1e-9 and s0["sl"]["images_total"] == nz and s0["sl"]["mm"] == {"mean": 2.5, "mip": 10}
                   and c.js("document.querySelectorAll('[data-mip]').length") == 0, s0["sl"]["spacing_mm"])
                # ---- known values: both views, centres including both edges ----
                bad, n_cmp, seen, differs = [], 0, set(), 0
                centres = sorted({0, 1, 2, 5, 11, 12, 13, 20, nz - 3, nz - 2, nz - 1})
                for kind in ("mean", "mip"):
                    for k in centres:
                        d = c.js(f"window.__viewer.slabHU('{kind}', {k})")
                        p, pv = plan(k, nz, dz, MM[kind]), v6.slab_plan(k, nz, dz, MM[kind])
                        want = axial_display(ref_slab(kind, base, p["k0"], p["k1"]))
                        got = arr16(d)
                        n_cmp += 1
                        seen.add((kind, p["n"], p["truncated"]))
                        same_plan = (d["k0"], d["k1"], d["images"], d["planned_images"], d["truncated"]) == (p["k0"], p["k1"], p["images"], p["n"], p["truncated"]) \
                            and abs(d["actual_mm"] - p["actual_mm"]) < 1e-9 and (pv["k0"], pv["k1"], pv["planned_images"]) == (p["k0"], p["k1"], p["n"])
                        if not same_plan or got.shape != want.shape or not np.array_equal(got, want):
                            bad.append((kind, k, d["k0"], d["k1"], p))
                        if p["images"] > 1 and not np.array_equal(got, axial_display(base[:, :, k])):
                            differs += 1
                ok(f"{tag}: mean slabs equal floor(sum / n + 0.5) and MIP slabs equal the maximum of the known volume for {n_cmp} slabs (centre images at both scan edges and in between); image counts follow the stated rule",
                   not bad, f"2.5 mm -> {n25} image(s) = {g(n25 * dz)} mm; 10 mm -> {n10} image(s) = {g(n10 * dz)} mm; shortened slabs tested: {any(t for _, _, t in seen)}" if not bad else str(bad[:3]))
                # ---- the mean is a real average: a known isolated voxel, and it is not a skipped / relabelled native image ----
                (mi, mj, mk), mv = (3, 27, 12), 2500
                col, row = base.shape[0] - 1 - mi, base.shape[1] - 1 - mj
                pm = plan(mk, nz, dz, 2.5)
                want_v = int(np.floor(base[mi, mj, pm["k0"]:pm["k1"] + 1].astype(np.int64).sum() / pm["images"] + 0.5))
                got_v = int(arr16(c.js(f"window.__viewer.slabHU('mean', {mk})"))[row, col])
                ok(f"{tag}: averaging is real: the 2500 HU marker voxel reads {want_v} HU in the mean of {pm['images']} image(s) (hand calculation), "
                   + ("and every multi-image slab differs from its centre image (no skipping, no relabelling)" if n25 > 1 else "one image only, so the native image is shown unchanged"),
                   got_v == want_v and (n25 == 1 or (differs > 0 and got_v != mv)) and (n25 > 1 or got_v == mv), f"page {got_v} HU, expected {want_v} HU")
                # ---- orientation ----
                inside = []
                for kind in ("mean", "mip"):
                    for k in range(nz):
                        d = c.js(f"window.__viewer.slabHU('{kind}', {k})")
                        val = int(arr16(d)[row, col])
                        has = d["k0"] <= mk <= d["k1"]
                        inside.append((val == mv) == has if kind == "mip" else (val > 0) == has)       # background there is below -100 HU: a positive mean needs the marker
                edges = c.js("window.__viewer.edges('axial')")
                ok(f"{tag}: orientation: the patient-LEFT, ANTERIOR marker voxel shows on the screen RIGHT and TOP in both slab views, in every slab that contains its image and in no other",
                   all(inside) and col > base.shape[0] / 2 and row < base.shape[1] / 2 and edges == ["R", "L", "A", "P"], f"column {col} of {base.shape[0]}, row {row} of {base.shape[1]}; edge letters {edges}")
                # ---- what is DRAWN: window applied to the slab values ----
                k = 12
                wins, good = [(-1350, 150), (-160, 240), (-500, 1300), (-1024, 3071), (1000, 1001)], []
                for kind in ("mean", "mip"):
                    c.js(f"(() => {{ const v = window.__viewer, s = v.state().cur; v.setCur([s[0], s[1], {k}]); v.setSlab('{kind}'); return true; }})()")
                    time.sleep(0.15)
                    p = plan(k, nz, dz, MM[kind])
                    want_hu = axial_display(ref_slab(kind, base, p["k0"], p["k1"]))
                    for lo, hi in wins:
                        gg = c.js(f"(() => {{ const v = window.__viewer; v.setWindow({lo}, {hi}, false); return v.shownGray('axial'); }})()")
                        good.append(gg["slab"] == kind and gg["coloured_pixels"] == 0 and np.array_equal(arr8(gg), lut(want_hu, lo, hi)))
                    good.append(np.array_equal(arr16(c.js(f"window.__viewer.slabHU('{kind}', {k})")), want_hu))
                ok(f"{tag}: the drawn picture is the window applied to the slab values, for 5 windows in both views; the slab values are HU and do not change with the window", all(good), f"windows {wins}")
                c.js("window.__viewer.setWindow(-1350, 150, true)")
                # ---- labels and Details ----
                texts = []
                for kind in ("mean", "mip"):
                    for kk in (12, 0, nz - 1):
                        info = c.js(f"(() => {{ const v = window.__viewer, s = v.state().cur; v.setCur([s[0], s[1], {kk}]); v.setSlab('{kind}'); return v.slab(); }})()")
                        time.sleep(0.12)
                        info = c.js("window.__viewer.slab()")
                        p = plan(kk, nz, dz, MM[kind])
                        det = info["details"]
                        if p["n"] == 1 and kind == "mean":
                            want_det = ("the native slice, as stored" if abs(dz - 2.5) <= 0.25 else "A 2.5 mm view cannot be made for this scan")
                        else:
                            want_det = (f"{'MIP 10 mm here: brightest of' if kind == 'mip' else 'Axial 2.5 mm here: mean of'} {p['images']} slice{'' if p['images'] == 1 else 's'} × {g(dz)} mm spacing = {g(p['actual_mm'])} mm "
                                        f"(asked for {MM[kind]:g} mm). Centres of the first and last slice: {g((p['images'] - 1) * dz)} mm apart.")
                        thick_ok = (f"If each slice is 1.5 mm thick, as the radiology report states, the tissue covered is {g((p['images'] - 1) * dz + 1.5)} mm." in det) if (thick and not (p["n"] == 1 and kind == "mean")) \
                            else (thick is not None or "The slice thickness is not recorded" in det or (p["n"] == 1 and kind == "mean"))
                        edge_ok = (not p["truncated"]) or p["n"] == 1 or f"a full slab is {p['n']} slices, {g(p['n'] * dz)} mm" in det
                        texts.append(info["info"] == label(kind, p, dz, nz, kk) and want_det in det and thick_ok and edge_ok and info["first_image_no"] == nz - p["k1"] and info["last_image_no"] == nz - p["k0"])
                        if not texts[-1]:
                            texts.append((kind, kk, info["info"], label(kind, p, dz, nz, kk), want_det))
                src = c.js("document.getElementById('sldetsrc').textContent")
                ok(f"{tag}: one-line label (approximate thickness, slice count, image range, centre) for both views in the middle and at both scan edges; Details give spacing, exact slices x spacing, centre distance, "
                   "the thickness statement and the reconstruction note",
                   all(x is True for x in texts) and f"slices {g(dz)} mm apart (from the file)" in src and "TEST NOTE." in src
                   and (("1.5 mm, as stated in the radiology report (not in the image file)" in src) if thick else ("Slice thickness: not recorded" in src)),
                   f"{label('mean', plan(12, nz, dz, 2.5), dz, nz, 12)}  |  {label('mip', plan(12, nz, dz, 10), dz, nz, 12)}" if all(x is True for x in texts) else str([x for x in texts if x is not True][:2]))
                # ---- scrolling ----
                sc = c.js("(() => { const v = window.__viewer, s = v.state().cur, o = {}; for (const kind of ['mean', 'mip']) { v.setCur([s[0], s[1], 14]); v.setSlab(kind); const a = v.slab(); v.step(1); const b = v.slab(); v.step(-1); v.step(-1); const d = v.slab();"
                          " o[kind] = {a: [a.k0, a.k1, a.centre], b: [b.k0, b.k1, b.centre], d: [d.k0, d.k1, d.centre]}; } return o; })()")
                ok(f"{tag}: scrolling moves one native image per step in both slab views, so neighbouring slabs overlap and no image is skipped",
                   all(sc[kd]["b"][2] == 15 and sc[kd]["d"][2] == 13 and sc[kd]["b"][0] <= sc[kd]["a"][1] + 1 for kd in ("mean", "mip"))
                   and all(plan(kc, nz, dz, MM[kd])["k0"] <= kc <= plan(kc, nz, dz, MM[kd])["k1"] for kd in ("mean", "mip") for kc in range(nz)), str(sc))
                # ---- switching between the three views ----
                sw = c.js("(async () => { const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 130)), s = v.state().cur, q = id => document.getElementById(id); v.setCur([s[0], s[1], 12]); v.setSlab('mean'); await w();"
                          " const o = {cur: [v.state().cur]}; q('slmip').click(); await w(); o.m1 = v.slab().mode; o.shown1 = v.shownGray('axial').slab; o.onMip = q('slmip').classList.contains('on'); o.cur.push(v.state().cur);"
                          " q('sl25').click(); await w(); o.m2 = v.slab().mode; o.shown2 = v.shownGray('axial').slab; o.cur.push(v.state().cur);"
                          " q('sldetbtn').click(); await w(); o.detOpen = !q('sldet').hidden; o.cvDet = q('gl').getBoundingClientRect().height; o.natText = q('slnat').textContent; q('slnat').click(); await w(); await v.idle();"
                          " o.m3 = v.slab().mode; o.shown3 = v.shownGray('axial').slab; o.same3 = v.shownGray('axial').b64 === v.sliceGray('axial', 12).b64; o.body3 = document.body.classList.contains('slabmode'); o.ovbar3 = getComputedStyle(q('ovbar')).display;"
                          " o.cache3 = v.slab().cache_entries; o.detClosed = q('sldet').hidden; o.info3 = q('slinfo').textContent; o.on3 = [q('sl25').classList.contains('on'), q('slmip').classList.contains('on')]; o.cur.push(v.state().cur); o.cvNat = q('gl').getBoundingClientRect().height;"
                          " q('sldetbtn').click(); await w(); o.natText2 = q('slnat').textContent; q('sldetclose').click(); await w(); o.detClosed2 = q('sldet').hidden;"
                          " q('sl25').click(); await w(); o.m4 = v.slab().mode; o.body4 = document.body.classList.contains('slabmode'); o.cvSlab = q('gl').getBoundingClientRect().height; o.cur.push(v.state().cur); return o; })()", 60)
                ok(f"{tag}: switching: 'MIP 10 mm' and 'Axial 2.5 mm' buttons, and native thin slices from Details, each in one click; the image position never changes; native slices are the stored image, "
                   "bring the outline bar back and release the slab buffers; opening Details does not resize the scan",
                   sw["m1"] == "mip" and sw["shown1"] == "mip" and sw["onMip"] and sw["m2"] == "mean" and sw["shown2"] == "mean" and sw["detOpen"] and sw["m3"] == "native" and sw["shown3"] is None and sw["same3"]
                   and not sw["body3"] and sw["ovbar3"] != "none" and sw["cache3"] == 0 and sw["detClosed"] and sw["detClosed2"] and sw["on3"] == [False, False] and sw["m4"] == "mean" and sw["body4"]
                   and all(x == sw["cur"][0] for x in sw["cur"]) and sw["info3"] == f"Native slice  ·  {g(dz)} mm apart  ·  Image {nz - 12}  ·  outlines can be shown"
                   and sw["natText"] == f"Show native thin slices ({g(dz)} mm apart)" and sw["natText2"] == f"Native thin slices are shown ({g(dz)} mm apart)" and abs(sw["cvDet"] - sw["cvSlab"]) < 0.5 and sw["cvSlab"] >= sw["cvNat"] - 0.5,
                   sw["info3"])
                # ---- other planes ----
                ot = c.js("(async () => { const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 130)), q = id => document.getElementById(id), o = {}; v.setSlab('mip'); await w();"
                          " document.querySelector('[data-v=\"coronal\"]').click(); await w(); o.dis = [q('sl25').disabled, q('slmip').disabled]; o.shown = v.shownGray('coronal').slab; o.body = document.body.classList.contains('slabmode');"
                          " o.info = q('slinfo').textContent; o.refused = v.setSlab('mean'); o.cor = v.shownGray('coronal').b64 === (() => { const g = v.sliceGray('coronal', v.state().cur[1]); return g.b64; })();"
                          " await v.setOverlay('m'); o.actCor = v.state().active; document.querySelector('[data-v=\"three\"]').click(); await w(); o.dis3 = q('slmip').disabled; o.shown3 = v.shownGray('axial').slab;"
                          " document.querySelector('[data-v=\"axial\"]').click(); await w(); await v.idle(); await w(); o.back = v.slab().mode; o.shownBack = v.shownGray('axial').slab; o.bodyBack = document.body.classList.contains('slabmode');"
                          " o.actBack = v.state().active; o.colBack = v.shownGray('axial').coloured_pixels; o.disBack = q('slmip').disabled; o.fit = [v.state().zoom, v.state().pan]; v.setSlab('mean'); return o; })()", 60)
                ok(f"{tag}: coronal, sagittal and 3 planes show native images (slab buttons disabled, a slab cannot be forced there, outlines available); back on Axial the chosen slab view returns and an outline switched on meanwhile is released",
                   ot["dis"] == [True, True] and ot["shown"] is None and not ot["body"] and ot["refused"] is False and ot["cor"] and ot["actCor"] == "m" and ot["dis3"] and ot["shown3"] is None
                   and ot["back"] == "mip" and ot["shownBack"] == "mip" and ot["bodyBack"] and ot["actBack"] is None and ot["colBack"] == 0 and not ot["disBack"] and "native slices" in ot["info"], ot["info"])
                # ---- outlines ----
                ov = c.js("(async () => { const v = window.__viewer, s = v.state().cur, q = id => document.getElementById(id), w = () => new Promise(r => setTimeout(r, 130)), o = {}; v.setCur([s[0], s[1], 12]);"
                          " for (const kind of ['mean', 'mip']) { v.setSlab(kind); await w(); await v.setOverlay('m'); o[kind] = {act: v.state().active, col: v.shownGray('axial').coloured_pixels, chip: document.querySelector('#ovbar .ov').offsetParent === null,"
                          " note: getComputedStyle(q('slnote')).display !== 'none', noteH: q('slnote').getBoundingClientRect().height, lineH: q('slinfo').getBoundingClientRect().height}; }"
                          " v.setSlab('native'); await w(); await v.setOverlay('m'); o.nat = {act: v.state().active, col: v.shownGray('axial').coloured_pixels, note: getComputedStyle(q('slnote')).display};"
                          " q('sl25').click(); await w(); await v.idle(); o.after = {act: v.state().active, bytes: v.state().overlayBytes, col: v.shownGray('axial').coloured_pixels};"
                          " v.setSlab('native'); await w(); await v.setOverlay('m'); q('slmip').click(); await w(); await v.idle(); o.after2 = {act: v.state().active, col: v.shownGray('axial').coloured_pixels}; v.setSlab('mean'); return o; })()", 60)
                ok(f"{tag}: outlines exist only on native slices: none can be switched on over the mean slab or the MIP, no coloured pixel is drawn there, an active outline is switched off and released when a slab view is chosen; "
                   "a one-line notice says where outlines are",
                   all(ov[kd]["act"] is None and ov[kd]["col"] == 0 and ov[kd]["chip"] and ov[kd]["note"] and ov[kd]["noteH"] <= 24 and ov[kd]["lineH"] <= 24 for kd in ("mean", "mip"))
                   and ov["nat"]["act"] == "m" and ov["nat"]["col"] > 0 and ov["nat"]["note"] == "none" and ov["after"] == {"act": None, "bytes": 0, "col": 0} and ov["after2"] == {"act": None, "col": 0}
                   and s0["noteText"] == "Outlines: native slices only (Details)", s0["noteText"])
                # ---- zoom chosen by the reader ----
                zz = c.js("(async () => { const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 150)), gl = document.getElementById('gl'), q = id => document.getElementById(id); v.setSlab('native'); await w(); v.zoomBy(2.5, undefined, undefined, 'axial'); await w();"
                          " const cr = v.canvasRect(); v.zoomBy(1.2, cr[0] + cr[2] * 0.3, cr[1] + cr[3] * 0.4, 'axial'); await w(); " + SNAP +
                          " const a = snap(); q('sl25').click(); await w(); const b = snap(); q('slmip').click(); await w(); const b5 = snap(); v.step(1); v.step(-1); await w();"
                          " q('slnat').click(); await w(); const d = snap(); q('slmip').click(); await w(); q('slnat').click(); await w(); const e = snap();"
                          " q('zfit').click(); await w(); const f = snap(); q('sl25').click(); await w(); const gg = snap(); q('slnat').click(); await w(); const h = snap(); v.setSlab('mean'); return {a, b, b5, d, e, f, g: gg, h}; })()", 60)
                same, zdetail = zoom_kept(zz)
                ok(f"{tag}: a zoom chosen by the reader is kept between native slices, Axial 2.5 mm and MIP 10 mm (same screen pixels per image pixel, same image point at the centre) and restored exactly on return; from Fit the image stays at Fit",
                   same and abs(zz["a"]["zoom"] - 3.0) < 1e-9, zdetail.replace("on the single image", "on native slices").replace("in MIP", "in the slab views"))
                # ---- cache bound ----
                cb = c.js("(() => { const v = window.__viewer, s = v.state().cur; let mx = 0; for (let r = 0; r < 6; r++) for (let k = 0; k < v.images('axial'); k++) { v.setCur([s[0], s[1], k]); v.setSlab(r % 2 ? 'mip' : 'mean'); v.shownGray('axial');"
                          " v.slabHU('mean', k); v.slabHU('mip', k); mx = Math.max(mx, v.slab().cache_entries); } const m = v.slab(); v.setSlab('native'); const after = v.slab().cache_entries; v.setSlab('mean');"
                          " return {mx, max: m.cache_max, bytes: m.cache_bytes, limit: m.cache_limit_bytes, scratch: m.scratch_bytes, px: m.w, computed: m.computed, hits: m.hits, reused: m.reused, after, errors: window.__viewerErrors}; })()", 120)
                ok(f"{tag}: slab cache stays inside its limit over {6 * nz * 3} slab requests in both views, one sum buffer is used for the means, and the cache is emptied when native slices are chosen",
                   cb["mx"] <= cb["max"] and cb["bytes"] <= cb["limit"] and cb["after"] == 0 and not cb["errors"] and cb["scratch"] == base.shape[0] * base.shape[1] * 4,
                   f"at most {cb['mx']} of {cb['max']} slabs kept ({cb['bytes']} bytes); computed {cb['computed']}, reused from cache {cb['hits']}")
                if vi == 0:
                    c.js("(() => { const v = window.__viewer; v.setCur([3, 27, 12]); v.setSlab('mean'); return true; })()")
                    time.sleep(0.5)
                    (out / f"synthetic_dz{dz}_axial25.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
            finally:
                b.close()


def run_page(args, out: Path, R: dict, ok) -> None:
    import SimpleITK as sitk
    page = (args.page if args.page.is_absolute() else ROOT / args.page).resolve()
    R["page"], R["file_mb"] = str(page.relative_to(ROOT)).replace("\\", "/"), round(page.stat().st_size / 1e6, 1)
    g0 = gpu_mb()
    b = Browser(args.browser, args.gpu_mode, page.as_uri(), args.port, args.headed)
    R["browser_version"] = b.version
    c = b.c
    try:
        peak, st = wait_ready(b)
        R["timings"]["launch_to_ready_s"] = round(time.time() - b.t0, 1)
        R["memory"]["peak_while_loading"] = peak
        if not st["ready"]:
            ok("scan loaded", False, st["err"])
            return
        time.sleep(2)
        R["memory"]["sustained_after_startup"] = b.gc_mem()
        (out / "startup_axial_2_5mm.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
        c.js(PREP)
        s = c.js(OPEN)
        extra = c.js("(() => { const d = document.documentElement.cloneNode(true); d.querySelectorAll('script.vp').forEach(e => e.remove()); return {webgl: /webgl|niivue/i.test(d.innerHTML), case_id: META.case_id, codes: META.ct.original_axis_codes, slab: META.slab || null,"
                     " renderer: (() => { try { const g = document.createElement('canvas').getContext('webgl'); const e = g.getExtension('WEBGL_debug_renderer_info'); return g.getParameter(e.UNMASKED_RENDERER_WEBGL); } catch (x) { return 'n/a'; } })()}; })()")
        dims, sp = s["st"]["dims"], s["st"]["spacing"]
        nz, dz = dims[2], sp[2]
        k = s["st"]["cur"][2]
        R["startup"] = {"build": s["build"], "case_id": extra["case_id"], "codes": extra["codes"], "fit": s["fit"], "win": s["sl"]["winText"], "dims": dims, "spacing": sp, "slab_meta": extra["slab"], "graphics": extra["renderer"]}
        p25, p10 = plan(k, nz, dz, 2.5), plan(k, nz, dz, 10)
        R["rounding"] = {"axial_2_5": {"images": p25["n"], "mm": round(p25["n"] * dz, 3)}, "mip_10": {"images": p10["n"], "mm": round(p10["n"] * dz, 3)}}
        ok("opened from file://; no WebGL and no viewer library in the file", s["proto"] == "file:" and s["kind"] == "canvas2d" and not extra["webgl"] and s["build"] == args.viewer_build, s["build"])
        ok("opens on 'Axial 2.5 mm': regular axial view, fitted (zoom 100 % of fit, no pan, whole image inside the scan area), lung window W 1500 / L -600, no outline, outline bar absent, Details closed; no image zoom stored between visits",
           s["sl"]["mode"] == "mean" and s["sl"]["shown"] == "mean" and s["body"] and s["on25"] and not s["onMip"] and s["st"]["view"] == "axial" and s["st"]["mode"] == "single" and s["viewOn"] == "axial" and s["winOn"] == "Lung"
           and s["fit"]["zoom"] == 1 and s["fit"]["pan"] == [0, 0] and s["fit"]["inside"] and 0.93 < s["fit"]["fills"] <= 0.95 and not s["saved"] and s["sl"]["winText"] == "W 1500  ·  L -600"
           and s["st"]["active"] is None and s["chipsOn"] == 0 and s["ovbar"] == "none" and s["detHidden"] and not s["errors"] and s["sl"]["info"] == label("mean", p25, dz, nz, k),
           f"{s['sl']['info']}; image fills {s['fit']['fills']:.2f} of the scan area")
        ok("simple controls: 'Axial 2.5 mm', 'MIP 10 mm' and 'Details' only; no 5 mm or 8 mm option",
           s["btns"] == ["Axial 2.5 mm", "MIP 10 mm", "Details"] and c.js("document.querySelectorAll('[data-mip]').length") == 0 and not any(x.strip() in ("5 mm", "8 mm") for x in s["btns"]), str(s["btns"]))
        # ---- both slabs against the ORIGINAL file, read with SimpleITK ----
        case = extra["case_id"]
        img = sitk.ReadImage(str(ROOT / "data" / "raw" / "ct_rate" / "volumes" / f"{case}.nii.gz"))
        orig = sitk.GetArrayFromImage(img)                        # [z, y, x] in file index order
        file_dz = float(img.GetSpacing()[2])
        w, h = dims[0], dims[1]
        worst, n_cmp, rows, sl_cache = {"mean": 0, "mip": 0}, 0, [], {}

        def native(z):
            if z not in sl_cache:
                gi = grid_from_sitk(img, b, "axial", z, w, h)
                sl_cache[z] = np.clip(orig[gi[..., 2], gi[..., 1], gi[..., 0]], -1024, 3071).astype(np.int16)      # the page's prepared display range
            return sl_cache[z]
        for kind in ("mean", "mip"):
            for kc in sorted({0, 1, 2, nz // 4, nz // 2, (3 * nz) // 4, nz - 3, nz - 2, nz - 1}):
                d = c.js(f"window.__viewer.slabHU('{kind}', {kc})")
                p = plan(kc, nz, dz, MM[kind])
                stack = np.stack([native(z) for z in range(p["k0"], p["k1"] + 1)], axis=2)
                ref = stack.max(axis=2) if kind == "mip" else np.floor(stack.astype(np.int64).sum(axis=2) / float(p["images"]) + 0.5).astype(np.int16)
                diff = int(np.abs(arr16(d).astype(np.int32) - ref.astype(np.int32)).max())
                worst[kind] = max(worst[kind], diff)
                n_cmp += 1
                same_plan = (d["k0"], d["k1"], d["images"]) == (p["k0"], p["k1"], p["images"]) and abs(d["actual_mm"] - p["actual_mm"]) < 1e-6
                rows.append({"view": kind, "centre_index": kc, "k0": d["k0"], "k1": d["k1"], "images": d["images"], "actual_mm": round(d["actual_mm"], 3), "max_abs_difference_hu": diff, "plan_ok": same_plan})
        R["slab_vs_original_file"] = rows
        ok(f"both slab views equal a direct calculation over the same images of the ORIGINAL file read with SimpleITK ({n_cmp} slabs: 9 centre images each, including both scan edges): "
           "mean = floor(sum / n + 0.5), MIP = maximum", worst == {"mean": 0, "mip": 0} and all(r["plan_ok"] for r in rows) and abs(file_dz - dz) < 1e-4,
           f"largest difference: mean {worst['mean']} HU, MIP {worst['mip']} HU; spacing {dz:g} mm in the page and {file_dz:g} mm in the file; 2.5 mm -> {p25['n']} images = {g(p25['n'] * dz)} mm; 10 mm -> {p10['n']} images = {g(p10['n'] * dz)} mm")
        # ---- source thickness / spacing statements against the data set ----
        gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"][case]
        m = re.search(r"section thickness of ([0-9.]+) mm", gt.get("technique") or "")
        thick = float(m.group(1)) if m else None
        det = c.js("(() => { const q = id => document.getElementById(id); q('sldetbtn').click(); const o = {open: !q('sldet').hidden, src: q('sldetsrc').textContent, d25: q('sldet25').textContent, dmip: q('sldetmip').textContent, all: q('sldet').textContent,"
                   " cv: q('gl').getBoundingClientRect().height}; q('sldetclose').click(); o.closed = q('sldet').hidden; o.cv2 = q('gl').getBoundingClientRect().height; return o; })()")
        want25 = f"Axial 2.5 mm here: mean of {p25['images']} slices × {g(dz)} mm spacing = {g(p25['actual_mm'])} mm (asked for 2.5 mm). Centres of the first and last slice: {g((p25['images'] - 1) * dz)} mm apart."
        want10 = f"MIP 10 mm here: brightest of {p10['images']} slices × {g(dz)} mm spacing = {g(p10['actual_mm'])} mm (asked for 10 mm). Centres of the first and last slice: {g((p10['images'] - 1) * dz)} mm apart."
        t_ok = (f"Slice thickness: {g(thick)} mm, as stated in the radiology report (not in the image file)" in det["src"]
                and f"the tissue covered is {g((p25['images'] - 1) * dz + thick)} mm" in det["d25"] and f"the tissue covered is {g((p10['images'] - 1) * dz + thick)} mm" in det["dmip"]) if thick \
            else ("Slice thickness: not recorded" in det["src"] and "not known exactly" in det["d25"])
        R["source"] = {"slice_spacing_mm_from_file": file_dz, "section_thickness_mm_stated_in_report": thick, "details_source_line": det["src"]}
        ok("Details: slice spacing and slice thickness are stated separately (spacing from the file; thickness only as stated in the report, or 'not recorded'); exact slices x spacing and centre distance for both views; "
           "no 2.5 mm reconstruction in the data set, so the view is calculated; opening Details does not resize the scan",
           det["open"] and det["closed"] and want25 in det["d25"] and want10 in det["dmip"] and t_ok and f"slices {g(dz)} mm apart (from the file)" in det["src"]
           and "No 2.5 mm reconstruction of this scan exists in the data set" in det["src"] and "it is not a reconstruction from the scanner" in det["all"] and "not the slice thickness" in det["all"] and abs(det["cv"] - det["cv2"]) < 0.5,
           det["src"])
        # ---- buttons, labels, drawn pixels, native slices ----
        ui = c.js(f"(async () => {{ const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 200)), q = id => document.getElementById(id), o = {{}}; const nat = v.sliceGray('axial', {k}).b64;"
                  " o.mean = {drawn: v.shownGray('axial'), loc: q('loc').textContent, cv: q('gl').getBoundingClientRect().height}; o.meanDiffers = o.mean.drawn.b64 !== nat;"
                  " q('slmip').click(); await w(); o.mip = {sl: v.slab(), drawn: v.shownGray('axial'), loc: q('loc').textContent, body: document.body.classList.contains('slabmode'), cv: q('gl').getBoundingClientRect().height}; o.mipDiffers = o.mip.drawn.b64 !== o.mean.drawn.b64;"
                  " q('sldetbtn').click(); await w(); q('slnat').click(); await w(); await v.idle(); o.nat = {sl: v.slab(), same: v.shownGray('axial').b64 === nat, ovbar: getComputedStyle(q('ovbar')).display, chip: getComputedStyle(document.querySelector('#ovbar .ov')).visibility,"
                  " cv: q('gl').getBoundingClientRect().height, cur: v.state().cur, det: q('sldet').hidden, note: getComputedStyle(q('slnote')).display};"
                  " q('sl25').click(); await w(); o.back = {mode: v.slab().mode, cur: v.state().cur, same: v.shownGray('axial').b64 === o.mean.drawn.b64}; return o; })()", 60)
        ok("'MIP 10 mm' shows the maximum slab with its label; 'Axial 2.5 mm' returns to the mean slab; both differ from the native slice and from each other; the read-out under the scan names the slab value and the centre slice value",
           ui["meanDiffers"] and ui["mipDiffers"] and ui["mean"]["drawn"]["slab"] == "mean" and ui["mip"]["drawn"]["slab"] == "mip" and ui["mip"]["drawn"]["coloured_pixels"] == 0 and ui["mean"]["drawn"]["coloured_pixels"] == 0
           and ui["mip"]["sl"]["info"] == label("mip", p10, dz, nz, k) and "MIP: brightest in slab" in ui["mip"]["loc"] and "mean of slab" in ui["mean"]["loc"] and "centre slice" in ui["mean"]["loc"]
           and ui["back"]["mode"] == "mean" and ui["back"]["same"] and ui["back"]["cur"][2] == k, ui["mip"]["sl"]["info"])
        ok("native thin slices are one click inside Details: the stored image at the same position, outline bar back, slab buffers released; the scan area is taller in the slab views (no empty strip)",
           ui["nat"]["sl"]["mode"] == "native" and ui["nat"]["same"] and ui["nat"]["ovbar"] != "none" and ui["nat"]["chip"] == "visible" and ui["nat"]["cur"][2] == k and ui["nat"]["det"] and ui["nat"]["note"] == "none"
           and ui["nat"]["sl"]["cache_entries"] == 0 and ui["mean"]["cv"] > ui["nat"]["cv"] and abs(ui["mean"]["cv"] - ui["mip"]["cv"]) < 0.5,
           f"scan area {ui['nat']['cv']:.0f} px high on native slices, {ui['mean']['cv']:.0f} px in the slab views")
        # ---- zoom ----
        zz = c.js("(async () => { const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 200)), gl = document.getElementById('gl'), q = id => document.getElementById(id); v.setSlab('native'); await w(); q('zin').click(); q('zin').click(); await w();"
                  " const cr = v.canvasRect(); v.zoomBy(1.3, cr[0] + cr[2] * 0.35, cr[1] + cr[3] * 0.4, 'axial'); await w(); " + SNAP +
                  " const a = snap(); q('sl25').click(); await w(); const b = snap(); q('slmip').click(); await w(); const b5 = snap(); q('slnat').click(); await w(); const d = snap(); q('slmip').click(); await w(); q('slnat').click(); await w(); const e = snap();"
                  " q('zfit').click(); await w(); const f = snap(); q('sl25').click(); await w(); const gg = snap(); q('slnat').click(); await w(); const h = snap(); v.setSlab('mean'); await w(); return {a, b, b5, d, e, f, g: gg, h, end: snap()}; })()", 60)
        same, zdetail = zoom_kept(zz)
        R["zoom_preserved"] = {"native": {kk: zz["a"][kk] for kk in ("zoom", "pan", "px", "cv")}, "slab": {kk: zz["b"][kk] for kk in ("zoom", "pan", "px", "cv")}, "back": {kk: zz["d"][kk] for kk in ("zoom", "pan", "px", "cv")}}
        ok("a zoom chosen by the reader is kept between native slices, Axial 2.5 mm and MIP 10 mm (same screen pixels per image pixel, same image point at the centre) and restored exactly on return; from Fit the image stays at Fit",
           same and zz["a"]["zoom"] > 1.9 and zz["end"]["zoom"] == 1, zdetail.replace("on the single image", "on native slices").replace("in MIP", "in the slab views"))
        # ---- real wheel scrolling ----
        rect = c.js("window.__viewer.canvasRect()")
        cx, cy = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
        c.js("(() => { window.__paints = 0; const tick = () => { window.__paints++; requestAnimationFrame(tick); }; requestAnimationFrame(tick); return true; })()")
        n_wheel = args.wheel_notches
        net = sum(1 if (i // 60) % 2 == 0 else -1 for i in range(n_wheel))
        for kind in ("mean", "mip"):
            c.js(f"(() => {{ const v = window.__viewer, s = v.state().cur; v.setCur([s[0], s[1], {k}]); v.setSlab('{kind}'); return true; }})()")
            for _ in range(5):
                c.call("Input.dispatchMouseEvent", type="mouseWheel", x=cx, y=cy, deltaX=0, deltaY=100)
                time.sleep(0.05)
            time.sleep(0.3)
            wm = c.js("window.__viewer.slab()")
            pk = plan(k - 5, nz, dz, MM[kind])
            t_w = time.time()
            for i in range(n_wheel):
                c.call("Input.dispatchMouseEvent", type="mouseWheel", x=cx, y=cy, deltaX=0, deltaY=100 if (i // 60) % 2 == 0 else -100)
            time.sleep(0.4)
            wheel_s = time.time() - t_w - 0.4
            ww = c.js("(() => { const v = window.__viewer; return {m: v.slab(), errors: window.__viewerErrors, ready: v.ready}; })()")
            R["timings"][f"wheel_ms_per_notch_{kind}_including_test_driver"] = round(1000 * wheel_s / n_wheel, 1)
            name = "Axial 2.5 mm" if kind == "mean" else "MIP 10 mm"
            ok(f"mouse wheel in {name}: one notch = one native slice (5 notches down = 5 slices towards the feet), slabs overlap; then {n_wheel} notches (direction reversed every 60) drawn on screen without error, cache inside its limit",
               wm["centre"] == k - 5 and wm["mode"] == kind and (wm["k0"], wm["k1"]) == (pk["k0"], pk["k1"]) and not ww["errors"] and ww["ready"] and ww["m"]["cache_entries"] <= ww["m"]["cache_max"] and ww["m"]["centre"] == k - 5 - net,
               f"{R['timings'][f'wheel_ms_per_notch_{kind}_including_test_driver']} ms per notch including the test driver; cache {ww['m']['cache_entries']} of {ww['m']['cache_max']} slabs")
            if kind == "mip":
                (out / "mip_10mm.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
        R["timings"]["wheel_notches_per_view"] = n_wheel
        # ---- "Show in scan" from a slab view ----
        jump = c.js("(async () => { const v = window.__viewer; const tr = Array.from(document.querySelectorAll('tr.jump')).find(t => t.dataset.jump && !document.body.classList.contains('stage1')); if (!tr) return null;"
                    " v.setSlab('mean'); await new Promise(r => setTimeout(r, 150)); tr.click(); for (let i = 0; i < 100 && v.state().active !== tr.dataset.jump; i++) await new Promise(r => setTimeout(r, 100)); await v.idle();"
                    " const o = {key: tr.dataset.jump, mode: v.slab().mode, active: v.state().active, col: v.shownGray('axial').coloured_pixels, slab: v.shownGray('axial').slab};"
                    " document.getElementById('slmip').click(); await new Promise(r => setTimeout(r, 200)); await v.idle(); o.after = [v.state().active, v.shownGray('axial').coloured_pixels, v.slab().mode]; v.setSlab('mean'); return o; })()", 60)
        if jump:
            ok("'Show in scan' pressed in a slab view: the page switches to native slices and shows the outline there, never over a slab; choosing a slab view again switches the outline off",
               jump["mode"] == "native" and jump["active"] == jump["key"] and jump["col"] > 0 and jump["slab"] is None and jump["after"] == [None, 0, "mip"], f"outline {jump['key']}, {jump['col']} outlined pixels")
        # ---- repeated scrolling, window and view changes; memory ----
        t0 = time.time()
        st2 = c.js(f"(async () => {{ const v = window.__viewer, s = v.state().cur, n = v.images('axial'); v.setSlab('mean'); let mx = 0, k = {k}, dir = 1, t = 0, tm = 0; const frame = () => new Promise(r => requestAnimationFrame(() => r()));"
                   f" for (let i = 0; i < {args.scroll_steps}; i++) {{ k += dir; if (k >= n - 1 || k <= 0) dir = -dir; v.setCur([s[0], s[1], k]);"
                   " if (i % 3 === 0) { const c = -1400 + (i * 37) % 1800, w = 200 + (i * 53) % 1800; v.setWindow(c - w / 2, c + w / 2, false); }"
                   " if (i % 25 === 0) v.setSlab(['mean', 'mip'][(i / 25) % 2]); if (i % 60 === 59) { v.setSlab('native'); v.shownGray('axial'); v.setSlab('mean'); }"
                   " const a = performance.now(); v.shownGray('axial'); const dt = performance.now() - a; if (v.slab().mode === 'mean') tm = Math.max(tm, dt); else t = Math.max(t, dt); mx = Math.max(mx, v.slab().cache_entries); if (i % 4 === 0) await frame(); }"
                   " const m = v.slab(); v.setWindow(-1350, 150, true); v.setSlab('mean'); return {mx, max: m.cache_max, bytes: m.cache_bytes, limit: m.cache_limit_bytes, computed: m.computed, hits: m.hits, reused: m.reused, slowest_mip_ms: Math.round(t), slowest_mean_ms: Math.round(tm),"
                   " errors: window.__viewerErrors, ready: v.ready}; })()", 600)
        R["timings"]["stress_s"] = round(time.time() - t0, 1)
        R["timings"]["slowest_slab_draw_ms"] = {"axial_2_5": st2["slowest_mean_ms"], "mip_10": st2["slowest_mip_ms"]}
        ok(f"{args.scroll_steps} scroll steps with {args.scroll_steps // 3} window changes and repeated switches between Axial 2.5 mm, MIP 10 mm and native slices: no error, slab cache inside its limit",
           st2["mx"] <= st2["max"] and st2["bytes"] <= st2["limit"] and not st2["errors"] and st2["ready"],
           f"at most {st2['mx']} of {st2['max']} slabs kept ({st2['mx'] * dims[0] * dims[1] * 2 / 1e6:.1f} of {st2['limit'] / 1e6:.0f} MB); slabs computed {st2['computed']}; slowest slab + colouring: mean {st2['slowest_mean_ms']} ms, MIP {st2['slowest_mip_ms']} ms")
        in_slab = b.gc_mem()
        base_m = R["memory"]["sustained_after_startup"]
        R["memory"]["sustained_after_stress"] = in_slab
        R["memory"]["retained_growth"] = {kk: in_slab[kk] - base_m[kk] for kk in ("private_mb", "working_set_mb", "renderer_ws_mb", "gpu_process_ws_mb")}
        R["memory"]["gpu_used_delta_mb"] = None if g0 is None else gpu_mb() - g0
        ok("memory after the wheel and stress blocks (the page stays on Axial 2.5 mm): retained growth below 150 MB private; the page still works",
           R["memory"]["retained_growth"]["private_mb"] < 150 and c.js("window.__viewer.ready") and not c.js("window.__viewerErrors.length"),
           f"retained growth {R['memory']['retained_growth']['private_mb']} MB private; after stress {in_slab['private_mb']} MB, start-up {base_m['private_mb']} MB")
        # ---- AI reader lines (if the page has them) ----
        rd = c.js("(() => { const l = Array.from(document.querySelectorAll('.aird')).map(e => e.textContent), card = document.getElementById('aireader'); return {lines: l, card: card ? {open: card.querySelector('details').open, text: card.textContent} : null,"
                  " tips: Array.from(document.querySelectorAll('.aird')).map(e => e.title), used: Array.from(document.querySelectorAll('.aird.aiused')).map(e => e.closest('tr').querySelector('td b').textContent),"
                  " badges: Array.from(document.querySelectorAll('#chk span.st')).map(e => [e.textContent, e.title || '', e.closest('tr').querySelector('td b').textContent])}; })()")
        taken = [b for b in rd["badges"] if b[1].startswith("Answer taken from AI reader")]
        R["ai_reader"] = {"lines": len(rd["lines"]), "card": bool(rd["card"]), "used_for": rd["used"], "badge_tooltips": [b[1] for b in taken]}
        n_used = len(rd["used"])
        if rd["card"]:
            ok("AI reader: one small line under each of the 18 reader conditions, marked 'shown only, not counted' or, where the policy takes its answer, 'used for this answer'; "
               "every answer taken from the reader is 'present', is one of the conditions named by the policy, and carries a tooltip naming the reader and its score; the report is in a closed card that says which answers use the reader",
               len(rd["lines"]) == 18 and all(x.startswith("AI reader ") and (("(shown only, not counted)" in x) != ("(used for this answer)" in x)) for x in rd["lines"])
               and sum("(used for this answer)" in x for x in rd["lines"]) == n_used == len(taken) and all(b[0] == "present" and re.search(r"score for 'present': [01]\.\d\d \(0 = no, 1 = yes; uncalibrated", b[1]) for b in taken)
               and sorted(b[2] for b in taken) == sorted(rd["used"]) and all(u in rd["card"]["text"] for u in rd["used"]) and not rd["card"]["open"] and "FINDINGS" in rd["card"]["text"]
               and (("is used as the tool" in rd["card"]["text"]) if ("policy decision, not validated" in rd["card"]["text"]) else (n_used == 0 and "do not change any answer of the tool" in rd["card"]["text"]))
               and all(t == "" or "score for 'present'" in t or "no score stored" in t for t in rd["tips"]),
               f"answers taken from the reader: {rd['used'] or 'none'}; " + (taken[0][1] if taken else rd["lines"][0]))
        # ---- grading and CSV export ----
        n_used = len(R.get("ai_reader", {}).get("used_for", []))
        if c.js("!!document.getElementById('gsave')"):
            gq = c.js("(async () => { const v = window.__viewer; v.setSlab('mip'); const rev = document.getElementById('grev'); const prefilled = !!document.querySelector('#chk td.grade .gb button.sel'); rev.value = window.__TEST_REVIEWER; rev.oninput();"
                      " const rows = Array.from(document.querySelectorAll('#chk tr')).filter(r => r.querySelector('td.grade'));"
                      " rows.forEach((tr, i) => { const td = tr.querySelector('td.grade'); const want = td.querySelector('.gb button.sel') || td.querySelector('.gb button[data-a=\"' + ['present', 'absent', 'cannot_tell'][i % 3] + '\"]'); if (!want.classList.contains('sel')) want.click(); });"
                      " const lock = document.getElementById('glock'); if (lock) lock.click(); await new Promise(r => setTimeout(r, 200));"
                      " document.getElementById('gsave').click(); await new Promise(r => setTimeout(r, 1200));"
                      " return {rows: rows.length, csv: window.__csv, graded: document.getElementById('gcount').textContent, prefilled, mode: v.slab().mode}; })()", 120)
            csv_text = (gq["csv"] or "").lstrip("﻿")
            lines = [ln for ln in csv_text.split("\r\n") if ln]
            (out / "exported_grades_AUTOMATED_TEST.csv").write_text(csv_text, encoding="utf-8", newline="")
            ok("grading in a slab view: every condition graded and autosaved", gq["graded"].startswith(f"{gq['rows']} of {gq['rows']}") and gq["mode"] == "mip", gq["graded"])
            ok("CSV export: header, one row per condition and the overall row; rows name this viewer build; the AI reader appears only as the decision basis of the answers taken from it",
               len(lines) == gq["rows"] + 2 and lines[0].startswith("case_id,reviewer,saved_utc,condition_key") and args.viewer_build in lines[1] and "AUTOMATED TEST" in lines[1] and csv_text.count("NV-Reason") == csv_text.count('"AI reader NV-Reason-CT"') == n_used,
               f"{len(lines)} lines; prefilled form: {gq['prefilled']}; rows whose decision basis is the AI reader: {n_used}")
        else:
            ok("grading form present", False, "no grading form on this page")
        c.js("window.__viewer.setSlab('mean')")
    finally:
        b.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--browser", choices=list(BROWSERS), default="edge")
    ap.add_argument("--gpu-mode", choices=list(GPU_FLAGS), default="igpu")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--scroll-steps", type=int, default=600)
    ap.add_argument("--wheel-notches", type=int, default=300)
    ap.add_argument("--label", default="")
    ap.add_argument("--port", type=int, default=9621)
    ap.add_argument("--viewer-build", default=v6.VIEWER_BUILD,
                    help="--page only: build name the page must carry. Give the v7 name to run these v6 checks on a v7 page (v7 must behave as v6 does)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not args.synthetic and not args.page:
        ap.error("use --synthetic or --page")
    name = "synthetic" if args.synthetic else args.page.stem
    out = ROOT / "runs" / f"V11_slab_test_{args.label + '_' if args.label else ''}{name}_{args.browser}_{'sw' if args.synthetic else args.gpu_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    R = {"kind": "technical checks only; not clinician usability feedback", "browser": args.browser, "gpu_mode": "sw" if args.synthetic else args.gpu_mode, "viewer_build": args.viewer_build,
         "checks": [], "memory": {}, "timings": {}}

    def ok(name, cond, detail=None):
        R["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
        print(("PASS " if cond else "FAIL ") + name + (f"  [{str(detail)[:300]}]" if detail is not None else ""), flush=True)
    try:
        (run_synthetic if args.synthetic else run_page)(args, out, R, ok)
    except Exception as e:  # reported, never dropped
        import traceback
        R["error"] = f"{type(e).__name__}: {e}"
        R["traceback"] = traceback.format_exc()
        print("ERROR", R["error"], flush=True)
        print(R["traceback"], flush=True)
    R["passed"] = "error" not in R and bool(R["checks"]) and all(x["pass"] for x in R["checks"])
    (out / "result.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    md = [f"# Viewer v6 (Axial 2.5 mm / MIP 10 mm) technical checks: {name} in {R.get('browser_version', args.browser)} - {'PASSED' if R['passed'] else 'FAILED'}", "",
          "Technical checks only. Not clinician usability feedback.", "", "| check | result | detail |", "|---|---|---|"]
    md += [f"| {x['name']} | {'pass' if x['pass'] else 'FAIL'} | {str(x['detail'] or '')[:300].replace('|', '/')} |" for x in R["checks"]]
    md += ["", "Timings: " + json.dumps(R["timings"]), "", "Memory (MB): " + json.dumps(R["memory"])]
    if "error" in R:
        md += ["", "ERROR: " + R["error"]]
    (out / "result.md").write_text("\n".join(md), encoding="utf-8")
    print(f"-> {out}   {'PASSED' if R['passed'] else 'FAILED'}  ({sum(x['pass'] for x in R['checks'])} of {len(R['checks'])} checks)")
    return 0 if R["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
