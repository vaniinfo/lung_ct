"""Technical checks of the axial sliding-slab MIP mode (viewer v5), run against the UNMODIFIED html from file:// (core env).

These are technical checks only. Whether the mode is useful for reading is a question for the clinician reviewer
and is not answered here.

  --synthetic   small volumes with KNOWN voxel maxima, stored in several axis orders / flips and with four slice
                spacings (0.75, 1.25, 2.5 and 6 mm). For every spacing, slab option and centre image (including both scan
                edges) the page's slab must equal the maximum computed here with numpy from the patient-oriented volume,
                and the page's stated thickness and image range must equal the arithmetic written out in this file.
                Also: start-up defaults, window independence, orientation, toggling back, other views, outlines, cache bound.
  --page FILE   a real case page: start-up defaults, the slab compared with SimpleITK reading the ORIGINAL file
                (independent of the page builder), buttons, return to the centre image, repeated scrolling / window /
                thickness changes with memory measured as in test_canvas_viewer.py, then grading and CSV export.

Usage: python scripts/test_mip_viewer.py --synthetic --browser edge
       python scripts/test_mip_viewer.py --page runs/M9_.../viewer_valid_301_M9_v3_10.html --browser edge --gpu-mode igpu
Writes runs/V7_mip_test_<label>_<time>/{result.json,result.md,*.png}.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import viewer_v5 as v5  # noqa: E402
from test_canvas_viewer import GPU_FLAGS, Browser, arr8, arr16, grid_from_sitk, variant, wait_ready  # noqa: E402
from test_viewer_page import BROWSERS, PREP, gpu_mb  # noqa: E402

OPTIONS = (5, 8, 10)


def plan(k: int, nz: int, dz: float, mm: float) -> dict:
    """The slab rule, written out here independently of scripts/viewer_v5.py (k: larger = more superior)."""
    n = int(mm / dz + 0.5)                      # nearest whole number of images, halves up
    if n < 1:
        n = 1
    lo = k - (n - 1) // 2                       # with an even count the extra image is on the superior side
    hi = lo + n - 1
    k0, k1 = max(lo, 0), min(hi, nz - 1)        # shortened at the scan edge, never shifted
    return {"n": n, "k0": k0, "k1": k1, "images": k1 - k0 + 1, "actual_mm": (k1 - k0 + 1) * dz, "truncated": (k0, k1) != (lo, hi)}


def label_mm(p: dict, asked: float) -> str:
    """The figure after the approximately-equal sign: the thickness asked for, or slices x spacing if the slab is shortened or a single slice."""
    if p["truncated"] or p["n"] == 1:
        return f"{int(p['actual_mm'] * 10 + 0.5) / 10:g}"
    return f"{asked:g}"


def zoom_kept(zz: dict) -> tuple[bool, str]:
    """a = reader's zoom on the single image; b, b5 = in MIP; d, e = back on the single image; g = MIP started from Fit; f = Fit."""
    def centre(x):       # image centre relative to the centre of the scan area (CSS px)
        return (x["L"][0] + x["L"][2] / 2 - (x["cr"][0] + x["cr"][2] / 2), x["L"][1] + x["L"][3] / 2 - (x["cr"][1] + x["cr"][3] / 2))
    a = zz["a"]
    same_scale = all(abs(zz[k]["px"][i] - a["px"][i]) < 1e-6 * a["px"][i] for k in ("b", "b5") for i in (0, 1))
    same_place = all(abs(centre(zz[k])[i] - centre(a)[i]) < 0.51 for k in ("b", "b5") for i in (0, 1))
    back_exact = all({kk: zz[k][kk] for kk in ("zoom", "pan", "px", "L", "cv")} == {kk: a[kk] for kk in ("zoom", "pan", "px", "L", "cv")} for k in ("d", "e"))
    fit = zz["f"]["zoom"] == 1 and zz["f"]["pan"] == [0, 0] and zz["g"]["zoom"] == 1 and zz["g"]["pan"] == [0, 0] and zz["g"]["inside"] and zz["h"]["zoom"] == 1 and zz["h"]["pan"] == [0, 0]
    taller = zz["b"]["cv"][1] >= a["cv"][1]
    good = same_scale and same_place and back_exact and fit and taller and a["pan"] != [0, 0]
    return good, (f"zoom {a['zoom']:.2f} of fit; 1 image pixel = {a['px'][0]:.3f} screen pixels on the single image and {zz['b']['px'][0]:.3f} in MIP; scan area {a['cv'][1]} px high, {zz['b']['cv'][1]} px in MIP"
                  if good else f"scale {same_scale} place {same_place} back {back_exact} fit {fit} taller {taller} {zz}")


def axial_display(a2: np.ndarray) -> np.ndarray:
    """Canonical [i -> Right, j -> Anterior] image as it must be drawn: left = Right (high i first), top = Anterior (high j first)."""
    return a2[::-1, ::-1].T


def lut(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = (vals.astype(np.float64) - lo) * (255.0 / max(1.0, hi - lo))
    return np.where(g <= 0, 0, np.where(g >= 255, 255, np.floor(g + 0.5))).astype(np.uint8)


def phantom(dz: float):
    """RAS+ volume (axis 0 -> Right, 1 -> Anterior, 2 -> Superior) with a texture and isolated bright voxels of known value."""
    shape = (40, 32, 30)
    i, j, k = np.indices(shape)
    a = ((i * 7 + j * 13 + k * 29) % 900 - 1000).astype(np.int16)         # background -1000..-101, differs between neighbouring images
    marks = {(3, 27, 12): 2500, (30, 5, 12): 1800, (20, 16, 0): 2100, (20, 16, 29): 2200, (8, 8, 20): 900, (9, 8, 21): 1500}
    for (x, y, z), v in marks.items():           # (3, 27, 12): patient LEFT (low i), ANTERIOR (high j)
        a[x, y, z] = v
    a[12:15, 12:15, 5:9] = -1024                                           # a dark block: a MIP must not be pulled down by it
    mask = np.zeros(shape, np.uint8)
    mask[10:20, 10:20, 10:14] = 1
    aff = np.diag((0.8, 1.3, dz, 1.0))
    aff[:3, 3] = (-37.0, 12.5, -140.0)
    return a, mask, aff, marks


VARIANTS = [((0, 1, 2), (0, 0, 0)), ((0, 1, 2), (1, 1, 0)), ((0, 1, 2), (0, 0, 1)), ((1, 0, 2), (0, 1, 1)), ((2, 1, 0), (1, 0, 0)), ((1, 2, 0), (1, 1, 1))]


def run_synthetic(args, out: Path, R: dict, ok) -> None:
    tmp = out / "pages"
    tmp.mkdir()
    port = args.port
    for dz in (0.75, 1.25, 2.5, 6.0):
        base, mask, aff, marks = phantom(dz)
        nz = base.shape[2]
        for vi, (perm, flips) in enumerate(VARIANTS if dz == 0.75 else VARIANTS[:2]):
            arr, a2 = variant(base, aff, perm, flips)
            m2, _ = variant(mask, aff, perm, flips)
            page = tmp / f"mip_dz{dz}_v{vi}.html"
            page.write_text(v5.standalone_page(f"mip dz {dz} variant {vi}", arr, a2, {"m": ("test outline", "#ef4444", m2)}), encoding="utf-8")
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
                s0 = c.js("(() => { const v = window.__viewer; return {st: v.state(), mip: v.mip(), body: document.body.classList.contains('mipmode'), on10: document.querySelector('[data-mip=\"10\"]').classList.contains('on'),"
                          " single: document.getElementById('mipoff').classList.contains('on'), mipBtn: document.getElementById('mipon').classList.contains('on'), errors: window.__viewerErrors, build: v.build,"
                          " same: (() => { const a = v.shownGray('axial').b64, b = v.sliceGray('axial', v.state().cur[2]).b64; return a === b; })(),"
                          " fit: (() => { const L = v.layout('axial'), r = v.canvasRect(); return {zoom: v.state().zoom, pan: v.state().pan, inside: L.x >= r[0] - 0.5 && L.y >= r[1] - 0.5 && L.x + L.w <= r[0] + r[2] + 0.5 && L.y + L.h <= r[1] + r[3] + 0.5,"
                          " fills: Math.max(L.w / r[2], L.h / r[3])}; })(), win: document.getElementById('winval').textContent, helpHidden: document.getElementById('miphelp').hidden,"
                          " centreBtn: getComputedStyle(document.getElementById('mipcentre')).visibility}; })()")
                ok(f"{tag}: first opening uses Fit: zoom 100 % of fit, no pan, the whole image inside the scan area and touching it in one direction; window read-out shows the lung preset; help closed",
                   s0["fit"]["zoom"] == 1 and s0["fit"]["pan"] == [0, 0] and s0["fit"]["inside"] and 0.93 < s0["fit"]["fills"] <= 0.95 and s0["win"] == "W 1500  ·  L -600" and s0["helpHidden"] and s0["centreBtn"] == "hidden",
                   f"image fills {s0['fit']['fills']:.2f} of the scan area; {s0['win']}")
                ok(f"{tag}: opens on a single axial image, MIP off, 10 mm selected, no outline, no error",
                   s0["st"]["view"] == "axial" and s0["st"]["mode"] == "single" and not s0["mip"]["on"] and not s0["body"] and s0["on10"] and s0["single"] and not s0["mipBtn"]
                   and s0["st"]["active"] is None and not s0["errors"] and s0["same"] and s0["build"] == v5.VIEWER_BUILD and s0["mip"]["requested_mm"] == 10, s0["build"])
                ok(f"{tag}: slice spacing read from the affine", abs(s0["mip"]["spacing_mm"] - dz) < 1e-9 and s0["mip"]["images_total"] == nz, s0["mip"]["spacing_mm"])
                # ---- known maxima: every option, centres including both edges ----
                bad, n_cmp, seen = [], 0, set()
                centres = sorted({0, 1, 2, 5, 11, 12, 13, 20, nz - 3, nz - 2, nz - 1})
                for mm in OPTIONS:
                    for k in centres:
                        d = c.js(f"window.__viewer.mipHU({k}, {mm})")
                        p, pv = plan(k, nz, dz, mm), v5.slab_plan(k, nz, dz, mm)
                        want = axial_display(base[:, :, p["k0"]:p["k1"] + 1].max(axis=2))
                        got = arr16(d)
                        n_cmp += 1
                        seen.add((p["n"], p["truncated"]))
                        same_plan = (d["k0"], d["k1"], d["images"], d["planned_images"], d["truncated"]) == (p["k0"], p["k1"], p["images"], p["n"], p["truncated"]) \
                            and abs(d["actual_mm"] - p["actual_mm"]) < 1e-9 and (pv["k0"], pv["k1"], pv["planned_images"]) == (p["k0"], p["k1"], p["n"])
                        if not same_plan or got.shape != want.shape or not np.array_equal(got, want):
                            bad.append((mm, k, d["k0"], d["k1"], p))
                ok(f"{tag}: slab maxima equal the known maxima for {n_cmp} slabs (5 / 8 / 10 mm; centre images at both scan edges and in between); image counts and thickness follow the stated rule",
                   not bad, f"image counts seen: {sorted({n for n, _ in seen})}; shortened slabs tested: {sum(1 for _, t in seen if t) > 0}" if not bad else str(bad[:3]))
                exp_n = {mm: max(1, math.floor(mm / dz + 0.5)) for mm in OPTIONS}
                R.setdefault("rounding", {})[str(dz)] = {str(mm): {"images": n, "actual_mm": n * dz} for mm, n in exp_n.items()}
                # ---- an isolated bright voxel is in the slab exactly when its image is, at the right place on screen ----
                (mi, mj, mk), mv = (3, 27, 12), 2500
                col, row = base.shape[0] - 1 - mi, base.shape[1] - 1 - mj
                inside = []
                for k in range(nz):
                    d = c.js(f"window.__viewer.mipHU({k}, 10)")
                    inside.append((int(arr16(d)[row, col]) == mv) == (d["k0"] <= mk <= d["k1"]))
                edges = c.js("window.__viewer.edges('axial')")
                ok(f"{tag}: orientation: the patient-LEFT, ANTERIOR marker voxel (2500 HU, image index 12) shows on the screen RIGHT and TOP, in every slab that contains its image and in no other",
                   all(inside) and col > base.shape[0] / 2 and row < base.shape[1] / 2 and edges == ["R", "L", "A", "P"], f"column {col} of {base.shape[0]}, row {row} of {base.shape[1]}; edge letters {edges}")
                # ---- what is DRAWN: window applied to the maxima; changing the window re-colours the same maxima ----
                k = 12
                c.js(f"(() => {{ const v = window.__viewer, s = v.state().cur; v.setCur([s[0], s[1], {k}]); v.setMip(true, 8); return true; }})()")
                time.sleep(0.2)
                p = plan(k, nz, dz, 8)
                want_hu = axial_display(base[:, :, p["k0"]:p["k1"] + 1].max(axis=2))
                wins, good = [(-1350, 150), (-160, 240), (-500, 1300), (-1024, 3071), (1000, 1001)], []
                for lo, hi in wins:
                    g = c.js(f"(() => {{ const v = window.__viewer; v.setWindow({lo}, {hi}, false); return v.shownGray('axial'); }})()")
                    good.append(g["mip"] and g["coloured_pixels"] == 0 and np.array_equal(arr8(g), lut(want_hu, lo, hi)))
                hu_after = arr16(c.js(f"window.__viewer.mipHU({k}, 8)"))
                ok(f"{tag}: the drawn MIP is the window applied to the HU maxima, for 5 windows; the stored maxima are HU and do not change with the window",
                   all(good) and np.array_equal(hu_after, want_hu), f"windows {wins}")
                c.js("window.__viewer.setWindow(-1350, 150, true)")
                # ---- the bar states the real thickness and image range ----
                info = c.js("window.__viewer.mip()")
                first, last = nz - p["k1"], nz - p["k0"]
                txt = info["info"]
                ok(f"{tag}: short label: approximate thickness, slice count, image range and centre image; spacing, exact coverage and centre distance are in the help text",
                   txt == f"MIP ≈{label_mm(p, 8)} mm  ·  {p['images']} slice{'' if p['images'] == 1 else 's'}  ·  Images {first}–{last}  ·  Centre {nz - k}" + ("  ·  shortened at scan edge" if p["truncated"] else "") + ("  ·  same as single image" if p["n"] == 1 else "")
                   and f"This slab: {p['images']} slice" in info["help"] and f"× {dz:g} mm spacing = {round(p['actual_mm'], 2):g} mm coverage (asked for 8 mm)" in info["help"]
                   and f"Centres of the first and last slice: {round((p['images'] - 1) * dz, 2):g} mm apart" in info["help"] and abs(info["centre_span_mm"] - (p["images"] - 1) * dz) < 1e-9
                   and info["first_image_no"] == first and info["last_image_no"] == last, txt)
                edge = c.js("(() => { const v = window.__viewer, s = v.state().cur; v.setCur([s[0], s[1], 0]); v.setMip(true, 10); return v.mip(); })()")
                pe = plan(0, nz, dz, 10)
                ok(f"{tag}: at the scan edge the slab is shortened and the page says so (planned {pe['n']} images, shown {pe['images']})",
                   edge["images"] == pe["images"] and edge["truncated"] == pe["truncated"] and (("shortened at scan edge" in edge["info"]) == pe["truncated"]) and edge["info"].startswith(f"MIP ≈{label_mm(pe, 10)} mm  ·  {pe['images']} slice")
                   and (not pe["truncated"] or f"a full slab is {pe['n']} slices" in edge["help"])
                   and abs(edge["actual_mm"] - pe["actual_mm"]) < 1e-9, edge["info"])
                if exp_n[5] == 1:
                    one = c.js("(() => { const v = window.__viewer, s = v.state().cur; v.setCur([s[0], s[1], 12]); v.setMip(true, 5); return {m: v.mip(), a: v.shownGray('axial').b64, b: v.sliceGray('axial', 12).b64}; })()")
                    ok(f"{tag}: when the spacing is larger than the requested slab, the slab is one image, equals the single image, and the page says so",
                       one["m"]["images"] == 1 and one["a"] == one["b"] and "same as single image" in one["m"]["info"] and one["m"]["info"].startswith(f"MIP ≈{dz:g} mm  ·  1 slice  ·"), one["m"]["info"])
                # ---- scrolling: one image per step, neighbouring slabs overlap ----
                sc = c.js("(() => { const v = window.__viewer, s = v.state().cur; v.setCur([s[0], s[1], 14]); v.setMip(true, 10); const a = v.mip(); v.step(1); const b = v.mip(); v.step(-1); v.step(-1); const d = v.mip();"
                          " return {a: [a.k0, a.k1, a.centre], b: [b.k0, b.k1, b.centre], d: [d.k0, d.k1, d.centre]}; })()")
                ok(f"{tag}: scrolling moves the slab by one image, so neighbouring slabs overlap",
                   sc["b"] == [min(x + 1, nz - 1) if i < 2 else x + 1 for i, x in enumerate(sc["a"])] or (sc["b"][2] == sc["a"][2] + 1 and sc["d"][2] == sc["a"][2] - 1), str(sc))
                # ---- back to the single image, by both buttons ----
                back = c.js("(async () => { const v = window.__viewer, s = v.state().cur; v.setCur([s[0], s[1], 12]); v.setMip(true, 10); await new Promise(r => setTimeout(r, 120)); const before = v.state().cur, lbl = document.getElementById('mipcentre').textContent,"
                            " shown = getComputedStyle(document.getElementById('mipcentre')).visibility === 'visible'; document.getElementById('mipcentre').click(); await new Promise(r => setTimeout(r, 120));"
                            " const o = {before, after: v.state().cur, on: v.mip().on, lbl, shown, same: v.shownGray('axial').b64 === v.sliceGray('axial', 12).b64, body: document.body.classList.contains('mipmode'), cache: v.mip().cache_entries};"
                            " document.getElementById('mipon').click(); await new Promise(r => setTimeout(r, 120)); o.on2 = v.mip().on; document.getElementById('mipoff').click(); await new Promise(r => setTimeout(r, 120));"
                            " o.on3 = v.mip().on; o.same3 = v.shownGray('axial').b64 === v.sliceGray('axial', 12).b64; o.after3 = v.state().cur; return o; })()")
                ok(f"{tag}: one click returns to the single image at the centre of the slab (green button and 'Single slice'): same image index, same pixels as before MIP, slab buffers released",
                   back["shown"] and back["before"] == back["after"] == back["after3"] and not back["on"] and back["same"] and not back["body"] and back["cache"] == 0 and back["on2"] and not back["on3"]
                   and back["same3"] and str(nz - 12) in back["lbl"], back["lbl"])
                # ---- thickness buttons; other views ----
                ui = c.js("(async () => { const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 120)); document.querySelector('[data-mip=\"5\"]').click(); await w(); const o = {sel5: v.mip().requested_mm, onAfterPick: v.mip().on};"
                          " document.getElementById('mipon').click(); await w(); o.on = v.mip().on; document.querySelector('[data-mip=\"8\"]').click(); await w(); o.sel8 = v.mip().requested_mm; o.still = v.mip().on;"
                          " document.querySelector('[data-v=\"coronal\"]').click(); await w(); o.cor = v.mip().on; o.dis = document.getElementById('mipon').disabled; o.txt = document.getElementById('mipinfo').textContent; v.setMip(true); o.refused = v.mip().on;"
                          " document.querySelector('[data-v=\"three\"]').click(); await w(); o.dis3 = document.getElementById('mipon').disabled; document.querySelector('[data-v=\"axial\"]').click(); await w(); o.dis2 = document.getElementById('mipon').disabled;"
                          " o.onBack = v.mip().on; v.setMip(false, 10); return o; })()")
                ok(f"{tag}: 5 / 8 / 10 mm buttons choose the slab without switching MIP on by themselves; Coronal, Sagittal and 3 planes leave MIP and disable its button; Axial enables it again (MIP stays off)",
                   ui["sel5"] == 5 and not ui["onAfterPick"] and ui["on"] and ui["sel8"] == 8 and ui["still"] and not ui["cor"] and ui["dis"] and not ui["refused"] and ui["dis3"] and not ui["dis2"] and not ui["onBack"], ui["txt"])
                # ---- outlines ----
                ov = c.js("(async () => { const v = window.__viewer, s = v.state().cur; v.setCur([s[0], s[1], 12]); await v.setOverlay('m'); const o = {single: v.shownGray('axial').coloured_pixels, act0: v.state().active};"
                          " v.setMip(true, 10); await v.idle(); o.act1 = v.state().active; o.bytes1 = v.state().overlayBytes; o.mipCol = v.shownGray('axial').coloured_pixels; await v.setOverlay('m'); o.act2 = v.state().active;"
                          " const bar = document.getElementById('ovbar'), chip = document.querySelector('#ovbar .ov'); const note = document.getElementById('mipnoout'); o.barGone = getComputedStyle(bar).display === 'none'; o.chipGone = chip.offsetParent === null;"
                          " o.barMsg = note.textContent; o.noteShown = getComputedStyle(note).display !== 'none'; o.noteInBar = note.parentElement.id === 'mipbar'; o.noteH = note.getBoundingClientRect().height; o.cvMip = document.getElementById('gl').getBoundingClientRect().height;"
                          " const h = document.getElementById('miphelp'); o.help0 = h.hidden; document.getElementById('miphelpbtn').click(); o.help1 = !h.hidden; o.helpText = h.textContent; o.cvHelp = document.getElementById('gl').getBoundingClientRect().height;"
                          " document.getElementById('miphelpclose').click(); o.help2 = h.hidden; o.infoLines = document.getElementById('mipinfo').getBoundingClientRect().height; v.setMip(false); await v.idle(); o.act3 = v.state().active; o.col3 = v.shownGray('axial').coloured_pixels; o.cvSingle = document.getElementById('gl').getBoundingClientRect().height; o.noteAfter = getComputedStyle(note).display;"
                          " o.chipBack = getComputedStyle(bar).display !== 'none' && chip.offsetParent !== null;"
                          " await v.setOverlay('m'); o.col4 = v.shownGray('axial').coloured_pixels; await v.setOverlay(null); return o; })()")
                ok(f"{tag}: outlines: an active outline is switched off and released when MIP starts, none can be switched on in MIP, no coloured pixel is drawn over a slab, the outline bar is removed in MIP and replaced by a one-line notice beside the MIP buttons, and outlines work again in the single image",
                   ov["single"] > 0 and ov["act0"] == "m" and ov["act1"] is None and ov["bytes1"] == 0 and ov["mipCol"] == 0 and ov["act2"] is None and ov["barGone"] and ov["chipGone"] and ov["barMsg"] == "Outlines are not shown in MIP view"
                   and ov["noteShown"] and ov["noteInBar"] and ov["noteH"] <= 24 and ov["noteAfter"] == "none" and ov["chipBack"]
                   and ov["act3"] is None and ov["col3"] == 0 and ov["col4"] == ov["single"], ov["barMsg"])
                ok(f"{tag}: one-line label with optional help: help is closed at first, opens and closes on click, explains the approximate figure, spacing versus slice thickness and why outlines are not shown, and does not resize the scan; "
                   "no empty outline strip in MIP (the scan area is at least as tall as on the single image)",
                   ov["help0"] and ov["help1"] and ov["help2"] and "slices × spacing" in ov["helpText"] and "Spacing is not slice thickness" in ov["helpText"] and "does not know it" in ov["helpText"] and "Outlines are not shown in MIP" in ov["helpText"]
                   and ov["infoLines"] <= 24 and ov["cvMip"] >= ov["cvSingle"] - 0.5 and abs(ov["cvHelp"] - ov["cvMip"]) < 0.5,
                   f"label height {ov['infoLines']} px; scan area {ov['cvSingle']} px high on the single image, {ov['cvMip']} px in MIP")
                # ---- fit at opening is checked on a fresh page below; here: zoom / pan chosen by the reader survive toggling ----
                zz = c.js("(async () => { const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 150)), gl = document.getElementById('gl'); v.setMip(false); await w(); v.zoomBy(2.5, undefined, undefined, 'axial'); await w();"
                          " const cr = v.canvasRect(); v.zoomBy(1.2, cr[0] + cr[2] * 0.3, cr[1] + cr[3] * 0.4, 'axial'); await w(); const snap = () => { const s = v.state(), L = v.layout('axial'), r = v.canvasRect(); return {zoom: s.zoom, pan: s.pan, px: s.pxScale, L: [L.x, L.y, L.w, L.h], cv: [gl.width, gl.height], cr: r, inside: L.x >= r[0] - 0.5 && L.y >= r[1] - 0.5 && L.x + L.w <= r[0] + r[2] + 0.5 && L.y + L.h <= r[1] + r[3] + 0.5}; };"
                          " const a = snap(); document.getElementById('mipon').click(); await w(); const b = snap(); document.querySelector('[data-mip=\"5\"]').click(); await w(); const b5 = snap(); v.step(1); v.step(-1); await w();"
                          " document.getElementById('mipcentre').click(); await w(); const d = snap(); document.getElementById('mipon').click(); await w(); document.getElementById('mipoff').click(); await w(); const e = snap();"
                          " document.getElementById('zfit').click(); await w(); const f = snap(); document.getElementById('mipon').click(); await w(); const g = snap(); document.getElementById('mipoff').click(); await w(); const h = snap(); v.setMip(false, 10); return {a, b, b5, d, e, f, g, h}; })()", 60)
                same, zdetail = zoom_kept(zz)
                ok(f"{tag}: a zoom chosen by the reader is kept when MIP is switched on (same screen pixels per image pixel, same image point at the centre of the scan area), when the slab is changed, "
                   "and is restored exactly when MIP is left by either button; from Fit the image stays at Fit in both modes",
                   same and abs(zz["a"]["zoom"] - 3.0) < 1e-9, zdetail)
                wr = c.js("(async () => { const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 120)), t = () => document.getElementById('winval').textContent, o = {};"
                          " document.querySelector('[data-w=\"-1350,150\"]').click(); await w(); o.lung = t(); document.querySelector('[data-w=\"-160,240\"]').click(); await w(); o.soft = t();"
                          " document.querySelector('[data-w=\"-500,1300\"]').click(); await w(); o.bone = t(); v.setWindow(-700, 500, false); await w(); o.custom = t(); v.setMip(true); await w(); v.setWindow(-100, 300, false); await w(); o.mip = t();"
                          " v.setMip(false); document.querySelector('[data-w=\"-1350,150\"]').click(); await w(); o.back = t(); return o; })()")
                ok(f"{tag}: window width and level are shown next to the presets and follow presets and free changes, in single-image and MIP mode",
                   wr == {"lung": "W 1500  ·  L -600", "soft": "W 400  ·  L 40", "bone": "W 1800  ·  L 400", "custom": "W 1200  ·  L -100", "mip": "W 400  ·  L 100", "back": "W 1500  ·  L -600"}, str(wr))
                # ---- cache bound ----
                cb = c.js("(() => { const v = window.__viewer, s = v.state().cur; v.setMip(true, 10); let mx = 0; for (let r = 0; r < 6; r++) for (let k = 0; k < v.images('axial'); k++) { v.setCur([s[0], s[1], k]); v.shownGray('axial'); for (const mm of [5, 8, 10]) v.mipHU(k, mm);"
                          " mx = Math.max(mx, v.mip().cache_entries); } const m = v.mip(); v.setMip(false); return {mx, max: m.cache_max, bytes: m.cache_bytes, limit: m.cache_limit_bytes, computed: m.computed, hits: m.hits, reused: m.reused, after: v.mip().cache_entries, errors: window.__viewerErrors}; })()", 120)
                ok(f"{tag}: slab cache stays inside its limit over {6 * nz * 4} slab requests and is emptied when MIP is left",
                   cb["mx"] <= cb["max"] and cb["bytes"] <= cb["limit"] and cb["after"] == 0 and not cb["errors"], f"at most {cb['mx']} of {cb['max']} slabs kept ({cb['bytes']} bytes); computed {cb['computed']}, reused from cache {cb['hits']}")
                if vi == 0:
                    c.js("(() => { const v = window.__viewer, s = v.state().cur; v.setCur([3, 27, 12]); v.setMip(true, 10); return true; })()")
                    time.sleep(0.5)
                    (out / f"synthetic_dz{dz}_mip.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
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
        (out / "startup_single_slice.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
        c.js(PREP)
        s = c.js("(() => { const v = window.__viewer, st = v.state(); return {st, mip: v.mip(), build: v.build, kind: v.kind, proto: location.protocol, chipsOn: document.querySelectorAll('#ovbar .ov:not(.off)').length,"
                 " on10: document.querySelector('[data-mip=\"10\"]').classList.contains('on'), single: document.getElementById('mipoff').classList.contains('on'), body: document.body.classList.contains('mipmode'),"
                 " viewOn: document.querySelector('[data-v].on').dataset.v, winOn: document.querySelector('[data-w].on').textContent, errors: window.__viewerErrors, case_id: META.case_id, codes: META.ct.original_axis_codes,"
                 " webgl: (() => { const d = document.documentElement.cloneNode(true); d.querySelectorAll('script.vp').forEach(e => e.remove()); return /webgl|niivue/i.test(d.innerHTML); })(),"
                 " same: v.shownGray('axial').b64 === v.sliceGray('axial', st.cur[2]).b64,"
                 " fit: (() => { const L = v.layout('axial'), r = v.canvasRect(); return {zoom: st.zoom, pan: st.pan, inside: L.x >= r[0] - 0.5 && L.y >= r[1] - 0.5 && L.x + L.w <= r[0] + r[2] + 0.5 && L.y + L.h <= r[1] + r[3] + 0.5, fills: Math.max(L.w / r[2], L.h / r[3])}; })(),"
                 " win: document.getElementById('winval').textContent, helpHidden: document.getElementById('miphelp').hidden, ovbarShown: getComputedStyle(document.querySelector('#ovbar .ov')).visibility,"
                 " saved: Object.keys(localStorage).filter(k => /zoom|pan/i.test(k) && k !== 'lungct_text_zoom')}; })()")
        R["startup"] = {k: s[k] for k in ("build", "case_id", "codes", "fit", "win")}
        ok("first opening uses Fit: zoom 100 % of fit, no pan, the whole image inside the scan area; no image zoom is stored between visits; window read-out shows W 1500, L -600; help closed",
           s["fit"]["zoom"] == 1 and s["fit"]["pan"] == [0, 0] and s["fit"]["inside"] and 0.93 < s["fit"]["fills"] <= 0.95 and not s["saved"] and s["win"] == "W 1500  ·  L -600" and s["helpHidden"] and s["ovbarShown"] == "visible",
           f"image fills {s['fit']['fills']:.2f} of the scan area; {s['win']}")
        dims, sp = s["st"]["dims"], s["st"]["spacing"]
        nz, dz = dims[2], sp[2]
        R["startup"].update(dims=dims, spacing=sp)
        ok("opened from file://; no WebGL and no viewer library in the file", s["proto"] == "file:" and s["kind"] == "canvas2d" and not s["webgl"] and s["build"] == v5.VIEWER_BUILD, s["build"])
        ok("opens as before: single axial image, lung window, no outline, MIP off with 10 mm selected",
           s["st"]["view"] == "axial" and s["st"]["mode"] == "single" and s["viewOn"] == "axial" and s["winOn"] == "Lung" and s["chipsOn"] == 0 and s["st"]["active"] is None and not s["mip"]["on"]
           and s["on10"] and s["single"] and not s["body"] and s["same"] and not s["errors"], f"image {s['st']['imageNo']} of {s['st']['images']}")
        # ---- the slab against the ORIGINAL file, read with SimpleITK ----
        case = s["case_id"]
        src = ROOT / "data" / "raw" / "ct_rate" / "volumes" / f"{case}.nii.gz"
        img = sitk.ReadImage(str(src))
        orig = sitk.GetArrayFromImage(img)                        # [z, y, x] in file index order
        w, h = dims[0], dims[1]
        worst, n_cmp, rows = 0, 0, []
        for mm in OPTIONS:
            for k in sorted({0, 2, nz // 4, nz // 2, (3 * nz) // 4, nz - 3, nz - 1}):
                d = c.js(f"window.__viewer.mipHU({k}, {mm})")
                p = plan(k, nz, dz, mm)
                ref = None
                for z in range(p["k0"], p["k1"] + 1):
                    gi = grid_from_sitk(img, b, "axial", z, w, h)
                    sl = np.clip(orig[gi[..., 2], gi[..., 1], gi[..., 0]], -1024, 3071).astype(np.int16)      # the page's prepared display range
                    ref = sl if ref is None else np.maximum(ref, sl)
                got = arr16(d)
                diff = int(np.abs(got.astype(np.int32) - ref.astype(np.int32)).max())
                worst = max(worst, diff)
                n_cmp += 1
                same_plan = (d["k0"], d["k1"], d["images"]) == (p["k0"], p["k1"], p["images"]) and abs(d["actual_mm"] - p["actual_mm"]) < 1e-6
                rows.append({"mm": mm, "centre_index": k, "k0": d["k0"], "k1": d["k1"], "images": d["images"], "actual_mm": round(d["actual_mm"], 3), "max_abs_difference_hu": diff, "plan_ok": same_plan})
        R["slab_vs_original_file"] = rows
        R["rounding"] = {str(mm): {"images": plan(nz // 2, nz, dz, mm)["images"], "actual_mm": round(plan(nz // 2, nz, dz, mm)["actual_mm"], 3)} for mm in OPTIONS}
        ok(f"slab maxima equal the maximum over the same images of the ORIGINAL file read with SimpleITK ({n_cmp} slabs: 5 / 8 / 10 mm, 7 centre images including both scan edges)",
           worst == 0 and all(r["plan_ok"] for r in rows), f"largest difference {worst} HU; slice spacing {dz:.3f} mm; 5 / 8 / 10 mm -> {R['rounding']}")
        # ---- buttons, text, drawn pixels, back to the centre image ----
        k = nz // 2
        ui = c.js(f"(async () => {{ const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 200)), s = v.state().cur; v.setCur([s[0], s[1], {k}]); await w(); const single = v.sliceGray('axial', {k}).b64;"
                  " document.getElementById('mipon').click(); await w(); const o = {on: v.mip().on, m10: v.mip(), body: document.body.classList.contains('mipmode'), loc: document.getElementById('loc').textContent,"
                  " lines: document.getElementById('mipinfo').getBoundingClientRect().height, chipHidden: getComputedStyle(document.getElementById('ovbar')).display === 'none' && document.querySelector('#ovbar .ov').offsetParent === null, barMsg: document.getElementById('mipnoout').textContent,"
                  " noteShown: getComputedStyle(document.getElementById('mipnoout')).display !== 'none', noteH: document.getElementById('mipnoout').getBoundingClientRect().height, cvMip: document.getElementById('gl').getBoundingClientRect().height,"
                  " help: (() => { const h = document.getElementById('miphelp'), o = {closed: h.hidden}; document.getElementById('miphelpbtn').click(); o.open = !h.hidden; o.text = h.textContent; document.getElementById('miphelpclose').click(); o.closedAgain = h.hidden; return o; })(), drawn: v.shownGray('axial'), differs: v.shownGray('axial').b64 !== single};"
                  " document.querySelector('[data-mip=\"5\"]').click(); await w(); o.m5 = v.mip(); document.querySelector('[data-mip=\"8\"]').click(); await w(); o.m8 = v.mip();"
                  " o.btn = document.getElementById('mipcentre').textContent; document.getElementById('mipcentre').click(); await w(); o.off = !v.mip().on; o.cur = v.state().cur; o.back = v.shownGray('axial').b64 === single;"
                  " o.cache = v.mip().cache_entries; o.sel = v.mip().requested_mm; o.cvSingle = document.getElementById('gl').getBoundingClientRect().height; o.ovBack = getComputedStyle(document.getElementById('ovbar')).display !== 'none'; return o; })()", 60)
        p10 = plan(k, nz, dz, 10)
        m10 = ui["m10"]
        ok("MIP button shows the slab with a short label (approximate thickness, slice count, image range, centre); spacing, exact coverage and centre distance are in the help text, which opens on request",
           ui["on"] and ui["body"] and ui["differs"] and ui["drawn"]["mip"] and ui["drawn"]["coloured_pixels"] == 0
           and m10["info"] == f"MIP ≈10 mm  ·  {p10['images']} slices  ·  Images {nz - p10['k1']}–{nz - p10['k0']}  ·  Centre {nz - k}" and ui["lines"] <= 24
           and ui["help"]["closed"] and ui["help"]["open"] and ui["help"]["closedAgain"] and "does not know it" in ui["help"]["text"]
           and f"This slab: {p10['images']} slices × {dz:g} mm spacing = {round(p10['actual_mm'], 2):g} mm coverage (asked for 10 mm)" in ui["help"]["text"]
           and f"{round((p10['images'] - 1) * dz, 2):g} mm apart" in ui["help"]["text"] and "MIP: brightest in slab" in ui["loc"], m10["info"])
        ok("no empty strip in MIP: the outline bar is removed and a one-line notice sits beside the MIP buttons; the scan area is taller in MIP; the outline bar is back on the single image",
           ui["chipHidden"] and ui["barMsg"] == "Outlines are not shown in MIP view" and ui["noteShown"] and ui["noteH"] <= 24 and ui["cvMip"] > ui["cvSingle"] and ui["ovBack"],
           f"scan area {ui['cvSingle']:.0f} px high on the single image, {ui['cvMip']:.0f} px in MIP")
        ok("5, 8 and 10 mm buttons change the slab", ui["m5"]["requested_mm"] == 5 and ui["m8"]["requested_mm"] == 8 and ui["m5"]["images"] == plan(k, nz, dz, 5)["images"] and ui["m8"]["images"] == plan(k, nz, dz, 8)["images"],
           f"5 mm: {ui['m5']['images']} images; 8 mm: {ui['m8']['images']}; 10 mm: {ui['m10']['images']}")
        ok("one click returns to the single image at the centre of the slab: same image, same pixels as before MIP", ui["off"] and ui["cur"][2] == k and ui["back"] and ui["cache"] == 0, ui["btn"])
        # ---- zoom chosen by the reader is kept when toggling ----
        zz = c.js("(async () => { const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 200)), gl = document.getElementById('gl'); document.getElementById('zin').click(); document.getElementById('zin').click(); await w();"
                  " const cr = v.canvasRect(); v.zoomBy(1.3, cr[0] + cr[2] * 0.35, cr[1] + cr[3] * 0.4, 'axial'); await w(); const snap = () => { const s = v.state(), L = v.layout('axial'), r = v.canvasRect(); return {zoom: s.zoom, pan: s.pan, px: s.pxScale, L: [L.x, L.y, L.w, L.h], cv: [gl.width, gl.height], cr: r, inside: L.x >= r[0] - 0.5 && L.y >= r[1] - 0.5 && L.x + L.w <= r[0] + r[2] + 0.5 && L.y + L.h <= r[1] + r[3] + 0.5}; };"
                  " const a = snap(); document.getElementById('mipon').click(); await w(); const b = snap(); document.querySelector('[data-mip=\"5\"]').click(); await w(); const b5 = snap(); document.querySelector('[data-mip=\"10\"]').click(); await w();"
                  " document.getElementById('mipcentre').click(); await w(); const d = snap(); document.getElementById('mipon').click(); await w(); document.getElementById('mipoff').click(); await w(); const e = snap();"
                  " const wsoft = (() => { document.querySelector('[data-w=\"-160,240\"]').click(); return 0; })(); await w(); const soft = document.getElementById('winval').textContent; v.setWindow(-700, 500, false); await w(); const custom = document.getElementById('winval').textContent;"
                  " document.querySelector('[data-w=\"-1350,150\"]').click(); await w(); document.getElementById('zfit').click(); await w(); const f = snap(); document.getElementById('mipon').click(); await w(); const g = snap(); document.getElementById('mipoff').click(); await w(); const h = snap(); return {a, b, b5, d, e, f, g, h, soft, custom}; })()", 60)
        same, zdetail = zoom_kept(zz)
        R["zoom_preserved"] = {"single": {kk: zz["a"][kk] for kk in ("zoom", "pan", "px", "cv")}, "mip": {kk: zz["b"][kk] for kk in ("zoom", "pan", "px", "cv")}, "back": {kk: zz["d"][kk] for kk in ("zoom", "pan", "px", "cv")}}
        ok("a zoom chosen by the reader is kept when MIP is switched on (same screen pixels per image pixel, same image point at the centre of the scan area), when the slab is changed, "
           "and is restored exactly when MIP is left by either button; from Fit the image stays at Fit in both modes", same and zz["a"]["zoom"] > 1.9, zdetail)
        ok("window width and level are shown next to the presets and follow a free window change", zz["soft"] == "W 400  ·  L 40" and zz["custom"] == "W 1200  ·  L -100", f"soft tissue: {zz['soft']}; free: {zz['custom']}")
        # ---- real wheel scrolling in MIP ----
        rect = c.js("window.__viewer.canvasRect()")
        cx, cy = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
        c.js(f"(() => {{ const v = window.__viewer, s = v.state().cur; v.setCur([s[0], s[1], {k}]); v.setMip(true, 10); return true; }})()")
        for _ in range(5):
            c.call("Input.dispatchMouseEvent", type="mouseWheel", x=cx, y=cy, deltaX=0, deltaY=100)
            time.sleep(0.05)
        time.sleep(0.3)
        wm = c.js("window.__viewer.mip()")
        ok("mouse wheel in MIP: one notch moves the slab by one image (5 notches down = 5 images towards the feet); slabs overlap", wm["centre"] == k - 5 and wm["on"] and wm["k1"] - wm["k0"] + 1 == p10["images"], wm["info"])
        c.js("(() => { window.__paints = 0; const v = window.__viewer; const tick = () => { window.__paints++; requestAnimationFrame(tick); }; requestAnimationFrame(tick); return true; })()")
        n_wheel = args.wheel_notches
        t_w = time.time()
        for i in range(n_wheel):
            c.call("Input.dispatchMouseEvent", type="mouseWheel", x=cx, y=cy, deltaX=0, deltaY=100 if (i // 60) % 2 == 0 else -100)
        time.sleep(0.4)
        wheel_s = time.time() - t_w - 0.4
        ww = c.js("(() => { const v = window.__viewer; return {m: v.mip(), frames: window.__paints, errors: window.__viewerErrors, ready: v.ready}; })()")
        R["timings"]["wheel_notches"] = n_wheel
        R["timings"]["wheel_ms_per_notch_including_test_driver"] = round(1000 * wheel_s / n_wheel, 1)
        ok(f"{n_wheel} real mouse-wheel notches in MIP (direction reversed every 60), drawn on screen: no error, slab cache inside its limit",
           ww["m"]["on"] and not ww["errors"] and ww["ready"] and ww["m"]["cache_entries"] <= ww["m"]["cache_max"] and ww["m"]["centre"] == k - 5 + sum(1 if (i // 60) % 2 == 0 else -1 for i in range(n_wheel)) * -1,
           f"{R['timings']['wheel_ms_per_notch_including_test_driver']} ms per notch including the test driver; cache {ww['m']['cache_entries']} of {ww['m']['cache_max']} slabs")
        (out / "mip_10mm.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
        # ---- "Show in scan" from MIP returns to the single image with the outline ----
        jump = c.js("(async () => { const v = window.__viewer; const tr = Array.from(document.querySelectorAll('tr.jump')).find(t => t.dataset.jump && !document.body.classList.contains('stage1')); if (!tr) return null;"
                    " v.setMip(true, 10); await new Promise(r => setTimeout(r, 150)); tr.click(); for (let i = 0; i < 100 && v.state().active !== tr.dataset.jump; i++) await new Promise(r => setTimeout(r, 100)); await v.idle();"
                    " const o = {key: tr.dataset.jump, on: v.mip().on, active: v.state().active, col: v.shownGray('axial').coloured_pixels}; await v.setOverlay(null); return o; })()", 60)
        if jump:
            ok("'Show in scan' pressed while in MIP: the page returns to the single image and shows the outline there, never over a slab", not jump["on"] and jump["active"] == jump["key"] and jump["col"] > 0, f"outline {jump['key']}, {jump['col']} outlined pixels")
        # ---- repeated scrolling, window and thickness changes; memory ----
        t0 = time.time()
        st2 = c.js(f"(async () => {{ const v = window.__viewer, s = v.state().cur, n = v.images('axial'); v.setMip(true, 10); let mx = 0, k = {k}, dir = 1, t = 0; const frame = () => new Promise(r => requestAnimationFrame(() => r()));"
                   f" for (let i = 0; i < {args.scroll_steps}; i++) {{ k += dir; if (k >= n - 1 || k <= 0) dir = -dir; v.setCur([s[0], s[1], k]);"
                   " if (i % 3 === 0) { const c = -1400 + (i * 37) % 1800, w = 200 + (i * 53) % 1800; v.setWindow(c - w / 2, c + w / 2, false); }"
                   " if (i % 25 === 0) v.setMip(true, [5, 8, 10][(i / 25) % 3]); if (i % 60 === 59) { v.setMip(false); v.setMip(true); }"
                   " const a = performance.now(); v.shownGray('axial'); t = Math.max(t, performance.now() - a); mx = Math.max(mx, v.mip().cache_entries); if (i % 4 === 0) await frame(); }"
                   " const m = v.mip(); v.setWindow(-1350, 150, true); return {mx, max: m.cache_max, bytes: m.cache_bytes, limit: m.cache_limit_bytes, computed: m.computed, hits: m.hits, reused: m.reused, slowest_ms: Math.round(t), errors: window.__viewerErrors, ready: v.ready}; })()", 600)
        R["timings"]["stress_s"] = round(time.time() - t0, 1)
        R["timings"]["slowest_slab_draw_ms"] = st2["slowest_ms"]
        ok(f"{args.scroll_steps} scroll steps in MIP with {args.scroll_steps // 3} window changes and repeated 5 / 8 / 10 mm and on / off switches: no error, slab cache inside its limit",
           st2["mx"] <= st2["max"] and st2["bytes"] <= st2["limit"] and not st2["errors"] and st2["ready"],
           f"at most {st2['mx']} of {st2['max']} slabs kept ({st2['mx'] * dims[0] * dims[1] * 2 / 1e6:.1f} of {st2['limit'] / 1e6:.0f} MB); slabs computed {st2['computed']}; slowest slab + colouring {st2['slowest_ms']} ms")
        in_mip = b.gc_mem()
        c.js("window.__viewer.setMip(false)")
        time.sleep(1)
        after = b.gc_mem()
        base_m = R["memory"]["sustained_after_startup"]
        R["memory"]["sustained_in_mip_after_stress"] = in_mip
        R["memory"]["sustained_after_leaving_mip"] = after
        R["memory"]["retained_growth"] = {kk: after[kk] - base_m[kk] for kk in ("private_mb", "working_set_mb", "renderer_ws_mb", "gpu_process_ws_mb")}
        R["memory"]["gpu_used_delta_mb"] = None if g0 is None else gpu_mb() - g0
        ok("memory after the stress block and leaving MIP: retained growth below 150 MB private; the page still works",
           R["memory"]["retained_growth"]["private_mb"] < 150 and c.js("window.__viewer.ready") and not c.js("window.__viewerErrors.length"),
           f"retained growth {R['memory']['retained_growth']['private_mb']} MB private; in MIP {in_mip['private_mb']} MB, start-up {base_m['private_mb']} MB")
        # ---- grading and CSV export after using MIP ----
        if c.js("!!document.getElementById('gsave')"):
            g = c.js("(async () => { const v = window.__viewer; v.setMip(true, 10); const rev = document.getElementById('grev'); const prefilled = !!document.querySelector('#chk td.grade .gb button.sel'); rev.value = window.__TEST_REVIEWER; rev.oninput();"
                     " const rows = Array.from(document.querySelectorAll('#chk tr')).filter(r => r.querySelector('td.grade'));"
                     " rows.forEach((tr, i) => { const td = tr.querySelector('td.grade'); const want = td.querySelector('.gb button.sel') || td.querySelector('.gb button[data-a=\"' + ['present', 'absent', 'cannot_tell'][i % 3] + '\"]'); if (!want.classList.contains('sel')) want.click(); });"
                     " const lock = document.getElementById('glock'); if (lock) lock.click(); await new Promise(r => setTimeout(r, 200));"
                     " document.getElementById('gsave').click(); await new Promise(r => setTimeout(r, 1200));"
                     " return {rows: rows.length, csv: window.__csv, graded: document.getElementById('gcount').textContent, prefilled, blocked: window.__blockedDownloads || 0, mipStill: v.mip().on}; })()", 120)
            csv_text = (g["csv"] or "").lstrip("﻿")
            lines = [ln for ln in csv_text.split("\r\n") if ln]
            (out / "exported_grades_AUTOMATED_TEST.csv").write_text(csv_text, encoding="utf-8", newline="")
            ok("grading with MIP switched on: every condition graded and autosaved", g["graded"].startswith(f"{g['rows']} of {g['rows']}") and g["mipStill"], g["graded"])
            ok("CSV export after using MIP: header, one row per condition and the overall row; rows name this viewer build",
               len(lines) == g["rows"] + 2 and lines[0].startswith("case_id,reviewer,saved_utc,condition_key") and v5.VIEWER_BUILD in lines[1] and "AUTOMATED TEST" in lines[1], f"{len(lines)} lines; prefilled form: {g['prefilled']}")
        else:
            ok("grading form present", False, "no grading form on this page")
        c.js("window.__viewer.setMip(false)")
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
    ap.add_argument("--port", type=int, default=9521)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not args.synthetic and not args.page:
        ap.error("use --synthetic or --page")
    name = "synthetic" if args.synthetic else args.page.stem
    out = ROOT / "runs" / f"V7_mip_test_{args.label + '_' if args.label else ''}{name}_{args.browser}_{'sw' if args.synthetic else args.gpu_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    R = {"kind": "technical checks only; not clinician usability feedback", "browser": args.browser, "gpu_mode": "sw" if args.synthetic else args.gpu_mode, "viewer_build": v5.VIEWER_BUILD,
         "checks": [], "memory": {}, "timings": {}}

    def ok(name, cond, detail=None):
        R["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
        print(("PASS " if cond else "FAIL ") + name + (f"  [{str(detail)[:260]}]" if detail is not None else ""), flush=True)
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
    md = [f"# MIP viewer technical checks: {name} in {R.get('browser_version', args.browser)} - {'PASSED' if R['passed'] else 'FAILED'}", "",
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
