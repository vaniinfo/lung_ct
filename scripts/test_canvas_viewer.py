"""Checks for the 2D canvas reviewer page (viewer v4), run against the UNMODIFIED html from file:// (core env).

Two modes:
  --synthetic            geometry: an asymmetric phantom defined in patient terms is stored in all 48 axis orders /
                         flips with anisotropic spacing; every page must DRAW the same image, with the Left-Anterior-
                         Superior marker on the correct side of each view, the physical aspect ratio, and "go to mm".
  --page FILE            a real case page: start-up defaults, scrolling, window and contrast changes, views, zoom / pan,
                         50+ outline switches, "Show in scan", grading, CSV, error handling, and a comparison of drawn
                         slices and outlines with SimpleITK reading the ORIGINAL file (independent of the page builder).

Memory, always measured the same way (Windows, summed over all browser processes of the test profile, MB):
  peak_private      highest private bytes seen while the page loads (sampled every 0.5 s)
  sustained_private private bytes a few seconds after loading, after a forced garbage collection
  retained_growth   sustained_private after the whole interaction block (again after garbage collection) minus the
                    sustained value after start-up
  baseline_private  the same browser with an empty page, for reference
Working set is recorded next to private bytes. The browser is headless unless --headed; --gpu-mode igpu forces the
low-power (integrated) adapter.

Usage: python scripts/test_canvas_viewer.py --synthetic --browser edge --gpu-mode igpu
       python scripts/test_canvas_viewer.py --page runs/M3_canvas_viewer_20261122/viewer_valid_832_v4.html --browser chrome --gpu-mode igpu
Writes runs/V2_canvas_test_<label>_<time>/{result.json,result.md,*.png}.
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from test_viewer_page import BROWSERS, CDP, PREP, gpu_mb, proc_mem_mb  # noqa: E402

GPU_FLAGS = {"igpu": ["--enable-gpu", "--ignore-gpu-blocklist", "--force_low_power_gpu"], "sw": ["--disable-gpu"],
             "gpu": ["--enable-gpu", "--ignore-gpu-blocklist", "--force_high_performance_gpu"]}


class Browser:
    def __init__(self, browser: str, mode: str, url: str, port: int, headed: bool = False):
        exe = next((p for p in BROWSERS[browser] if Path(p).exists()), None)
        if exe is None:
            raise SystemExit(f"{browser} is not installed")
        self.profile = ROOT / "runs" / "test_logs" / f"browser_profile_cv_{browser}_{mode}_{os.getpid()}_{int(time.time() * 10) % 100000}"
        flags = [exe, f"--remote-debugging-port={port}", "--remote-allow-origins=*", f"--user-data-dir={self.profile}", "--no-first-run",
                 "--no-default-browser-check", "--window-size=1700,1000", "--disable-extensions"] + GPU_FLAGS[mode]
        if not headed:
            flags.insert(1, "--headless=new")
        self.t0 = time.time()
        self.proc = subprocess.Popen(flags + [url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tgt = None
        for _ in range(240):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2).read())
                tgt = next((t for t in tabs if t.get("type") == "page" and not t.get("url", "").startswith("devtools")), None)
                if tgt:
                    break
            except Exception:
                pass
            time.sleep(0.25)
        if not tgt:
            raise RuntimeError("the browser did not open the page")
        self.c = CDP(tgt["webSocketDebuggerUrl"])
        self.c.call("Runtime.enable")
        self.c.call("Page.enable")
        try:   # belt and braces: the browser itself refuses downloads during a test
            self.c.call("Browser.setDownloadBehavior", behavior="deny")
        except Exception:
            pass
        self.version = self.c.call("Browser.getVersion")["product"]

    def mem(self) -> dict:
        m = proc_mem_mb(self.profile)
        return {"private_mb": m["private_mb"], "working_set_mb": m["working_set_mb"], "renderer_ws_mb": m["by_type_mb"].get("renderer", 0),
                "gpu_process_ws_mb": m["by_type_mb"].get("gpu-process", 0)}

    def gc_mem(self) -> dict:
        self.c.call("HeapProfiler.collectGarbage")
        time.sleep(2.5)
        return self.mem()

    def close(self):
        try:
            self.c.call("Browser.close")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()
        subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*" + self.profile.name
                        + "*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"], capture_output=True)
        time.sleep(1)
        shutil.rmtree(self.profile, ignore_errors=True)


def wait_ready(b: Browser, timeout_s: int = 600) -> tuple[dict, dict]:
    peak = {"private_mb": 0, "working_set_mb": 0}
    st = None
    t = time.time()
    while time.time() - t < timeout_s:
        m = b.mem()
        for k in peak:
            peak[k] = max(peak[k], m[k])
        try:
            st = b.c.js("(() => { const v = window.__viewer; if (!v) return null; const s = document.getElementById('vstat');"
                        " return {ready: v.ready, err: s && s.className === 'err' ? s.textContent : null}; })()", 30)
        except Exception:
            st = None
        if st and (st["ready"] or st["err"]):
            break
        time.sleep(0.5)
    return peak, st or {"ready": False, "err": "timed out"}


def arr8(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["b64"]), np.uint8).reshape(d["h"], d["w"])


def arr16(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["b64"]), "<i2").reshape(d["h"], d["w"])


# ---------------------------------------------------------------- synthetic geometry
def phantom():
    shape, spacing, origin = (44, 36, 28), (0.8, 1.3, 2.5), (-37.0, 12.5, -140.0)
    i, j, k = np.indices(shape)
    a = ((i * 7 + j * 13 + k * 29) % 1500 - 1000).astype(np.int16)        # texture: any mis-ordering changes the picture
    a[2:10, 26:34, 20:26] = 3000                                          # LEFT (low i), ANTERIOR (high j), SUPERIOR (high k)
    mask = np.zeros(shape, np.uint8)
    mask[4:14, 20:30, 16:24] = 1
    aff = np.diag(spacing + (1.0,))
    aff[:3, 3] = origin
    return a, mask, aff


def variant(arr, aff, perm, flips):
    out = np.transpose(arr, perm)
    m = aff[:3, :3][:, list(perm)].copy()
    t = aff[:3, 3].copy()
    for ax in range(3):
        if flips[ax]:
            out = np.flip(out, axis=ax)
            t = t + m[:, ax] * (out.shape[ax] - 1)
            m[:, ax] = -m[:, ax]
    v = np.eye(4)
    v[:3, :3], v[:3, 3] = m, t
    return np.ascontiguousarray(out), v


def expected_display(base: np.ndarray, view: str, s: int) -> np.ndarray:
    """What a radiological display must show, written from the physical convention (independent of the page code):
    base is RAS+ (axis 0 -> Right, 1 -> Anterior, 2 -> Superior). Result is [row from top, column from left]."""
    if view == "axial":       # left = Right (high i first), top = Anterior (high j first)
        return base[::-1, ::-1, s].T
    if view == "coronal":     # left = Right, top = Superior (high k first)
        return base[::-1, s, ::-1].T
    return base[s, ::-1, ::-1].T   # sagittal: left = Anterior (high j first), top = Superior


def run_synthetic(args, out: Path, R: dict, ok) -> None:
    import viewer_v4 as v4
    base, mask, aff = phantom()
    d = ROOT / "runs" / "test_logs" / "synth_v4"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    variants = [(p, f) for p in itertools.permutations(range(3)) for f in itertools.product((0, 1), repeat=3)]
    marker_mm = (aff @ np.append(np.argwhere(base == 3000).mean(0), 1))[:3]
    target_vox = [9, 25, 20]
    target_mm = (aff @ np.append(target_vox, 1))[:3]
    for n, (perm, flips) in enumerate(variants):
        a, v = variant(base, aff, perm, flips)
        m, _ = variant(mask, aff, perm, flips)
        (d / f"v{n:02d}.html").write_text(v4.standalone_page(f"phantom {perm} {flips}", a, v, {"m": ("Marker", "#ff0000", m)},
                                                             {"m": [float(x) for x in target_mm]}), encoding="utf-8")
    b = Browser(args.browser, args.gpu_mode, (d / "v00.html").as_uri(), args.port)
    R["browser_version"] = b.version
    bad, codes = [], set()
    try:
        for n, (perm, flips) in enumerate(variants):
            if n:
                b.c.call("Page.navigate", url=(d / f"v{n:02d}.html").as_uri())
            _, st = wait_ready(b, 60)
            if not st["ready"]:
                bad.append((n, "not ready", st["err"]))
                continue
            info = b.c.js("(() => { const v = window.__viewer; return {dims: v.state().dims, sp: v.state().spacing, codes: META.ct.original_axis_codes, verified: v.verified,"
                          " ax: v.edges('axial'), co: v.edges('coronal'), sa: v.edges('sagittal')}; })()")
            codes.add(info["codes"])
            prob = []
            if info["dims"] != list(base.shape) or not np.allclose(info["sp"], (0.8, 1.3, 2.5)):
                prob.append(f"dims/spacing {info['dims']} {info['sp']}")
            if info["ax"] != ["R", "L", "A", "P"] or info["co"] != ["R", "L", "S", "I"] or info["sa"] != ["A", "P", "S", "I"]:
                prob.append("edge labels")
            if not info["verified"] or info["verified"]["sha256"] is not True:
                prob.append(f"not verified {info['verified']}")
            for view, s in (("axial", 3), ("axial", 22), ("coronal", 5), ("coronal", 30), ("sagittal", 6), ("sagittal", 40)):
                got = arr16(b.c.js(f"window.__viewer.sliceHU('{view}', {s})"))
                if not np.array_equal(got, expected_display(base, view, s)):
                    prob.append(f"{view} {s} pixels differ")
            b.c.js("(async () => { const v = window.__viewer; v.setView('axial'); await v.setOverlay('m'); return 1; })()")
            for view, s in (("axial", 20), ("coronal", 25), ("sagittal", 8)):
                got = arr8(b.c.js(f"window.__viewer.sliceMask('{view}', {s})"))
                if not np.array_equal(got, expected_display(mask, view, s)):
                    prob.append(f"{view} {s} outline differs")
            # marker side on screen, in every view, from its PHYSICAL position
            for view, want in (("axial", ("right", "top")), ("coronal", ("right", "top")), ("sagittal", ("left", "top"))):
                p = b.c.js(f"(async () => {{ const v = window.__viewer; v.setView('{view}'); v.goToMm({[float(x) for x in marker_mm]}); await new Promise(r => setTimeout(r, 60));"
                           " const s = v.screenOf(v.state().cur), L = v.layout(); return {fx: (s[0] - L.x) / L.w, fy: (s[1] - L.y) / L.h, aspect: L.w / L.h, want: (L.cols * L.pw) / (L.rows * L.ph)}; })()")
                side = ("right" if p["fx"] > 0.5 else "left", "top" if p["fy"] < 0.5 else "bottom")
                if side != want:
                    prob.append(f"{view}: marker drawn {side}, must be {want}")
                if abs(p["aspect"] - p["want"]) > 1e-6 * p["want"]:
                    prob.append(f"{view}: aspect {p['aspect']:.4f} vs physical {p['want']:.4f}")
            cur = b.c.js(f"(() => {{ const v = window.__viewer; v.setView('axial'); return v.goToMm({[float(x) for x in target_mm]}); }})()")
            if cur != target_vox:
                prob.append(f"go to mm -> {cur}, expected {target_vox}")
            if prob:
                bad.append((n, perm, flips, prob))
        (out / "phantom_axial.png").write_bytes(base64.b64decode(b.c.call("Page.captureScreenshot", format="png")["data"]))
    finally:
        b.close()
    R["synthetic"] = {"variants": len(variants), "stored_orientations": sorted(codes), "problems": bad}
    ok(f"{len(variants)} stored orientations ({len(codes)} distinct axis codes): drawn pixels identical to the physical expectation in axial, coronal, sagittal",
       not bad, f"{len(bad)} variants with problems: {bad[:2]}" if bad else "6 CT slices and 3 outline slices compared exactly per variant")
    ok("Left-Anterior-Superior marker on the correct screen side in all views; physical aspect ratio; 'go to mm' picks the right image", not bad)
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- real page
def grid_from_sitk(img, b: Browser, view: str, s: int, w: int, h: int) -> np.ndarray:
    """Index grid into a SimpleITK image (LPS physical space) for every pixel of a drawn slice, from 3 corner positions."""
    def idx(c, r):
        mm = b.c.js(f"window.__viewer.pixelMm('{view}', {s}, {c}, {r})")
        return np.array(img.TransformPhysicalPointToContinuousIndex((-mm[0], -mm[1], mm[2])))     # RAS mm -> LPS mm
    i00, i10, i01 = idx(0, 0), idx(w - 1, 0), idx(0, h - 1)
    cc, rr = np.meshgrid(np.arange(w), np.arange(h))
    g = i00[None, None, :] + (i10 - i00)[None, None, :] * (cc / max(w - 1, 1))[..., None] + (i01 - i00)[None, None, :] * (rr / max(h - 1, 1))[..., None]
    gi = np.rint(g).astype(int)
    if np.abs(g - gi).max() > 0.05:
        raise ValueError(f"drawn pixels do not sit on voxel centres of the original file (max offset {np.abs(g - gi).max():.3f} voxel)")
    return gi


def run_page(args, out: Path, R: dict, ok) -> None:
    import SimpleITK as sitk
    page = (args.page if args.page.is_absolute() else ROOT / args.page).resolve()
    R["page"], R["file_mb"] = str(page.relative_to(ROOT)), round(page.stat().st_size / 1e6, 1)
    url = page.as_uri() + (f"?hash={args.hash}" if args.hash != "chunks" else "")
    if not args.skip_baseline:
        bb = Browser(args.browser, args.gpu_mode, "about:blank", args.port, args.headed)
        time.sleep(4)
        R["memory"]["baseline_empty_page"] = bb.gc_mem()
        bb.close()
    g0 = gpu_mb()
    b = Browser(args.browser, args.gpu_mode, url, args.port, args.headed)
    R["browser_version"] = b.version
    c = b.c
    try:
        peak, st = wait_ready(b)
        R["timings"]["launch_to_ready_s"] = round(time.time() - b.t0, 1)
        R["memory"]["peak_while_loading"] = peak
        if not st["ready"]:
            ok("scan loaded", False, st["err"])
            return
        ok("opened from file:// and the scan loaded", c.js("location.protocol") == "file:", f"{R['timings']['launch_to_ready_s']} s from browser launch")
        time.sleep(2)
        R["memory"]["sustained_after_startup"] = b.gc_mem()
        (out / "startup.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
        c.js(PREP)
        s = c.js("(() => { const v = window.__viewer, st = v.state(); return {st, kind: v.kind, verified: v.verified, timings: v.timings, chips: document.querySelectorAll('#ovbar .ov').length,"
                 " chipsOn: document.querySelectorAll('#ovbar .ov:not(.off)').length, ctBlocksLeft: document.querySelectorAll('script.vp[data-vol=ct]').length, errors: window.__viewerErrors,"
                 " webgl: (() => { const d = document.documentElement.cloneNode(true); d.querySelectorAll('script.vp').forEach(e => e.remove()); return /webgl|niivue/i.test(d.innerHTML); })(), meta: {codes: META.ct.original_axis_codes, obl: META.ct.obliquity_deg, case_id: META.case_id,"
                 " min: META.ct.min, max: META.ct.max, voxels: META.ct.voxels}, staged: document.body.classList.contains('stage1'), three: !!document.querySelector('[data-v=\"three\"]'), mode: st.mode,"
                 " viewOn: document.querySelector('[data-v].on').dataset.v, winOn: document.querySelector('[data-w].on').textContent}; })()")
        R["startup"] = s
        R["timings"].update({"page_" + k: v for k, v in s["timings"].items()})
        ok("2D canvas viewer: no WebGL, no viewer library in the file", s["kind"] == "canvas2d" and not s["webgl"])
        ok("axial view first", s["st"]["view"] == "axial" and s["viewOn"] == "axial")
        ok("plain CT at start-up: no outline decoded or shown", s["chipsOn"] == 0 and s["st"]["active"] is None and s["st"]["overlayBytes"] == 0, f"{s['chips']} outline buttons, all off")
        ok("lung window at start-up", s["st"]["win"] == [-1350, 150] and s["winOn"] == "Lung")
        ok("opens in the single axial view; a '3 planes' button is available", s["three"] and s["mode"] == "single")
        ok("scan verified in the browser (checksum" + (" and SHA-256, " + str(s["verified"]["mode"]) if s["verified"]["sha256"] is True else "") + ")",
           s["verified"]["checksum"] is True and (s["verified"]["sha256"] is True or args.hash == "none"), str(s["verified"]))
        ok("compressed CT text removed from the page after decoding", s["ctBlocksLeft"] == 0)
        ok("no viewer error at start-up", not s["errors"], s["errors"] or None)
        dims = s["st"]["dims"]
        rect = c.js("window.__viewer.canvasRect()")
        cx, cy = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
        staged = s["staged"]

        # ---- trusted comparison: SimpleITK reads the ORIGINAL scan; nothing from the page builder is used ----
        cid = s["meta"]["case_id"]
        src = ROOT / "data" / "raw" / "ct_rate" / "volumes" / f"{cid}.nii.gz"
        img = sitk.DICOMOrient(sitk.ReadImage(str(src)), "LPS")
        ref = sitk.GetArrayViewFromImage(img)                                  # [z, y, x]
        cmp_n, cmp_bad = 0, []
        for view, fr in (("axial", 0.2), ("axial", 0.5), ("axial", 0.8), ("coronal", 0.35), ("coronal", 0.6), ("sagittal", 0.3), ("sagittal", 0.7)):
            n_img = c.js(f"window.__viewer.images('{view}')")
            sl = int(fr * (n_img - 1))
            got = arr16(c.js(f"window.__viewer.sliceHU('{view}', {sl})"))
            gi = grid_from_sitk(img, b, view, sl, got.shape[1], got.shape[0])
            want = np.clip(ref[gi[..., 2], gi[..., 1], gi[..., 0]], -1024, 3071).astype(np.int16)
            cmp_n += 1
            if not np.array_equal(got, want):
                cmp_bad.append((view, sl, int((got != want).sum())))
        ok(f"{cmp_n} drawn slices (axial, coronal, sagittal) equal SimpleITK's reading of the original file, voxel for voxel (after the documented -1024..3071 clip)",
           not cmp_bad, str(cmp_bad) if cmp_bad else f"original stored as {s['meta']['codes']}, obliquity {s['meta']['obl']} deg")
        # display convention against the reference in physical terms: in LPS, +x = Left and +y = Posterior
        e = c.js("(() => { const v = window.__viewer; const a = v.pixelMm('axial', 0, 0, 0), b2 = v.pixelMm('axial', 0, 5, 0), d = v.pixelMm('axial', 0, 0, 5), k = v.pixelMm('coronal', 0, 0, 0), k2 = v.pixelMm('coronal', 0, 0, 5),"
                 " g = v.pixelMm('sagittal', 0, 0, 0), g2 = v.pixelMm('sagittal', 0, 5, 0); return {colX: b2[0] - a[0], rowY: d[1] - a[1], corZ: k2[2] - k[2], sagY: g2[1] - g[1]}; })()")
        ok("screen directions from physical coordinates: axial left = patient Right and top = Anterior; coronal top = Superior; sagittal left = Anterior",
           e["colX"] < 0 and e["rowY"] < 0 and e["corZ"] < 0 and e["sagY"] < 0, str({k: round(v, 2) for k, v in e.items()}))
        # anatomical landmarks (supplementary only): heart mostly on the patient's left = screen right; liver on screen left
        seg = ROOT / "data" / "derived" / cid / "segmentation" / "total"
        lm = {}
        for organ in ("heart", "liver"):
            f = seg / f"{organ}.nii.gz"
            if f.exists():
                mk = sitk.DICOMOrient(sitk.ReadImage(str(f)), "LPS")
                ar = sitk.GetArrayViewFromImage(mk)
                zyx = np.argwhere(ar > 0).mean(0)
                p = mk.TransformContinuousIndexToPhysicalPoint((float(zyx[2]), float(zyx[1]), float(zyx[0])))
                fx = c.js(f"(async () => {{ const v = window.__viewer; v.setView('axial'); v.goToMm([{-p[0]}, {-p[1]}, {p[2]}]); await new Promise(r => setTimeout(r, 80));"
                          " const s = v.screenOf(v.state().cur), L = v.layout(); return (s[0] - L.x) / L.w; })()")
                lm[organ] = round(fx, 3)
        if lm:
            ok("landmarks (supplementary): heart centre on the screen-right half (patient's left), liver centre on the screen-left half",
               lm.get("heart", 1) > 0.5 and lm.get("liver", 0) < 0.5, str(lm))
        c.js("(() => { const v = window.__viewer; v.setView('axial'); v.setCur([v.state().dims[0] >> 1, v.state().dims[1] >> 1, v.state().dims[2] >> 1]); return 1; })()")

        # ---- interaction through real input events ----
        def wheel(n, dy=120, mod=0):
            for _ in range(n):
                c.call("Input.dispatchMouseEvent", type="mouseWheel", x=cx, y=cy, deltaX=0, deltaY=dy, modifiers=mod)
        st0 = c.js("window.__viewer.state()")
        wheel(10)
        time.sleep(0.4)
        st1 = c.js("window.__viewer.state()")
        ok("mouse wheel: one image per notch, downwards = next image (towards the feet in axial)", st1["imageNo"] - st0["imageNo"] == 10 and st1["cur"][2] == st0["cur"][2] - 10,
           f"image {st0['imageNo']} -> {st1['imageNo']} of {st1['images']}")
        wheel(10, -120)
        c.js("document.getElementById('gl').focus()")
        for key in ("ArrowDown", "ArrowDown", "PageDown", "ArrowUp"):
            c.call("Input.dispatchKeyEvent", type="keyDown", key=key, code=key, windowsVirtualKeyCode={"ArrowDown": 40, "ArrowUp": 38, "PageDown": 34}[key])
            c.call("Input.dispatchKeyEvent", type="keyUp", key=key, code=key, windowsVirtualKeyCode={"ArrowDown": 40, "ArrowUp": 38, "PageDown": 34}[key])
        time.sleep(0.3)
        st2 = c.js("window.__viewer.state()")
        ok("arrow and page keys move through the images", st2["imageNo"] - st0["imageNo"] == 11, f"image {st0['imageNo']} -> {st2['imageNo']}")
        w = c.js("(() => { const out = []; for (const b of document.querySelectorAll('[data-w]')) { b.click(); out.push(window.__viewer.state().win); } document.querySelector('[data-w]').click(); return out; })()")
        ok("window presets", w == [[-1350, 150], [-160, 240], [-500, 1300]], str(w))

        def drag(x0, y0, x1, y1, button="left", mod=0):
            bt = {"left": 1, "right": 2}[button]
            c.call("Input.dispatchMouseEvent", type="mousePressed", x=x0, y=y0, button=button, buttons=bt, clickCount=1, modifiers=mod)
            for t in range(1, 9):
                c.call("Input.dispatchMouseEvent", type="mouseMoved", x=x0 + (x1 - x0) * t / 8, y=y0 + (y1 - y0) * t / 8, button=button, buttons=bt, modifiers=mod)
            c.call("Input.dispatchMouseEvent", type="mouseReleased", x=x1, y=y1, button=button, buttons=0, clickCount=1, modifiers=mod)
            time.sleep(0.15)
        drag(cx, cy, cx + 80, cy + 40, "right")
        wr = c.js("window.__viewer.state().win")
        drag(cx, cy, cx - 60, cy, "left", 8)                     # Shift + drag = the laptop alternative
        ws = c.js("window.__viewer.state().win")
        ok("contrast: right-drag and Shift+drag change the window", wr != [-1350, 150] and ws != wr and (wr[1] - wr[0]) > 1500 and (ws[1] - ws[0]) < (wr[1] - wr[0]), f"{[round(x) for x in wr]} then {[round(x) for x in ws]}")
        c.js("document.querySelector('[data-w]').click()")
        v = c.js("(() => { const v = window.__viewer, out = []; for (const b of document.querySelectorAll('[data-v]:not([data-v=\"three\"])')) { b.click(); out.push(v.state().view + ':' + v.state().images); } document.querySelector('[data-v]').click(); return out; })()")
        ok("axial, coronal and sagittal views", v == [f"axial:{dims[2]}", f"coronal:{dims[1]}", f"sagittal:{dims[0]}"], str(v))
        wheel(4, -120, 2)                                        # Ctrl + wheel = zoom in (also what a touchpad pinch sends)
        time.sleep(0.3)
        z1 = c.js("window.__viewer.state()")
        drag(cx, cy, cx + 70, cy - 30, "left")
        z2 = c.js("window.__viewer.state()")
        c.js("document.getElementById('zin').click(); document.getElementById('zout').click(); document.getElementById('zout').click()")
        z3 = c.js("window.__viewer.state().zoom")
        c.js("document.getElementById('zfit').click()")
        z4 = c.js("window.__viewer.state()")
        ok("zoom (Ctrl+wheel and buttons), pan (drag), Fit resets", z1["zoom"] > 1.5 and z2["pan"] != z1["pan"] and abs(z2["pan"][0] - z1["pan"][0]) > 20 and z3 < z1["zoom"] and z4["zoom"] == 1 and z4["pan"] == [0, 0],
           f"zoom {z1['zoom']:.2f}, pan {[round(x) for x in z2['pan']]}, after Fit {z4['zoom']}")
        # zoom is relative to fit; "1:1 pixels" must give one image pixel per screen pixel; smoothing can be switched off
        z5 = c.js("(async () => { const v = window.__viewer, out = {}; for (const vw of ['axial', 'coronal', 'sagittal']) { v.setView(vw); document.getElementById('z11').click(); await new Promise(r => requestAnimationFrame(r)); const s = v.state(), L = v.layout();"
                  " out[vw] = {px: s.pxScale, zoom: s.zoom, aspect: L.w / L.h, want: (L.cols * L.pw) / (L.rows * L.ph)}; } v.setView('axial'); const fit = v.state().pxScale; const sm0 = v.state().smooth; document.getElementById('smooth').click();"
                  " await new Promise(r => requestAnimationFrame(r)); const sm1 = v.state().smooth, txt = document.getElementById('loc').textContent; document.getElementById('smooth').click(); await new Promise(r => requestAnimationFrame(r));"
                  " return {views: out, fit, sm0, sm1, sm2: v.state().smooth, txt, txt2: document.getElementById('loc').textContent}; })()")
        ok("'1:1 pixels': one image pixel per screen pixel across the image in every view, physical aspect ratio kept; zoom read-out is relative to fit",
           all(abs(x["px"][0] - 1) < 1e-6 and abs(x["aspect"] - x["want"]) < 1e-6 * x["want"] for x in z5["views"].values()) and "% of fit (1 image pixel = " in z5["txt2"],
           f"fit = {z5['fit'][0]:.2f} screen pixels per image pixel; 1:1 = {round(100 * z5['views']['axial']['zoom'])}% of fit")
        ok("'Smooth' toggle: off draws unsmoothed pixels for inspection, and the read-out says so", z5["sm0"] is True and z5["sm1"] is False and z5["sm2"] is True and "not smoothed" in z5["txt"] and "not smoothed" not in z5["txt2"])
        # click sets the cross-hair; the same physical point is kept when the view changes
        L = c.js("window.__viewer.layout()")
        c.call("Input.dispatchMouseEvent", type="mousePressed", x=L["x"] + L["w"] * 0.3, y=L["y"] + L["h"] * 0.4, button="left", buttons=1, clickCount=1)
        c.call("Input.dispatchMouseEvent", type="mouseReleased", x=L["x"] + L["w"] * 0.3, y=L["y"] + L["h"] * 0.4, button="left", buttons=0, clickCount=1)
        time.sleep(0.2)
        k1 = c.js("(() => { const v = window.__viewer, a = v.state().cur.slice(), s = v.screenOf(a), L = v.layout(); v.setView('coronal'); const b2 = v.state(); v.setView('axial'); return {cur: a, fx: (s[0] - L.x) / L.w, fy: (s[1] - L.y) / L.h, cor: b2.cur, corImages: b2.images}; })()")
        ok("click moves the cross-hair to the clicked voxel; the position is kept across views", abs(k1["fx"] - 0.3) < 0.01 and abs(k1["fy"] - 0.4) < 0.01 and k1["cor"] == k1["cur"], f"voxel {k1['cur']}")

        # ---- three linked planes ----
        c.js("(() => { const v = window.__viewer; v.setView('axial'); v.setCur([v.state().dims[0] >> 1, v.state().dims[1] >> 1, v.state().dims[2] >> 1]); document.querySelector('[data-v=\"three\"]').click(); return 1; })()")
        time.sleep(0.4)
        t3 = c.js("(() => { const v = window.__viewer, s = v.state(), out = {mode: s.mode, panes: s.panes, cur: s.cur, views: {}}; for (const p of s.panes) { const L = v.layout(p.view), sc = v.screenOf(s.cur, p.view);"
                  " out.views[p.view] = {inside: L.x >= p.x - 1 && L.y >= p.y - 1 && L.x + L.w <= p.x + p.w + 1 && L.y + L.h <= p.y + p.h + 1, aspect: L.w / L.h, want: (L.cols * L.pw) / (L.rows * L.ph), sc: sc, L: L}; } return out; })()")
        pv = {p["view"]: p for p in t3["panes"]}
        nolap = all(a["x"] + a["w"] <= b["x"] + 1 or b["x"] + b["w"] <= a["x"] + 1 or a["y"] + a["h"] <= b["y"] + 1 or b["y"] + b["h"] <= a["y"] + 1
                    for a in t3["panes"] for b in t3["panes"] if a["view"] < b["view"])
        ok("3 planes: axial, coronal and sagittal shown together, panes do not overlap, each image inside its pane with the physical aspect ratio",
           t3["mode"] == "three" and sorted(pv) == ["axial", "coronal", "sagittal"] and nolap and all(x["inside"] and abs(x["aspect"] - x["want"]) < 1e-6 * x["want"] for x in t3["views"].values()))
        (out / "three_planes.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
        # a click in ONE pane must move the cross-hair in all three to the same physical point
        La = t3["views"]["axial"]["L"]
        ax_x, ax_y = La["x"] + La["w"] * 0.62, La["y"] + La["h"] * 0.35
        c.call("Input.dispatchMouseEvent", type="mousePressed", x=ax_x, y=ax_y, button="left", buttons=1, clickCount=1)
        c.call("Input.dispatchMouseEvent", type="mouseReleased", x=ax_x, y=ax_y, button="left", buttons=0, clickCount=1)
        time.sleep(0.25)
        l1 = c.js("(() => { const v = window.__viewer, s = v.state(), o = {cur: s.cur, focus: s.focus, sc: {}}; for (const p of s.panes) { const L = v.layout(p.view), q = v.screenOf(s.cur, p.view); o.sc[p.view] = [(q[0] - L.x) / L.w, (q[1] - L.y) / L.h]; }"
                  " o.mmCor = v.pixelMm('coronal', s.cur[1], s.dims[0] - 1 - s.cur[0], s.dims[2] - 1 - s.cur[2]); o.mmSag = v.pixelMm('sagittal', s.cur[0], s.dims[1] - 1 - s.cur[1], s.dims[2] - 1 - s.cur[2]); o.mm = v.voxToMm(s.cur); return o; })()")
        same = all(abs(l1["mm"][k] - l1["mmCor"][k]) < 1e-6 and abs(l1["mm"][k] - l1["mmSag"][k]) < 1e-6 for k in range(3))
        ok("3 planes: a click in the axial pane moves the cross-hair there, keeps the axial image, and the coronal and sagittal panes show the same physical point",
           abs(l1["sc"]["axial"][0] - 0.62) < 0.01 and abs(l1["sc"]["axial"][1] - 0.35) < 0.01 and l1["cur"][2] == t3["cur"][2] and l1["cur"][:2] != t3["cur"][:2] and same and l1["focus"] == "axial",
           f"voxel {t3['cur']} -> {l1['cur']}")
        # wheel over the coronal pane scrolls coronal only (front-back), and the other panes follow the cross-hair
        pc = pv["coronal"]
        for _ in range(5):
            c.call("Input.dispatchMouseEvent", type="mouseWheel", x=pc["x"] + pc["w"] / 2, y=pc["y"] + pc["h"] / 2, deltaX=0, deltaY=120)
        time.sleep(0.3)
        l2 = c.js("(() => { const s = window.__viewer.state(); return {cur: s.cur, focus: s.focus, view: s.view}; })()")
        ok("3 planes: the wheel scrolls the pane under the pointer (coronal: 5 images towards the back), the other two axes stay", l2["cur"] == [l1["cur"][0], l1["cur"][1] - 5, l1["cur"][2]] and l2["focus"] == "coronal", f"{l1['cur']} -> {l2['cur']}")
        # the three panes draw the same pixels as the single views (same code path), with an outline switched on
        k0 = c.js("Array.from(document.querySelectorAll('#ovbar .ov')).map(e => e.dataset.k)[0] || null")
        if k0:
            o3 = c.js(f"(async () => {{ const v = window.__viewer; await v.setOverlay('{k0}'); await new Promise(r => setTimeout(r, 80)); const s = v.state(); await v.setOverlay(null); return {{active: s.active, mode: s.mode, bytes: s.overlayBytes}}; }})()", 120)
            ok("3 planes: one outline at a time still holds", o3["active"] == k0 and o3["mode"] == "three" and o3["bytes"] > 0)
        z3 = c.js("(async () => { const v = window.__viewer; document.getElementById('z11').click(); await new Promise(r => requestAnimationFrame(r)); const a = v.state(); document.getElementById('zfit').click(); await new Promise(r => requestAnimationFrame(r));"
                  " const b2 = v.state(); document.querySelector('[data-v=\"axial\"]').click(); await new Promise(r => requestAnimationFrame(r)); const d = v.state(); return {px: a.pxScale, focus: a.focus, zoomFit: b2.zoom, back: d.mode, view: d.view, panes: d.panes.length}; })()")
        ok("3 planes: zoom controls act on the pane last used; the Axial button returns to the single view", abs(z3["px"][0] - 1) < 1e-6 and z3["zoomFit"] == 1 and z3["back"] == "single" and z3["view"] == "axial" and z3["panes"] == 1, str(z3["focus"]))
        c.js("(() => { const v = window.__viewer; v.setView('axial'); v.setCur([v.state().dims[0] >> 1, v.state().dims[1] >> 1, v.state().dims[2] >> 1]); return 1; })()")
        # ---- outlines ----
        keys = c.js("Array.from(document.querySelectorAll('#ovbar .ov')).map(e => e.dataset.k)")
        n_sw, bad, maxbytes = 0, [], 0
        t0 = time.time()
        rounds = max(1, -(-args.switches // max(len(keys), 1)))
        for r in range(rounds):
            for k in keys:
                o = c.js(f"(async () => {{ const v = window.__viewer; await v.setOverlay('{k}'); await new Promise(r => setTimeout(r, 30)); const s = v.state();"
                         " return {active: s.active, bytes: s.overlayBytes, on: document.querySelectorAll('#ovbar .ov:not(.off)').length, err: window.__viewerErrors.length, want: META.vols['" + k + "'].raw_bytes}; })()", 120)
                n_sw += 1
                maxbytes = max(maxbytes, o["bytes"])
                if o["active"] != k or o["on"] != 1 or o["err"] or o["bytes"] != o["want"]:
                    bad.append((k, o))
        R["timings"]["outline_switch_mean_s"] = round((time.time() - t0) / max(n_sw, 1), 3)
        ok(f"{n_sw} outline switches: exactly one decoded outline at any time, previous one released", not bad, f"largest decoded outline {maxbytes / 1e6:.1f} MB" + (f"; problems {bad[:2]}" if bad else ""))
        # outline alignment against the original mask files, read by SimpleITK
        mb, mn = [], 0
        for key, rel in (("pleural", "pleural_pericard_effusion/pleural_effusion"), ("pericardial", "pleural_pericard_effusion/pericardial_effusion"), ("heart", "total/heart")):
            f = ROOT / "data" / "derived" / cid / "segmentation" / f"{rel}.nii.gz"
            if key not in keys or not f.exists():
                continue
            mk = sitk.DICOMOrient(sitk.ReadImage(str(f)), "LPS")
            mref = sitk.GetArrayViewFromImage(mk)
            t = c.js(f"(async () => {{ const v = window.__viewer; await v.setOverlay('{key}'); v.setView('axial'); return v.goToMm(META.targets['{key}']); }})()", 120)
            for view, sl in (("axial", t[2]), ("coronal", t[1]), ("sagittal", t[0])):
                got = arr8(c.js(f"window.__viewer.sliceMask('{view}', {sl})"))
                gi = grid_from_sitk(mk, b, view, sl, got.shape[1], got.shape[0])
                want = (mref[gi[..., 2], gi[..., 1], gi[..., 0]] > 0).astype(np.uint8)
                mn += 1
                if not np.array_equal(got, want):
                    mb.append((key, view, sl, int((got != want).sum())))
        if mn:
            ok(f"{mn} drawn outline slices equal the original mask files read by SimpleITK, pixel for pixel", not mb, str(mb) if mb else None)
        # 'Show in scan': the physical target must select the image that contains it
        jumps = c.js("(async () => { const v = window.__viewer, out = []; await v.setOverlay(null); for (const tr of document.querySelectorAll('tr.jump')) { const k = tr.dataset.jump, t = META.targets[k]; if (!t) continue; v.setView('axial'); tr.click();"
                     " await new Promise(r => setTimeout(r, 150)); await v.idle(); await new Promise(r => setTimeout(r, 60)); const s = v.state(), mm = v.voxToMm(s.cur), sp = s.spacing;"
                     " out.push({k, active: s.active, sel: tr.classList.contains('sel'), err: [Math.abs(mm[0] - t[0]) / sp[0], Math.abs(mm[1] - t[1]) / sp[1], Math.abs(mm[2] - t[2]) / sp[2]]}); } await v.setOverlay(null); return out; })()", 300)
        if staged:
            ok("staged page, step 1: 'Show in scan' does nothing before the lock", all(not j["sel"] and j["active"] is None for j in jumps), f"{len(jumps)} rows")
        elif jumps:
            ok(f"'Show in scan' on {len(jumps)} rows: outline on, cross-hair within half a voxel of the physical target on every axis",
               all(j["active"] == j["k"] and j["sel"] and max(j["err"]) <= 0.5001 for j in jumps), str([(j["k"], round(max(j["err"]), 2)) for j in jumps]))
        off = c.js("(async () => { const v = window.__viewer; await v.setOverlay(null); const s = v.state(); return {active: s.active, bytes: s.overlayBytes, on: document.querySelectorAll('#ovbar .ov:not(.off)').length}; })()")
        ok("outline off returns to the plain CT and frees the decoded outline", off == {"active": None, "bytes": 0, "on": 0}, str(off))

        # ---- sustained load: many window / contrast changes, scrolling, view switches ----
        t0 = time.time()
        c.js(f"(async () => {{ const v = window.__viewer; for (let i = 0; i < {args.window_changes}; i++) {{ v.setWindow(-1350 + (i % 40) * 20, 150 + (i % 25) * 40); if (i % 10 === 0) await new Promise(r => setTimeout(r, 0)); }}"
             " for (const b of ['axial', 'coronal', 'sagittal']) { v.setView(b); for (let i = 0; i < " + str(args.scroll_steps) + "; i++) { v.step(i % 2 ? -3 : 4); if (i % 8 === 0) await new Promise(r => requestAnimationFrame(r)); } }"
             " for (let i = 0; i < 30; i++) { v.setView(['axial', 'coronal', 'sagittal', 'three'][i % 4]); await new Promise(r => requestAnimationFrame(r)); } v.setView('three'); for (let i = 0; i < 60; i++) { v.step(i % 2 ? -2 : 3, ['axial', 'coronal', 'sagittal'][i % 3]); v.setWindow(-1350 + (i % 20) * 30, 150 + (i % 10) * 60); if (i % 6 === 0) await new Promise(r => requestAnimationFrame(r)); } v.setView('axial'); document.querySelector('[data-w]').click(); return 1; })()", 600)
        wheel(60)
        wheel(60, -120)
        for _ in range(10):
            drag(cx, cy, cx + 60, cy + 30, "right")
        c.js("document.querySelector('[data-w]').click(); document.getElementById('zfit').click()")
        R["timings"]["interaction_block_s"] = round(time.time() - t0, 1)
        time.sleep(1)
        after = b.gc_mem()
        R["memory"]["sustained_after_interaction"] = after
        base_m = R["memory"]["sustained_after_startup"]
        R["memory"]["retained_growth"] = {k: after[k] - base_m[k] for k in ("private_mb", "working_set_mb", "renderer_ws_mb", "gpu_process_ws_mb")}
        R["memory"]["gpu_used_delta_mb"] = None if g0 is None else gpu_mb() - g0
        ok(f"no viewer error after {args.window_changes} window changes, {3 * args.scroll_steps + 120} scroll steps, 30 view switches, {n_sw} outline switches",
           not c.js("window.__viewerErrors.length") and c.js("window.__viewer.ready"), f"retained growth {R['memory']['retained_growth']['private_mb']} MB private")
        (out / "after_interaction.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))

        # ---- grading form: the location choices are visible before any answer, and active only when they apply ----
        if not c.js("!!document.querySelector('#chk td.grade .gb button.sel')"):
            lf = c.js("(() => { const td = document.querySelector('#chk td.grade'), gl = td.querySelector('.gl'), vis = e => !!e && e.offsetParent !== null; const out = {shown: vis(gl), lobes: gl.querySelectorAll('.lb').length,"
                      " idle0: gl.classList.contains('idle'), sel0: gl.querySelectorAll('.sel').length, hint0: td.querySelector('.glhint') ? td.querySelector('.glhint').textContent : ''};"
                      " td.querySelector('.gb button[data-a=\"present\"]').click(); out.idle1 = gl.classList.contains('idle'); out.def1 = gl.querySelector('.lx[data-s=\"not_assessed\"]').classList.contains('sel');"
                      " td.querySelector('.gb button[data-a=\"absent\"]').click(); out.idle2 = gl.classList.contains('idle'); out.hint2 = td.querySelector('.glhint') ? td.querySelector('.glhint').textContent : '';"
                      " td.querySelector('.gb button[data-a=\"absent\"]').click(); out.sides = !!document.querySelector('#chk .lb[data-l=\"Right\"]'); return out; })()")
            ok("grading form: lobe choices (RUL RML RLL LUL LLL, Indeterminate, Not assessed) are visible under every lung condition before answering; active for Present / Can't tell",
               lf["shown"] and lf["lobes"] == 5 and lf["idle0"] and lf["sel0"] == 0 and not lf["idle1"] and lf["def1"] and lf["idle2"] and "absent" in lf["hint2"], lf["hint0"])
        tz = c.js("(() => { const side = document.querySelector('.side'); if (!side) return null; const z0 = parseFloat(getComputedStyle(side).zoom), f0 = window.__textZoom(); document.getElementById('tzin').click();"
                  " const z1 = parseFloat(getComputedStyle(side).zoom), saved = localStorage.getItem('lungct_text_zoom'); document.getElementById('tzout').click(); document.getElementById('tzout').click(); document.getElementById('tzout').click();"
                  " const z2 = parseFloat(getComputedStyle(side).zoom); document.getElementById('tzin').click(); document.getElementById('tzin').click(); const r = side.getBoundingClientRect();"
                  " return {z0, f0, z1, z2, saved, back: parseFloat(getComputedStyle(side).zoom), label: document.getElementById('tzval').textContent, overflowX: side.scrollWidth > side.clientWidth + 2}; })()")
        if tz:
            ok("larger text: the right-hand panel opens at 120 %, A+ / A- change it (100-180 %) and the choice is remembered; no sideways scrolling at 120 %",
               abs(tz["z0"] - 1.2) < 1e-6 and abs(tz["z1"] - 1.3) < 1e-6 and abs(tz["z2"] - 1.0) < 1e-6 and tz["saved"] == "1.3" and abs(tz["back"] - 1.2) < 1e-6 and tz["label"] == "120%" and not tz["overflowX"], str(tz))
        sb = c.js("(() => { const b = document.getElementById('splitbar'); if (!b) return null; const r = b.getBoundingClientRect(); return {x: r.left + r.width / 2, y: r.top + 300, pct: window.__split(), w: window.innerWidth, vw: document.querySelector('.viewer').getBoundingClientRect().width}; })()")
        if sb:
            x1 = sb["w"] * 0.65
            c.call("Input.dispatchMouseEvent", type="mousePressed", x=sb["x"], y=sb["y"], button="left", buttons=1, clickCount=1)
            for t in range(1, 7):
                c.call("Input.dispatchMouseEvent", type="mouseMoved", x=sb["x"] + (x1 - sb["x"]) * t / 6, y=sb["y"], button="left", buttons=1)
            c.call("Input.dispatchMouseEvent", type="mouseReleased", x=x1, y=sb["y"], button="left", buttons=0, clickCount=1)
            time.sleep(0.4)
            s2 = c.js("(() => { const o = {pct: window.__split(), vw: document.querySelector('.viewer').getBoundingClientRect().width, saved: localStorage.getItem('lungct_split_pct'), w: window.innerWidth, ready: window.__viewer.ready, cw: document.getElementById('gl').width};"
                      " const b = document.getElementById('splitbar'); b.dispatchEvent(new MouseEvent('dblclick', {bubbles: true})); o.back = window.__split(); return o; })()")
            ok("adjustable split: dragging the bar between the scan and the panel resizes both (25-75 %), the choice is remembered, double-click restores half and half",
               abs(sb["pct"] - 50) < 0.01 and abs(s2["pct"] - 65) < 1 and abs(s2["vw"] - 0.65 * s2["w"]) < 12 and s2["saved"] is not None and s2["back"] == 50 and s2["ready"], f"{sb['pct']}% -> {s2['pct']:.1f}% -> {s2['back']}%")
            time.sleep(0.3)
        cards = c.js("(() => { const side = document.querySelector('.side'); if (!side) return null; const d = document.getElementById('measdet'); const top = e => e ? e.getBoundingClientRect().top + side.scrollTop : null;"
                     " const gt = Array.from(side.querySelectorAll('.card')).find(x => /CT-RATE images and report text/.test(x.textContent)), rep = Array.from(side.querySelectorAll('.card')).find(x => x.firstElementChild && x.firstElementChild.textContent === 'Report');"
                     " const o = {closed: d && !d.open, yGt: top(gt), yMeas: top(document.getElementById('measpanel')), yRep: top(rep), hasGt: !!(gt && gt.querySelector('details'))};"
                     " document.querySelector('.panelbtn').click(); o.openedByButton = d.open; d.querySelector('summary').click(); o.closedBySummary = !d.open; d.querySelector('summary').click(); o.reopened = d.open; o.rows = d.querySelectorAll('table tr').length; d.open = false; return o; })()")
        if cards:
            ok("cards under the checklist: radiologist report first, then the measurement panel, then the tool's report; the measurement panel is closed at first and opens / closes on click and from the 'Open measurement panel' button",
               cards["closed"] and cards["yGt"] < cards["yMeas"] < cards["yRep"] and cards["openedByButton"] and cards["closedBySummary"] and cards["reopened"] and cards["rows"] > 10, f"radiologist report in page: {cards['hasGt']}")
        # ---- grading and CSV ----
        g = c.js("(async () => { const rev = document.getElementById('grev'); const prefilled = !!document.querySelector('#chk td.grade .gb button.sel'); rev.value = window.__TEST_REVIEWER; rev.oninput();"
                 " const rows = Array.from(document.querySelectorAll('#chk tr')).slice(1); let lobes = 0;"
                 " rows.forEach((tr, i) => { const td = tr.querySelector('td.grade'); const want = td.querySelector('.gb button.sel') || td.querySelector('.gb button[data-a=\"' + ['present', 'absent', 'cannot_tell'][i % 3] + '\"]'); if (!want.classList.contains('sel')) want.click();"
                 "   const lb = td.querySelector('.lb[data-l=\"RLL\"]'); if (i === 0 && lb && !lb.closest('.gl').classList.contains('idle') && !lb.classList.contains('sel')) { lb.click(); lobes++; } });"
                 " const lock = document.getElementById('glock'); if (lock) lock.click(); await new Promise(r => setTimeout(r, 200));"
                 " document.getElementById('gsave').click(); await new Promise(r => setTimeout(r, 1200));"
                 " return {rows: rows.length, lobes, csv: window.__csv, graded: document.getElementById('gcount').textContent, prefilled, stagedNow: document.body.classList.contains('stage1')}; })()", 120)
        csv_text = (g["csv"] or "").lstrip("﻿")
        lines = csv_text.split("\r\n")
        (out / "exported_grades.csv").write_text(csv_text, encoding="utf-8", newline="")
        ok("grades entered for every condition and autosaved", g["graded"].startswith(f"{g['rows']} of {g['rows']}"), g["graded"])
        ok("CSV export: header + one row per condition + overall row", len(lines) == g["rows"] + 2 and lines[0].startswith("case_id,reviewer,saved_utc,condition_key") and "checklist_viewer_v4_canvas2d" in lines[1], f"{len(lines)} lines")
        if staged:
            ok("staged page: locking step 1 reveals the tool's output", g["stagedNow"] is False)
        R["grading"] = {"rows": g["rows"], "lobes": g["lobes"], "prefilled": g["prefilled"]}
    finally:
        b.close()


def run_damaged(args, out: Path, R: dict, ok) -> None:
    """A copy of the page with a damaged CT block must show an error and leave grading and CSV working."""
    page = (args.page if args.page.is_absolute() else ROOT / args.page).resolve()
    txt = page.read_text(encoding="utf-8")
    a = txt.index('class="vp" data-vol="ct" data-part="1">') + len('class="vp" data-vol="ct" data-part="1">')
    bad = txt[:a + 5000] + ("A" * 64) + txt[a + 5064:]
    d = ROOT / "runs" / "test_logs" / "damaged_v4"
    d.mkdir(parents=True, exist_ok=True)
    f = d / ("damaged_" + page.name)
    f.write_text(bad, encoding="utf-8")
    del txt, bad
    b = Browser(args.browser, args.gpu_mode, f.as_uri(), args.port, args.headed)
    try:
        _, st = wait_ready(b, 300)
        ok("damaged scan data: the page says the scan could not be loaded (no silent wrong image)", (not st["ready"]) and bool(st["err"]) and "could not be loaded" in st["err"], (st["err"] or "")[:140])
        b.c.js(PREP)
        g = b.c.js("(async () => { const rev = document.getElementById('grev'); rev.value = window.__TEST_REVIEWER; rev.oninput(); const rows = Array.from(document.querySelectorAll('#chk tr')).slice(1);"
                   " rows.forEach(tr => { const w = tr.querySelector('td.grade .gb button[data-a=\"absent\"]'); if (!w.classList.contains('sel')) w.click(); });"
                   " const nb = document.querySelector('.mbtn'); const lock = document.getElementById('glock'); if (lock) lock.click(); await new Promise(r => setTimeout(r, 200)); if (nb) nb.click();"
                   " document.getElementById('gsave').click(); await new Promise(r => setTimeout(r, 1000));"
                   " return {graded: document.getElementById('gcount').textContent, rows: rows.length, lines: (window.__csv || '').split('\\r\\n').length, numbers: nb ? !nb.nextElementSibling.hidden : null,"
                   " reload: !!document.querySelector('#vstat button')}; })()", 60)
        ok("grading, 'Numbers' and CSV export still work when the scan viewer failed", g["graded"].startswith(f"{g['rows']} of {g['rows']}") and g["lines"] == g["rows"] + 2 and g["numbers"] is not False and g["reload"], str(g))
    finally:
        b.close()
        shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--damaged", action="store_true", help="with --page: test a deliberately damaged copy instead")
    ap.add_argument("--browser", choices=list(BROWSERS), default="edge")
    ap.add_argument("--gpu-mode", choices=list(GPU_FLAGS), default="igpu")
    ap.add_argument("--headed", action="store_true", help="visible browser window instead of headless")
    ap.add_argument("--hash", choices=["chunks", "whole", "none"], default="chunks", help="integrity-check mode, to measure its memory cost")
    ap.add_argument("--switches", type=int, default=56)
    ap.add_argument("--window-changes", type=int, default=200)
    ap.add_argument("--scroll-steps", type=int, default=150)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--port", type=int, default=9391)
    args = ap.parse_args()
    if not args.synthetic and not args.page:
        ap.error("use --synthetic or --page")
    name = "synthetic" if args.synthetic else ("damaged_" if args.damaged else "") + args.page.stem
    tag = f"{args.label + '_' if args.label else ''}{name}_{args.browser}_{args.gpu_mode}{'_headed' if args.headed else ''}{'' if args.hash == 'chunks' else '_hash-' + args.hash}"
    out = ROOT / "runs" / f"V2_canvas_test_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    R = {"browser": args.browser, "gpu_mode": args.gpu_mode, "headed": args.headed, "hash": args.hash, "checks": [], "memory": {}, "timings": {}}

    def ok(name, cond, detail=None):
        R["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
        print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail is not None else ""), flush=True)
    try:
        if args.synthetic:
            run_synthetic(args, out, R, ok)
        elif args.damaged:
            run_damaged(args, out, R, ok)
        else:
            run_page(args, out, R, ok)
    except Exception as e:  # reported, never dropped
        import traceback
        R["error"] = f"{type(e).__name__}: {e}"
        R["traceback"] = traceback.format_exc()
        print("ERROR", R["error"], flush=True)
    R["passed"] = "error" not in R and bool(R["checks"]) and all(x["pass"] for x in R["checks"])
    (out / "result.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    md = [f"# Canvas viewer test: {name} in {R.get('browser_version', args.browser)}, {args.gpu_mode} - {'PASSED' if R['passed'] else 'FAILED'}", "",
          "| check | result | detail |", "|---|---|---|"]
    md += [f"| {x['name']} | {'pass' if x['pass'] else 'FAIL'} | {str(x['detail'] or '')[:220].replace('|', '/')} |" for x in R["checks"]]
    md += ["", "Timings: " + json.dumps(R["timings"]), "", "Memory (MB): " + json.dumps(R["memory"])]
    if "error" in R:
        md += ["", "ERROR: " + R["error"]]
    (out / "result.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"timings": R["timings"], "memory": R["memory"]}))
    print(f"-> {out}")
    return 0 if R["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
