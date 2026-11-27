"""Technical checks of viewer v7 (the bar with the outlines of two segmentation models), run against the UNMODIFIED
html from file:// (core env).

Technical checks only. They show that the page draws what the files hold, in the right place. They say nothing about
whether either segmentation is anatomically right, and nothing about usefulness for reading.

  --synthetic   small volumes with KNOWN label maps, stored in six axis orders / flips. For every button and the three
                planes, the class number drawn at every pixel must equal the label map written here in patient
                orientation, and the colour must equal the stated blend. Also: one decode per label map, release of the
                previous map, the name under the cross-hair, "nothing outlined", slab views, the older single outlines.
  --page FILE --second-seg RUNDIR
                a real case page: every button compared, pixel by pixel on axial, coronal and sagittal images, with the
                ORIGINAL files read by SimpleITK (NV-Segment-CT label map and extra masks, TotalSegmentator masks);
                decode count, memory, speed of drawing with an outline on, slab views, "Show in scan".

Usage: python scripts/test_seg_viewer.py --synthetic --browser edge
       python scripts/test_seg_viewer.py --page runs/M14_.../viewer_valid_301_M14_v3_11.html --second-seg runs/P2_nvsegment_... --browser edge
Writes runs/V12_seg_test_<label>_<time>/{result.json,result.md,*.png}.
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
import viewer_v7 as v7  # noqa: E402
from test_canvas_viewer import GPU_FLAGS, Browser, arr8, expected_display, grid_from_sitk, variant, wait_ready  # noqa: E402
from test_mip_viewer import VARIANTS, phantom  # noqa: E402
from test_viewer_page import BROWSERS  # noqa: E402

ALPHA = 0.55
NAMES = {"28": "left lung upper lobe", "29": "left lung lower lobe", "30": "right lung upper lobe", "31": "right lung middle lobe", "32": "right lung lower lobe",
         "57": "trachea", "115": "heart", "6": "aorta", "1": "liver", "33": "vertebrae L5", "23": "lung tumor", "132": "airway", "128": "bone lesion"}
GROUPS = [{"key": "nv_all", "source": "everything", "label": "All structures", "labels": "all"},
          {"key": "nv_lobes", "source": "everything", "label": "Lung lobes", "labels": [28, 29, 30, 31, 32]},
          {"key": "nv_trachea", "source": "everything", "label": "Trachea", "labels": [57]},
          {"key": "nv_airway", "source": "extra", "label": "Airway", "labels": [132]},
          {"key": "nv_tumor", "source": "extra", "label": "Class 'lung tumor'", "labels": [23]},
          {"key": "nv_extra_rest", "source": "extra", "label": "Other tumour / vessel classes", "labels": [128]}]      # 128 is never written: must be listed as "nothing outlined"
TS_FILES = {28: "total/lung_upper_lobe_left", 29: "total/lung_lower_lobe_left", 30: "total/lung_upper_lobe_right", 31: "total/lung_middle_lobe_right",
            32: "total/lung_lower_lobe_right", 57: "total/trachea", 115: "total/heart", 6: "total/aorta"}


def rgba(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["b64"]), np.uint8).reshape(d["h"], d["w"], 4)


def blend(gray: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Colour the page must draw: grey where no shown class, else grey * 0.45 + class colour * 0.55, rounded half up per channel."""
    out = np.repeat(gray[..., None].astype(np.float64), 3, axis=2)
    for x in np.unique(lab):
        if x:
            c = v7.label_colour(int(x))
            rgb = np.array([int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)], float)
            m = lab == x
            out[m] = np.floor(gray[m][:, None] * (1 - ALPHA) + rgb[None, :] * ALPHA + 0.5)
    return out.astype(np.uint8)


def synthetic_maps(shape):
    """Four label maps in patient orientation (RAS+). The two models differ on purpose, so a mix-up of the maps would show."""
    ts, aw, nv, nx = (np.zeros(shape, np.uint8) for _ in range(4))
    ts[22:36, 6:26, 4:14], ts[22:36, 6:26, 14:26] = 29, 28            # high i = patient Left in RAS+ index order used by the phantom's affine
    ts[4:18, 6:26, 4:12], ts[4:18, 6:26, 12:18], ts[4:18, 6:26, 18:26] = 32, 31, 30
    ts[19:21, 14:18, 16:28], ts[18:24, 20:28, 6:14] = 57, 115
    aw[19:21, 14:18, 12:18], aw[17:19, 13:15, 12:14] = 132, 132
    nv[23:37, 5:25, 4:15], nv[23:37, 5:25, 15:27] = 29, 28            # shifted by one voxel and split at another height
    nv[3:17, 5:25, 4:11], nv[3:17, 5:25, 11:19], nv[3:17, 5:25, 19:27] = 32, 31, 30
    nv[19:21, 14:18, 17:29], nv[17:25, 21:29, 5:13], nv[2:38, 28:31, 0:3], nv[10:30, 0:4, 0:30] = 57, 115, 1, 33
    nx[19:21, 14:18, 10:20], nx[25:29, 10:14, 18:22], nx[27:28, 11:12, 19:20] = 132, 23, 23
    return {"tsseg": ts, "tsaw": aw, "nvseg": nv, "nvsegx": nx}


EXPECT = {"ts_lobes": ("tsseg", [28, 29, 30, 31, 32]), "ts_trachea": ("tsseg", [57]), "ts_airways": ("tsaw", [132]), "ts_heart": ("tsseg", [115]),
          "nv_all": ("nvseg", [1, 28, 29, 30, 31, 32, 33, 57, 115]), "nv_lobes": ("nvseg", [28, 29, 30, 31, 32]), "nv_trachea": ("nvseg", [57]),
          "nv_airway": ("nvsegx", [132]), "nv_tumor": ("nvsegx", [23])}
STATE = ("(() => { const v = window.__viewer, q = id => document.getElementById(id); return {build: v.build, seg: !!v.segBuild, slab: v.slab().mode, body: document.body.classList.contains('slabmode'),"
         " segbar: getComputedStyle(q('segbar')).display, ovbar: getComputedStyle(q('ovbar')).display, active: v.state().active, errors: window.__viewerErrors,"
         " chips: Array.from(document.querySelectorAll('#segbar .ov')).map(e => e.dataset.k), old: Array.from(document.querySelectorAll('#ovbar .ov')).map(e => e.dataset.k),"
         " groups: Array.from(document.querySelectorAll('#segbar .segname')).map(e => e.textContent), none: (q('segnone') || {}).textContent || '', proto: location.protocol, kind: v.kind}; })()")


def check_button(c, key: str, want_vol: np.ndarray, slices: dict, to_display) -> tuple[int, int, int]:
    """Switch one button on and compare class numbers and colours on the given images. Returns (pixels compared, label mismatches, colour mismatches)."""
    c.js(f"(async () => {{ await window.__viewer.setOverlay('{key}'); }})()", 120)
    n = bad = badc = 0
    for view, ss in slices.items():
        for s in ss:
            got = arr8(c.js(f"window.__viewer.sliceLabels('{view}', {s})"))
            want = to_display(want_vol, view, s)
            px = rgba(c.js(f"window.__viewer.sliceRGBA('{view}', {s})"))
            gray = arr8(c.js(f"window.__viewer.sliceGray('{view}', {s})"))
            n += got.size
            bad += int((got != want).sum())
            badc += int((px[..., :3] != blend(gray, want)).any(axis=2).sum()) + int((px[..., 3] != 255).sum())
    return n, bad, badc


def run_synthetic(args, out: Path, R: dict, ok) -> None:
    tmp = out / "pages"
    tmp.mkdir()
    base, mask, aff, _ = phantom(0.75)
    maps = synthetic_maps(base.shape)
    for vi, (perm, flips) in enumerate(VARIANTS):
        arr, a2 = variant(base, aff, perm, flips)
        m2, _ = variant(mask, aff, perm, flips)
        stored = {k: variant(m, aff, perm, flips)[0] for k, m in maps.items()}
        page = tmp / f"seg_v{vi}.html"
        page.write_text(v7.standalone_page(f"seg variant {vi}", arr, a2, {"m": ("test outline", "#ef4444", m2)}, label_maps=stored, nv_groups=GROUPS, names=NAMES), encoding="utf-8")
        b = Browser(args.browser, "sw", page.as_uri(), args.port + vi)
        c = b.c
        tag = f"stored as perm {perm} flips {flips}"
        try:
            _, st = wait_ready(b, 120)
            if not st["ready"]:
                ok(f"{tag}: page loads", False, st["err"])
                continue
            R.setdefault("browser_version", b.version)
            s0 = c.js(STATE)
            ok(f"{tag}: opens as v6 does (Axial 2.5 mm, no outline); both outline bars are absent in the slab view; no error",
               s0["build"] == v7.VIEWER_BUILD and s0["seg"] and s0["slab"] == "mean" and s0["body"] and s0["segbar"] == "none" and s0["ovbar"] == "none" and s0["active"] is None and not s0["errors"], s0["build"])
            blocked = c.js("(async () => { const v = window.__viewer; await v.setOverlay('nv_lobes'); return {a: v.state().active, col: v.shownGray('axial').coloured_pixels, dec: v.segDecodes()}; })()")
            ok(f"{tag}: no segmentation outline can be switched on over a mean slab (nothing decoded, nothing coloured)", blocked == {"a": None, "col": 0, "dec": 0}, blocked)
            c.js("(async () => { window.__viewer.setSlab('native'); await new Promise(r => setTimeout(r, 80)); })()")
            s1 = c.js(STATE)
            ok(f"{tag}: with native slices the bar lists both models and exactly the buttons that have voxels; the empty class is named as nothing outlined",
               s1["segbar"] != "none" and s1["chips"] == list(EXPECT) and s1["old"] == ["m"] and s1["groups"] == [v7.GROUP_TS + ":", v7.GROUP_NV + ":"]
               and s1["none"] == "Nothing outlined: NV-Segment-CT: Other tumour / vessel classes", f"{s1['chips']} | {s1['none']}")
            tot = bad = badc = 0
            sl = {"axial": [0, 5, 12, 16, 22, 29], "coronal": [0, 7, 15, 22, 31], "sagittal": [0, 4, 19, 25, 39]}
            for key, (src, labs) in EXPECT.items():
                want = np.where(np.isin(maps[src], labs), maps[src], 0).astype(np.uint8)
                n, b1, b2 = check_button(c, key, want, sl, expected_display)
                tot, bad, badc = tot + n, bad + b1, badc + b2
            ok(f"{tag}: every button draws exactly its classes at the right pixels in axial, coronal and sagittal images, in the stated colour",
               bad == 0 and badc == 0, f"{tot} pixels over {len(EXPECT)} buttons x 16 images: {bad} class mismatches, {badc} colour mismatches")
            dec = c.js("(async () => { const v = window.__viewer, o = {}; await v.setOverlay(null); const d0 = v.segDecodes();"
                       " await v.setOverlay('nv_lobes'); o.a = v.segDecodes() - d0; o.bytesA = v.state().overlayBytes; await v.setOverlay('nv_all'); o.b = v.segDecodes() - d0; await v.setOverlay('nv_trachea'); o.c = v.segDecodes() - d0;"
                       " await v.setOverlay('ts_lobes'); o.d = v.segDecodes() - d0; o.src = v.segSource(); o.bytesD = v.state().overlayBytes; await v.setOverlay('nv_airway'); o.e = v.segDecodes() - d0; o.bytesE = v.state().overlayBytes;"
                       " await v.setOverlay('m'); o.f = v.segDecodes() - d0; o.m = v.state().active; o.bytesM = v.state().overlayBytes; o.lab = v.segShown(); await v.setOverlay(null); o.off = v.state().overlayBytes; o.vols = META.vols; return o; })()")
            vols = dec["vols"]
            ok(f"{tag}: buttons on the same label map share one decoded copy; another map replaces it; only one map is held; switching off releases it",
               [dec[k] for k in "abcdef"] == [1, 1, 1, 2, 3, 3] and dec["src"] == "tsseg" and dec["bytesA"] == vols["nvseg"]["raw_bytes"] and dec["bytesD"] == vols["tsseg"]["raw_bytes"]
               and dec["bytesE"] == vols["nvsegx"]["raw_bytes"] and dec["bytesM"] == vols["m"]["raw_bytes"] and dec["m"] == "m" and dec["lab"] == [] and dec["off"] == 0, {k: dec[k] for k in "abcdef"})
            mk = arr8(c.js("(async () => { const v = window.__viewer; await v.setOverlay('m'); return v.sliceMask('axial', 12); })()"))
            ok(f"{tag}: the older single outline still draws exactly its mask", np.array_equal(mk, expected_display(mask, "axial", 12)))
            # name under the cross-hair: patient-oriented voxel (30, 10, 20) lies in class 28 of the second model; the page is asked by position in mm
            mm = (aff @ np.array([30, 10, 20, 1.0]))[:3].tolist()
            mm0 = (aff @ np.array([38, 30, 28, 1.0]))[:3].tolist()
            rd = c.js("(async () => { const v = window.__viewer, w = () => new Promise(r => requestAnimationFrame(() => setTimeout(r, 30))), o = {}; await v.setOverlay('nv_lobes');"
                      f" v.goToMm({json.dumps(mm)}); await w(); o.in = document.getElementById('loc').textContent; o.L = v.labelAt(v.state().cur);"
                      f" v.goToMm({json.dumps(mm0)}); await w(); o.out = document.getElementById('loc').textContent; await v.setOverlay('nv_trachea'); v.goToMm({json.dumps(mm)}); await w(); o.hidden = document.getElementById('loc').textContent; return o; }})()")
            ok(f"{tag}: the read-out names the model, the button and the structure under the cross-hair, and says so when none of the shown structures is there",
               "outline: NV-Segment-CT - Lung lobes  ·  at the cross-hair: left lung upper lobe" in rd["in"] and rd["L"] == 28 and "at the cross-hair: none of the shown structures" in rd["out"]
               and "outline: NV-Segment-CT - Trachea  ·  at the cross-hair: none of the shown structures" in rd["hidden"], rd["in"][-110:])
            sb = c.js("(async () => { const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 80)), o = {}; await v.setOverlay('nv_all'); document.getElementById('sl25').click(); await w(); await v.idle();"
                      " o.mean = {a: v.state().active, col: v.shownGray('axial').coloured_pixels, bar: getComputedStyle(document.getElementById('segbar')).display, bytes: v.state().overlayBytes};"
                      " v.setSlab('native'); await w(); await v.setOverlay('ts_lobes'); document.getElementById('slmip').click(); await w(); await v.idle();"
                      " o.mip = {a: v.state().active, col: v.shownGray('axial').coloured_pixels, bar: getComputedStyle(document.getElementById('segbar')).display, bytes: v.state().overlayBytes}; o.errors = window.__viewerErrors; return o; })()")
            ok(f"{tag}: choosing Axial 2.5 mm or MIP 10 mm switches a segmentation outline off, releases it and hides the bar",
               sb["mean"] == {"a": None, "col": 0, "bar": "none", "bytes": 0} and sb["mip"] == {"a": None, "col": 0, "bar": "none", "bytes": 0} and not sb["errors"], sb)
        finally:
            b.close()
    # a page without any label map must look and behave like v6: the bar is not shown
    page = tmp / "seg_none.html"
    page.write_text(v7.standalone_page("no label maps", base, aff, {"m": ("test outline", "#ef4444", mask)}), encoding="utf-8")
    b = Browser(args.browser, "sw", page.as_uri(), args.port + 20)
    try:
        _, st = wait_ready(b, 120)
        s = b.c.js("(async () => { window.__viewer.setSlab('native'); await new Promise(r => setTimeout(r, 80)); return " + STATE + "; })()") if st["ready"] else {}
        ok("a page built without label maps shows no segmentation bar and no error", bool(s) and s["segbar"] == "none" and s["chips"] == [] and s["old"] == ["m"] and not s["errors"], s.get("segbar"))
    finally:
        b.close()


def run_page(args, out: Path, R: dict, ok) -> None:
    import SimpleITK as sitk
    page = (args.page if args.page.is_absolute() else ROOT / args.page).resolve()
    R["page"], R["file_mb"] = str(page.relative_to(ROOT)).replace("\\", "/"), round(page.stat().st_size / 1e6, 1)
    b = Browser(args.browser, args.gpu_mode, page.as_uri(), args.port, args.headed)
    R["browser_version"] = b.version
    c = b.c
    try:
        peak, st = wait_ready(b)
        R["memory"]["peak_while_loading"] = peak
        if not st["ready"]:
            ok("scan loaded", False, st["err"])
            return
        time.sleep(2)
        R["memory"]["after_startup"] = b.gc_mem()
        s0 = c.js(STATE)
        meta = c.js("({case_id: META.case_id, seg: META.seg, vols: Object.fromEntries(Object.entries(META.vols).filter(e => /^(ts|nv)/.test(e[0]))), dims: META.ct.dims, ov: META.overlays.filter(o => o.src)})")
        case, dims = meta["case_id"], meta["dims"]
        R["case_id"], R["label_maps"] = case, {k: {"decoded_mb": round(v["raw_bytes"] / 1e6, 1), "in_file_mb": round(v["gz_bytes"] * 4 / 3 / 1e6, 2), "classes": len(v["labels_present"])} for k, v in meta["vols"].items()}
        R["seg_info"] = meta["seg"]["info"]
        ok("opened from file://; opens exactly as v6 (Axial 2.5 mm, no outline, both outline bars absent); no error",
           s0["proto"] == "file:" and s0["kind"] == "canvas2d" and s0["build"] == v7.VIEWER_BUILD and s0["slab"] == "mean" and s0["segbar"] == "none" and s0["ovbar"] == "none" and s0["active"] is None and not s0["errors"], s0["build"])
        # ---- what the buttons must show, read from the ORIGINAL files with SimpleITK ----
        ct = sitk.ReadImage(str(ROOT / "data" / "raw" / "ct_rate" / "volumes" / f"{case}.nii.gz"))
        sdir = (args.second_seg if args.second_seg.is_absolute() else ROOT / args.second_seg) / case
        cfg = json.loads((sdir.parent / "run_manifest.json").read_text(encoding="utf-8"))["config"]

        def read(path):
            img = sitk.ReadImage(str(path))
            if img.GetSize() != ct.GetSize():
                raise RuntimeError(f"{path}: size differs from the scan")
            return sitk.GetArrayFromImage(img)                     # [z, y, x]
        src = {"nvseg": read(sdir / "nvseg_labels.nii.gz").astype(np.uint8)}
        nx = np.zeros_like(src["nvseg"])
        overlap = np.zeros(nx.shape, np.uint8)
        for x in cfg["viewer_extra_priority"]["order"]:            # the first class in the list is the one shown
            m = read(sdir / "extra" / (re.sub(r"[^a-z0-9]+", "_", cfg["class_names"][str(x)].lower()).strip("_") + ".nii.gz")) > 0
            nx[m & (nx == 0)] = x
            overlap += m
        src["nvsegx"] = nx
        seg = ROOT / "data" / "derived" / case / "segmentation"
        ts = np.zeros_like(nx)
        for x in sorted(TS_FILES):                                 # lower class number kept where two masks overlap
            p = seg / f"{TS_FILES[x]}.nii.gz"
            if p.exists():
                m = read(p) > 0
                ts[m & (ts == 0)] = x
        src["tsseg"] = ts
        src["tsaw"] = (read(seg / "lung_vessels" / "lung_airways.nii.gz") > 0).astype(np.uint8) * 132
        want_btn = {}
        for k, lbl, s_, labs in v7.TS_GROUPS:
            want_btn[k] = (s_, labs)
        for g in cfg["viewer_groups"]:
            s_ = "nvseg" if g["source"] == "everything" else "nvsegx"
            want_btn[g["key"]] = (s_, sorted(int(x) for x in np.unique(src[s_]) if x) if g["labels"] == "all" else g["labels"])
        have = {k: (s_, [x for x in labs if (src[s_] == x).any()]) for k, (s_, labs) in want_btn.items()}
        expect_chips = [k for k, (_, labs) in have.items() if labs]
        expect_none = [k for k, (_, labs) in have.items() if not labs]
        c.js("(async () => { document.getElementById('sldetbtn').click(); document.getElementById('slnat').click(); await new Promise(r => setTimeout(r, 150)); })()")
        s1 = c.js(STATE)
        ok("with native slices the bar lists TotalSegmentator and NV-Segment-CT with exactly the buttons whose classes have voxels in the original files; classes without voxels are named as nothing outlined",
           s1["segbar"] != "none" and s1["chips"] == expect_chips and s1["none"].count(";") + (1 if s1["none"] else 0) == len(expect_none) and len(s1["old"]) >= 1
           and {o["key"]: o["labels"] for o in meta["ov"]} == {k: have[k][1] for k in expect_chips},
           f"{len(expect_chips)} buttons; {s1['none'] or 'no empty class'}")
        ok("the page records which second-model run it shows and that no answer uses it; overlap between the extra classes equals the files",
           meta["seg"]["info"].get("second_model_extra_classes_overlapping_voxels") == int((overlap > 1).sum()) and meta["seg"]["info"].get("second_model") == "nvidia/NV-Segment-CT"
           and "no answer" in meta["seg"]["info"].get("second_model_note", ""), meta["seg"]["info"].get("second_model_extra_classes_overlapping_voxels"))
        (out / "native_no_outline.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
        grids = {}

        def to_display(vol, view, s):
            key = (view, s)
            if key not in grids:
                g = {"axial": (dims[0], dims[1]), "coronal": (dims[0], dims[2]), "sagittal": (dims[1], dims[2])}[view]
                grids[key] = grid_from_sitk(ct, b, view, s, g[0], g[1])
            gi = grids[key]
            return vol[gi[..., 2], gi[..., 1], gi[..., 0]]
        nz = dims[2]
        sl = {"axial": sorted({0, nz // 6, nz // 3, nz // 2, (2 * nz) // 3, (5 * nz) // 6, nz - 1}), "coronal": [dims[1] // 3, dims[1] // 2], "sagittal": [dims[0] // 3, (2 * dims[0]) // 3]}
        rows, tot, bad, badc, t_first = [], 0, 0, 0, {}
        for key in expect_chips:
            s_, labs = have[key]
            want = np.where(np.isin(src[s_], labs), src[s_], 0).astype(np.uint8)
            t0 = time.time()
            n, b1, b2 = check_button(c, key, want, sl, to_display)
            col = sum(int((to_display(want, v_, s) > 0).sum()) for v_, ss in sl.items() for s in ss)
            rows.append({"button": key, "map": s_, "classes": len(labs), "pixels": n, "outlined_pixels_checked": col, "class_mismatches": b1, "colour_mismatches": b2, "seconds": round(time.time() - t0, 1)})
            tot, bad, badc = tot + n, bad + b1, badc + b2
        R["buttons"] = rows
        ok("every button draws exactly the classes of the original files at the right pixels (7 axial, 2 coronal, 2 sagittal images each), in the stated colours; left and right as in the files",
           bad == 0 and badc == 0 and all(r["outlined_pixels_checked"] > 0 for r in rows if r["button"] in ("ts_lobes", "nv_lobes", "nv_all")),
           f"{tot} pixels over {len(rows)} buttons: {bad} class mismatches, {badc} colour mismatches; outlined pixels checked: " + ", ".join(f"{r['button']} {r['outlined_pixels_checked']}" for r in rows))
        # ---- left / right from anatomy of the page itself: the class called 'left ... lobe' must lie on the image side marked L ----
        side = c.js("(async () => { const v = window.__viewer; await v.setOverlay('nv_lobes'); const k = Math.floor(v.state().dims[2] / 2), d = v.sliceLabels('axial', k), a = Uint8Array.from(atob(d.b64), x => x.charCodeAt(0));"
                    " let l = 0, r = 0, nl = 0, nr = 0; for (let p = 0; p < a.length; p++) { const col = p % d.w; if (a[p] === 28 || a[p] === 29) { l += col; nl++; } else if (a[p] >= 30) { r += col; nr++; } }"
                    " return {left_mean_col: l / Math.max(1, nl), right_mean_col: r / Math.max(1, nr), w: d.w, edges: v.edges('axial'), nl, nr}; })()")
        ok("left-lung classes are drawn on the side of the image marked L, right-lung classes on the side marked R",
           side["edges"][:2] == ["R", "L"] and side["nl"] > 0 and side["nr"] > 0 and side["left_mean_col"] > side["w"] / 2 > side["right_mean_col"], side)
        (out / "nv_lobes.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
        c.js("(async () => { await window.__viewer.setOverlay('ts_lobes'); await new Promise(r => setTimeout(r, 200)); })()")
        (out / "ts_lobes.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
        c.js("(async () => { await window.__viewer.setOverlay('nv_all'); await new Promise(r => setTimeout(r, 200)); })()")
        (out / "nv_all.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
        # ---- real clicks, decode count, read-out ----
        cl = c.js("(async () => { const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 60)), q = k => document.querySelector('#segbar .ov[data-k=\"' + k + '\"]'), o = {}; await v.setOverlay(null); const d0 = v.segDecodes();"
                  " q('nv_lobes').click(); await w(); await v.idle(); o.a = [v.state().active, v.segDecodes() - d0, v.state().overlayBytes, !q('nv_lobes').classList.contains('off')];"
                  " q('nv_all').click(); await w(); await v.idle(); o.b = [v.state().active, v.segDecodes() - d0, v.state().overlayBytes, q('nv_lobes').classList.contains('off')];"
                  " q('ts_lobes').click(); await w(); await v.idle(); o.c = [v.state().active, v.segDecodes() - d0, v.state().overlayBytes];"
                  " await new Promise(r => requestAnimationFrame(() => setTimeout(r, 60))); o.loc = document.getElementById('loc').textContent;"
                  " q('ts_lobes').click(); await w(); await v.idle(); o.d = [v.state().active, v.state().overlayBytes]; o.errors = window.__viewerErrors; return o; })()")
        nvb, tsb = meta["vols"]["nvseg"]["raw_bytes"], meta["vols"]["tsseg"]["raw_bytes"]
        ok("clicking the buttons: one outline at a time; two buttons of the same model share one decoded label map; the other model replaces it; a second click switches it off and releases the memory",
           cl["a"] == ["nv_lobes", 1, nvb, True] and cl["b"] == ["nv_all", 1, nvb, True] and cl["c"] == ["ts_lobes", 2, tsb] and cl["d"] == [None, 0] and not cl["errors"]
           and "outline: TotalSegmentator - Lung lobes  ·  at the cross-hair: " in cl["loc"], cl["loc"][-120:])
        # ---- speed and memory with the largest label map on ----
        sp = c.js("(async () => { const v = window.__viewer, o = {}, k0 = v.state().cur[2], n = 60; await v.setOverlay(null); let t = performance.now(); for (let i = 0; i < n; i++) v.sliceRGBA('axial', (k0 + i) % v.state().dims[2]); o.off = (performance.now() - t) / n;"
                  " t = performance.now(); await v.setOverlay('nv_all'); o.decode_ms = performance.now() - t; t = performance.now(); for (let i = 0; i < n; i++) v.sliceRGBA('axial', (k0 + i) % v.state().dims[2]); o.on = (performance.now() - t) / n; return o; })()", 300)
        R["timings"] = {"decode_largest_label_map_ms": round(sp["decode_ms"]), "fill_one_axial_image_ms_no_outline": round(sp["off"], 2), "fill_one_axial_image_ms_all_structures": round(sp["on"], 2)}
        R["memory"]["all_structures_on"] = b.gc_mem()
        keys = json.dumps(expect_chips)
        c.js(f"(async () => {{ const v = window.__viewer, ks = {keys}; for (let r = 0; r < 4; r++) for (const k of ks) {{ await v.setOverlay(k); v.step(1); await new Promise(x => setTimeout(x, 15)); }} await v.setOverlay(null); }})()", 600)
        R["memory"]["after_4_rounds_through_all_buttons_outline_off"] = b.gc_mem()
        grow = R["memory"]["after_4_rounds_through_all_buttons_outline_off"]["private_mb"] - R["memory"]["after_startup"]["private_mb"]
        held = R["memory"]["all_structures_on"]["private_mb"] - R["memory"]["after_startup"]["private_mb"]
        ok("memory stays bounded: after four rounds through every button and switching off, the page holds no more than 60 MB above start-up; drawing one image with all structures on takes under 50 ms",
           grow <= 60 and sp["on"] < 50, f"largest label map {nvb / 1e6:.0f} MB decoded in {sp['decode_ms']:.0f} ms; +{held} MB while on; +{grow} MB after switching off; {sp['on']:.1f} ms per image with it, {sp['off']:.1f} ms without")
        # ---- slab views and the checklist's own outlines are as in v6 ----
        sb = c.js("(async () => { const v = window.__viewer, w = () => new Promise(r => setTimeout(r, 120)), o = {}; await v.setOverlay('nv_all'); document.getElementById('sl25').click(); await w(); await v.idle();"
                  " o.mean = [v.state().active, v.shownGray('axial').coloured_pixels, getComputedStyle(document.getElementById('segbar')).display, v.state().overlayBytes];"
                  " const tr = document.querySelector('tr.jump'); o.hasRow = !!tr; if (tr) { tr.click(); await w(); await v.idle(); await w(); o.jump = [v.slab().mode, v.state().active, tr.dataset.jump, getComputedStyle(document.getElementById('segbar')).display]; }"
                  " document.getElementById('slmip').click(); await w(); await v.idle(); o.mip = [v.state().active, v.shownGray('axial').coloured_pixels, v.slab().mode]; document.getElementById('sl25').click(); await w(); o.errors = window.__viewerErrors; return o; })()")
        ok("Axial 2.5 mm switches a segmentation outline off and hides the bar; 'Show in scan' of a checklist row still goes to native slices and shows the tool's own outline; MIP 10 mm removes it again",
           sb["mean"] == [None, 0, "none", 0] and (not sb["hasRow"] or (sb["jump"][0] == "native" and sb["jump"][1] == sb["jump"][2] and sb["jump"][3] != "none")) and sb["mip"] == [None, 0, "mip"] and not sb["errors"], sb)
    finally:
        b.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page", type=Path)
    ap.add_argument("--second-seg", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--browser", choices=list(BROWSERS), default="edge")
    ap.add_argument("--gpu-mode", choices=list(GPU_FLAGS), default="igpu")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--port", type=int, default=9661)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not args.synthetic and not (args.page and args.second_seg):
        ap.error("use --synthetic, or --page with --second-seg")
    name = "synthetic" if args.synthetic else args.page.stem
    out = ROOT / "runs" / f"V12_seg_test_{args.label + '_' if args.label else ''}{name}_{args.browser}_{'sw' if args.synthetic else args.gpu_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    R = {"kind": "technical checks only; not clinician usability feedback and not a check of anatomical correctness", "browser": args.browser,
         "gpu_mode": "sw" if args.synthetic else args.gpu_mode, "viewer_build": v7.VIEWER_BUILD, "checks": [], "memory": {}, "timings": {}}

    def ok(name, cond, detail=None):
        R["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
        print(("PASS " if cond else "FAIL ") + name + (f"  [{str(detail)[:300]}]" if detail is not None else ""), flush=True)
    try:
        (run_synthetic if args.synthetic else run_page)(args, out, R, ok)
    except Exception as e:  # reported, never dropped
        import traceback
        R["error"], R["traceback"] = f"{type(e).__name__}: {e}", traceback.format_exc()
        print("ERROR", R["error"], flush=True)
        print(R["traceback"], flush=True)
    R["passed"] = "error" not in R and bool(R["checks"]) and all(x["pass"] for x in R["checks"])
    (out / "result.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    md = [f"# Viewer v7 (two segmentation models) technical checks: {name} in {R.get('browser_version', args.browser)} - {'PASSED' if R['passed'] else 'FAILED'}", "",
          "Technical checks only. Not clinician usability feedback, and not a check that either segmentation is anatomically right.", "", "| check | result | detail |", "|---|---|---|"]
    md += [f"| {x['name']} | {'pass' if x['pass'] else 'FAIL'} | {str(x['detail'] or '')[:300].replace('|', '/')} |" for x in R["checks"]]
    md += ["", "Timings: " + json.dumps(R["timings"]), "", "Memory (MB): " + json.dumps(R["memory"])]
    if "error" in R:
        md += ["", "ERROR: " + R["error"]]
    (out / "result.md").write_text("\n".join(md), encoding="utf-8")
    print(f"-> {out}   {'PASSED' if R['passed'] else 'FAILED'}  ({sum(x['pass'] for x in R['checks'])} of {len(R['checks'])} checks)")
    return 0 if R["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
