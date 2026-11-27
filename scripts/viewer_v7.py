"""Checklist viewer v7: viewer v6 plus a bar that shows the outlines of TWO segmentation models, for comparison.

v6 stays as it is (scripts/viewer_v6.py). This module takes the v6 page script and adds label maps by exact, asserted
text insertions. Slab views, geometry, decoding, grading and CSV are v6's.

What is new
  * A second bar under the outline bar, "Segmentation models", with two groups of buttons:
      TotalSegmentator   the model the tool's measurements already use (lung lobes, trachea, airways, heart, aorta)
      NV-Segment-CT      a second model, run for comparison only (scripts/run_nvsegment.py)
    Both are predictions. No answer, measurement or grade on the page uses the second model.
  * A button shows a chosen SET of structures from one label map (one byte per voxel = class number), each structure in
    its own colour. The same structure has the same colour in both models, so switching between the two buttons shows
    where they differ. The name of the structure under the cross-hair is written under the scan.
  * Still one outline at a time, decoded when asked for and released when another is chosen. Buttons that read the same
    label map share its decoded copy, so switching between them costs nothing.
  * As in v6, outlines are drawn on native slices only; the bar is hidden in the two slab views.

Label maps (built by seg_bundle below, all on the page's canonical grid, cropped to their bounding box, lossless):
  tsseg   TotalSegmentator masks written with NV-Segment-CT's class numbers (28-32 lobes, 57 trachea, 115 heart, 6 aorta)
  tsaw    TotalSegmentator lung_airways as class 132 (its own map: airways lie inside the lobes)
  nvseg   NV-Segment-CT "segment everything" label map, as saved
  nvsegx  NV-Segment-CT extra classes (class 'lung tumor', airway, ...) merged by a stated priority; overlaps are counted
A structure with no voxels gets no button; its name is listed in the bar as "nothing outlined".
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import zlib

import numpy as np

import viewer_v3 as v3
import viewer_v6 as v6
from viewer_v6 import (AXIAL_MM, CLIP_NOTE, MAX_OBLIQUE_DEG, MIP_MM, ObliqueOrientationError, canonical_transform, ct_payload,  # noqa: F401
                       obliquity_deg, overlay_payload, reported_thickness_mm, slab_max, slab_mean, slab_plan, to_canonical)

VIEWER_BUILD = "checklist_viewer_v7_canvas2d_axial25_mip10_two_segmentations"
GROUP_TS = "TotalSegmentator (used by the tool)"
GROUP_NV = "NV-Segment-CT (second model, comparison only)"
TS_GROUPS = [  # key, label, source map, class numbers
    ("ts_lobes", "Lung lobes", "tsseg", [28, 29, 30, 31, 32]), ("ts_trachea", "Trachea", "tsseg", [57]),
    ("ts_airways", "Airways", "tsaw", [132]), ("ts_heart", "Heart + aorta", "tsseg", [115, 6])]
FIXED_COLOURS = {28: "#facc15", 29: "#22c55e", 30: "#3b82f6", 31: "#f97316", 32: "#ec4899",      # five lobes
                 57: "#22d3ee", 132: "#00e5ff", 115: "#ef4444", 6: "#fb7185", 23: "#ff2020", 1: "#a16207"}


def label_colour(label: int) -> str:
    """One fixed colour per class number: listed ones by hand, the rest spread round the colour wheel."""
    if label in FIXED_COLOURS:
        return FIXED_COLOURS[label]
    r, g, b = colorsys.hsv_to_rgb((label * 137.508 % 360) / 360.0, 0.70, 1.0)
    return "#%02x%02x%02x" % (int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5))


def label_payload(key: str, lab_can: np.ndarray) -> tuple[str, dict]:
    """Canonical uint8 label map -> the bounding box of its non-zero part (same grid, integer offset): exact."""
    lab = np.asarray(lab_can)
    assert lab.dtype == np.uint8, "label maps are one byte per voxel"
    idx = np.argwhere(lab > 0)
    lo, hi = idx.min(0), idx.max(0) + 1
    sub = np.ascontiguousarray(lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    back = np.zeros(lab.shape, np.uint8)
    back[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = sub
    assert np.array_equal(back, lab), "bounding-box crop of the label map is not exact"
    raw = sub.tobytes(order="F")
    gz = zlib.compress(raw, 9, wbits=31)
    html, n_parts = v3._blocks(key, gz)
    present = np.flatnonzero(np.bincount(sub.ravel(), minlength=256))
    return html, {"codec": "bbox-labels-gzip-v1", "offset": [int(x) for x in lo], "dims": [int(x) for x in sub.shape], "raw_bytes": len(raw),
                  "gz_bytes": len(gz), "parts": n_parts, "voxels_on": int((sub > 0).sum()), "labels_present": [int(x) for x in present if x],
                  "sha256": hashlib.sha256(raw).hexdigest()}


def merge_extra(extra: dict[int, np.ndarray], order: list[int]) -> tuple[np.ndarray, int]:
    """Extra-class masks -> one label map; where two cover a voxel the earlier class in `order` is kept. Returns (map, overlapping voxels)."""
    assert sorted(order) == sorted(extra), "priority list and masks differ"
    first = next(iter(extra.values()))
    out, hits = np.zeros(first.shape, np.uint8), np.zeros(first.shape, np.uint8)
    for x in reversed(order):
        m = np.asarray(extra[x]) > 0
        out[m] = x
        hits += m
    return out, int((hits > 1).sum())


def ts_label_map(masks: dict[int, np.ndarray]) -> tuple[np.ndarray, int]:
    """TotalSegmentator masks {class number: mask} -> one label map. They should not overlap; overlapping voxels are counted (lower number kept)."""
    first = next(iter(masks.values()))
    out, hits = np.zeros(first.shape, np.uint8), np.zeros(first.shape, np.uint8)
    for x in sorted(masks, reverse=True):
        m = np.asarray(masks[x]) > 0
        out[m] = x
        hits += m
    return out, int((hits > 1).sum())


def seg_bundle(maps: dict[str, np.ndarray], nv_groups: list[dict], names: dict, info: dict | None = None) -> tuple[str, dict, list[dict], dict]:
    """Label maps on the canonical grid -> (payload html, vols meta, button list, META.seg).

    maps: any of tsseg / tsaw / nvseg / nvsegx (uint8; an all-zero map is treated as absent).
    nv_groups: 'viewer_groups' of configs/nvsegment_ct_v1.json. names: {class number as text: name}."""
    html, vols, overlays, empty = "", {}, [], []
    present = {}
    for src, lab in maps.items():
        if lab is not None and np.asarray(lab).any():
            h, vols[src] = label_payload(src, np.asarray(lab, np.uint8))
            html += h
            present[src] = set(vols[src]["labels_present"])
    plan = [(k, lbl, src, labs, GROUP_TS) for k, lbl, src, labs in TS_GROUPS]
    for g in nv_groups:
        src = "nvseg" if g["source"] == "everything" else "nvsegx"
        labs = sorted(present.get(src, [])) if g["labels"] == "all" else g["labels"]
        plan.append((g["key"], g["label"], src, labs, GROUP_NV))
    used = set()
    for key, label, src, labs, group in plan:
        if src not in maps:
            continue                                    # that model was not given to this page at all
        have = [x for x in labs if x in present.get(src, ())]
        if not have:
            empty.append(f"{group.split(' (')[0]}: {label}")
            continue
        used |= set(have)
        cols = [label_colour(x) for x in have]
        swatch = cols[0] if len(cols) == 1 else "linear-gradient(90deg," + ",".join(cols[:5]) + ")"
        overlays.append({"key": key, "label": label, "swatch": swatch, "src": src, "labels": have, "group": group,
                         "title": f"{len(have)} structure{'s' if len(have) != 1 else ''} predicted by {group.split(' (')[0]}: "
                                  + ", ".join(names[str(x)] for x in have[:12]) + (" ..." if len(have) > 12 else "")})
    seg = {"names": {str(x): names[str(x)] for x in sorted(used)}, "colours": {str(x): label_colour(x) for x in sorted(used)},
           "empty": empty, "info": info or {}}
    return html, vols, overlays, seg


CSS = v6.CSS + """
/* viewer v7: second bar with the outlines of the two segmentation models; native slices only, as every outline */
#segbar{row-gap:4px}
#segbar .segname{color:#9fc2ff;margin-left:8px}
#segbar .segnone{color:#9aa4b2;font-size:13px;margin-left:8px}
body.slabmode #segbar{display:none}
"""

_SEG_BAR = (' <div class="bar__AO__" id="segbar"><b title="Outlines predicted by two segmentation models. They are not checked by a person. '
            'The second model is shown for comparison only: no answer or measurement on this page uses it.">Segmentation models (predictions; one at a time)</b></div>\n')
VIEWER_HTML = v6._once(v6.VIEWER_HTML, ' <div id="vbanner"></div>', _SEG_BAR.rstrip("\n") + '\n <div id="vbanner"></div>')
EARLY_JS = v6.EARLY_JS

_JS_STATE = r"""
 // ---------- label maps (viewer v7): one byte per voxel = class number; a button shows a set of classes, each in its own colour ----------
 const SEG = META.seg || {names: {}, colours: {}, empty: []};
 const LON = new Uint8Array(256), LR = new Uint8Array(256), LG = new Uint8Array(256), LB = new Uint8Array(256), IA = 1 - ALPHA;
 for (const k in SEG.colours) { const c = SEG.colours[k]; LR[k] = parseInt(c.slice(1, 3), 16); LG[k] = parseInt(c.slice(3, 5), 16); LB[k] = parseInt(c.slice(5, 7), 16); }
 let SEGDEC = 0;                                     // label maps decoded so far (buttons on the same map share one decoded copy)
 function setLabels(list) { LON.fill(0); for (const x of list) LON[x] = 1; }
 function labelAt(v) {                               // class number shown at a voxel by the active label-map button, 0 = none
   if (!ov || !ov.lab) return 0;
   const d = ov.d, a = v[0] - ov.o[0], b = v[1] - ov.o[1], e = v[2] - ov.o[2];
   if (a < 0 || b < 0 || e < 0 || a >= d[0] || b >= d[1] || e >= d[2]) return 0;
   const L = ov.data[a + d[0] * b + d[0] * d[1] * e]; return LON[L] ? L : 0;
 }
"""

_JS_HOOKS = r"""
   segBuild: true, labelAt: v => labelAt(v), segNames: () => Object.assign({}, SEG.names), segShown: () => { const o = []; for (let i = 1; i < 256; i++) if (ov && ov.lab && LON[i]) o.push(i); return o; },
   segSource: () => ov ? (ov.src || null) : null, segDecodes: () => SEGDEC,
   sliceRGBA: (view, s) => { const r = grayOf(view, s, true); return {w: r.g.w, h: r.g.h, b64: b64(new Uint8Array(r.u.buffer))}; },
   sliceLabels: (view, s) => { const g = G[view], o = new Uint8Array(g.w * g.h); for (let r = 0, p = 0; r < g.h; r++) for (let c = 0; c < g.w; c++, p++) o[p] = labelAt(g.vox(s, c, r)); return {w: g.w, h: g.h, b64: b64(o)}; },
"""


def _build_js() -> str:
    js, _once = v6.JS, v6._once
    js = _once(js, " let ov = null;                                      // the ONE decoded outline: {key, data, o:[i,j,k], d:[mx,my,mz]}\n",
               " let ov = null;                                      // the ONE decoded outline: {key, data, o:[i,j,k], d:[mx,my,mz]}\n" + _JS_STATE)
    # the one place an outline is painted: a label map paints each shown class in its own colour
    js = _once(js, " && e < d[2] && m[a + mx * b + mxy * e]) u32[p] = blendPx[u32[p] & 255];\n",
               " && e < d[2]) { const L = m[a + mx * b + mxy * e];\n"
               "         if (L) { if (!ov.lab) u32[p] = blendPx[u32[p] & 255];\n"
               "           else if (LON[L]) { const q = u32[p] & 255; u32[p] = (0xFF000000 | ((q * IA + LB[L] * ALPHA + 0.5 | 0) << 16) | ((q * IA + LG[L] * ALPHA + 0.5 | 0) << 8) | (q * IA + LR[L] * ALPHA + 0.5 | 0)) >>> 0; } } }\n")
    # buttons that read the same label map keep its decoded copy; any other change releases it, as before
    js = _once(js, "     if (ov) { ov.data = null; ov = null; V.active = null; paintChips(); draw(); }      // release the previous decoded outline\n",
               "     let keep = null;\n"
               "     if (ov) { const nxt = key ? META.overlays.find(x => x.key === key) : null;\n"
               "       if (nxt && ov.src && nxt.src === ov.src) keep = ov.data; else ov.data = null;      // release the previous decoded outline unless the next button reads the same label map\n"
               "       ov = null; V.active = null; paintChips(); draw(); }\n")
    js = _once(js, "       const m = META.vols[key], data = await decodeOverlay(key);\n       setBlend(hexRgb(o.swatch));\n"
                   "       ov = {key: key, label: o.label, data: data, o: m.offset, d: m.dims}; V.active = key;\n",
               "       const src = o.src || key, m = META.vols[src], data = keep || await decodeOverlay(src);\n"
               "       if (o.src && !keep) SEGDEC++;\n"
               "       if (o.labels) setLabels(o.labels); else setBlend(hexRgb(o.swatch));\n"
               "       ov = {key: key, src: o.src || null, lab: !!o.labels, label: (o.group ? o.group.split(' (')[0] + ' - ' : '') + o.label, data: data, o: m.offset, d: m.dims}; V.active = key;\n")
    js = _once(js, "     $('#ovbar').appendChild(el);\n",
               "     if (o.src) { const bar = $('#segbar'); el.title = o.title || '';\n"
               "       if (bar.dataset.g !== o.group) { bar.dataset.g = o.group; const h = document.createElement('span'); h.className = 'segname'; h.textContent = o.group + ':'; bar.appendChild(h); }\n"
               "       bar.appendChild(el); }\n"
               "     else $('#ovbar').appendChild(el);\n")
    js = _once(js, "(ov ? '  ·  outline: ' + ov.label : '');\n",
               "(ov ? '  ·  outline: ' + ov.label + (ov.lab ? '  ·  at the cross-hair: ' + (SEG.names[labelAt(S.cur)] || 'none of the shown structures') : '') : '');\n")
    js = _once(js, " Object.assign(V, {\n", " Object.assign(V, {\n" + _JS_HOOKS.lstrip("\n"))
    js = _once(js, "   $('#z11').onclick = () => oneToOne();\n",
               "   $('#z11').onclick = () => oneToOne();\n"
               "   { const bar = $('#segbar'); if (!META.overlays.some(o => o.src)) bar.style.display = 'none';\n"
               "     else if (SEG.empty && SEG.empty.length) { const n = document.createElement('span'); n.className = 'segnone'; n.id = 'segnone'; n.textContent = 'Nothing outlined: ' + SEG.empty.join('; '); bar.appendChild(n); } }\n")
    return js


JS = _build_js()


def viewer_html(agentonly: bool) -> str:
    return VIEWER_HTML.replace("__AO__", " agentonly" if agentonly else "")


def viewer_script(meta: dict) -> str:
    return "<script>\n" + JS.replace("__META__", json.dumps(meta)) + "\n</script>"


def standalone_page(title: str, arr: np.ndarray, affine: np.ndarray, overlays: dict | None = None, targets: dict | None = None,
                    slab: dict | None = None, label_maps: dict | None = None, nv_groups: list | None = None, names: dict | None = None) -> str:
    """A minimal page with ONLY the v7 viewer: used for the tests on synthetic volumes.
    label_maps: {tsseg / tsaw / nvseg / nvsegx: uint8 array in arr's orientation}."""
    page = v6.standalone_page(title, arr, affine, overlays, targets, slab)
    ornt, _, _, _ = canonical_transform(affine, arr.shape)
    html, vols, ovs, seg = seg_bundle({k: to_canonical(np.asarray(v, np.uint8), ornt) for k, v in (label_maps or {}).items()}, nv_groups or [], names or {})
    a = page.index('<div class="viewer">')
    b = page.index("</div>\n<script>\n", a)
    page = page[:a] + viewer_html(False).rstrip("\n") + page[b:]
    page = v6._once(page, v6.CSS + "</style>", CSS + "</style>")
    a = page.rindex("<script>")
    m0 = page.index("const META = ", a) + len("const META = ")
    meta = json.loads(page[m0:page.index("\n", m0)].rstrip().rstrip(";"))
    meta["build"] = VIEWER_BUILD
    meta["vols"].update(vols)
    meta["overlays"] += ovs
    meta["seg"] = seg
    return page[:a] + html + viewer_script(meta) + "\n</body></html>"
