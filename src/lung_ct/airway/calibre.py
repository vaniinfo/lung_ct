"""Exploratory per-branch lumen calibre and tapering from a predicted airway mask ("lung_ct airway calibre v1").

Our own Python implementation. It is NOT AirQuant, does not run AirQuant and makes no claim of equivalence
(AirQuant needs MATLAB, which this project does not use). It is a MEASUREMENT stage only: it detects nothing and
diagnoses nothing. Every value depends on what the segmentation model outlined.

NOT VALIDATED. The values below are not validated airway diameters. Before any claim they need the validation in
docs/AIRWAY_BRONCHIECTASIS_MEASURES.md (synthetic tubes, then a dataset with airway reference annotations).

Input
-----
  * the binary airway mask (same file the tree stage reads) and the TreeResult of lung_ct.airway.tree for that mask
  * optional: a lobe label image (integers, any grid; sampled through world RAS mm coordinates)

Geometry and units
------------------
  * the same isotropic analysis grid as the tree stage (voxel edge `iso_mm`, taken from the NIfTI affine)
  * every reported length is in mm, every area in mm2, every volume in mm3. Voxel counts are never reported as sizes.

Definitions
-----------
Ownership
  Every lumen voxel of the isotropic mask is assigned to the nearest centreline voxel of a tree branch (Euclidean
  distance on the isotropic grid). For a straight tube the voxels owned by a piece of centreline form a slab
  perpendicular to it, so   volume owned / centreline length = cross-sectional area.
Arc length
  Position along a branch = cumulative length of its centreline after the same Gaussian smoothing the tree stage
  uses for `length_mm` (ends pinned), proximal -> distal.
Trimming
  The parts of a branch next to its two ends are excluded, because voxels near a bifurcation belong to more than one
  airway and the end of a terminal branch is where the segmentation stopped. Trim length at each end =
  max(trim_min_mm, trim_radius_factor * inscribed radius at that end).
Per-branch values (trimmed part only, unless stated)
  lumen_volume_mm3                 all voxels owned by the branch (untrimmed) x voxel volume
  usable_length_mm                 centreline length left after trimming
  mean_area_mm2                    owned volume in the trimmed part / usable_length_mm
  area_equivalent_diameter_mm      2 * sqrt(mean_area_mm2 / pi): diameter of a circle with that area
  inscribed_diameter_median_mm     2 * median distance-transform value on the trimmed centreline. QC cross-check only:
                                   it is the largest sphere that fits, NOT an airway diameter, and must not be reported as one
  calibre_ratio                    area_equivalent_diameter_mm / inscribed_diameter_median_mm. About 1 for a round tube;
                                   clearly above 1 for a flattened lumen, two touching airways or a leak
  intratapering_pct_per_mm         -100 * slope of ln(area-equivalent diameter) against arc length, least squares over
                                   consecutive windows of `window_mm`. Positive = the lumen narrows distally. None when
                                   fewer than `min_windows` windows fit
  intertapering_pct                100 * (parent diameter - this diameter) / parent diameter, parent = the tree branch
                                   ending where this one starts. Positive = narrower than its parent
Lobe
  Majority lobe label under the branch's centreline points; "central" when fewer than half of the points lie inside
  any lobe (trachea and main bronchi).

Quality flags (reporting conventions fixed before any result was seen; they are NOT validated thresholds)
  per branch: terminal_branch, short_usable_length, small_relative_to_voxel, non_circular_or_merged_lumen,
              no_parent_diameter, not_measured
  per case:   see case_quality()

Not computed: wall thickness, broncho-arterial ratio (no reliable airway-artery pairing in this project), any diagnosis.
"""
from __future__ import annotations

import math

import nibabel as nib
import numpy as np
from scipy import ndimage

from .tree import TreeResult, preprocess

METHOD = "lung_ct airway calibre v1 (Python; mask-derived estimates: ownership volume / centreline length); NOT AirQuant; exploratory, unvalidated"

DEFAULTS = {
    "trim_min_mm": 2.0,                 # shortest trim at each branch end
    "trim_radius_factor": 1.0,          # trim >= this x inscribed radius at the end
    "window_mm": 3.0,                   # window length of the diameter profile used for intra-branch tapering
    "min_windows": 3,                   # fewer windows: no intratapering
    "min_usable_mm": 3.0,               # shorter usable length: flagged short_usable_length
    "small_voxel_factor": 3.0,          # diameter below this x coarsest input spacing: flagged small_relative_to_voxel
    "non_circular_ratio": 1.5,          # calibre_ratio above this: flagged non_circular_or_merged_lumen
    "discarded_fraction_flag": 0.05,    # discarded / kept mask volume above this: case flag broken_airway_mask
    "anisotropy_flag": 2.0,             # coarsest / finest spacing above this: case flag anisotropic_voxels
    "thick_slice_mm": 1.5,              # coarsest spacing above this: case flag thick_slices
}
MEASURED_KINDS = ("root", "internal", "terminal")


def _smooth(pts: np.ndarray, sigma_samples: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    pts = np.asarray(pts, float)
    if len(pts) > 3 and sigma_samples > 0:
        sm = gaussian_filter1d(pts, sigma=sigma_samples, axis=0, mode="nearest")
        sm[0], sm[-1] = pts[0], pts[-1]
        return sm
    return pts


def _point_lengths(pts_mm: np.ndarray) -> np.ndarray:
    """Centreline length attributed to each point (half of each neighbouring step), mm. Sums to the path length."""
    n = len(pts_mm)
    if n < 2:
        return np.zeros(n)
    d = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    out = np.zeros(n)
    out[:-1] += d / 2
    out[1:] += d / 2
    return out


def measure_branches(mask_img: nib.Nifti1Image, res: TreeResult, params: dict | None = None,
                     lobe_img: nib.Nifti1Image | None = None, lobe_names: dict | None = None) -> list[dict]:
    """One row per tree branch (kind root / internal / terminal). See the module docstring for every column."""
    p = {**DEFAULTS, **(params or {})}
    mask, aff, info = preprocess(mask_img, res.summary["params"]["iso_mm"])
    iso = float(info["iso_mm"])
    if mask.shape != res.skeleton.shape or not np.allclose(aff, res.affine_iso, atol=1e-6):
        raise ValueError("the mask does not reproduce the analysis grid of the tree result (different mask or parameters)")
    vox_mm3 = iso ** 3
    coarsest = float(max(info["input_spacing_mm"]))
    sigma_samples = float(res.summary["params"]["path_smoothing_sigma_mm"]) / iso
    inv = np.linalg.inv(res.affine_iso)
    shape = np.array(mask.shape)

    rows = [r for r in res.branches if r["in_tree"] and r["kind"] in MEASURED_KINDS and r["branch_id"] in res.branch_paths]
    order = {r["branch_id"]: k for k, r in enumerate(rows)}
    owner = np.full(mask.shape, -1, np.int32)          # index into `rows` of the branch whose centreline passes here
    pos = np.zeros(mask.shape, np.int32)               # index of the centreline point along that branch
    vox_paths = {}
    for r in rows:
        w = np.asarray(res.branch_paths[r["branch_id"]], float)
        v = np.round((inv @ np.c_[w, np.ones(len(w))].T)[:3].T).astype(int)
        v = np.clip(v, 0, shape - 1)
        vox_paths[r["branch_id"]] = v
        owner[v[:, 0], v[:, 1], v[:, 2]] = order[r["branch_id"]]
        pos[v[:, 0], v[:, 1], v[:, 2]] = np.arange(len(v))
    if not rows:
        return []

    edt = ndimage.distance_transform_edt(mask) * iso                         # mm; inscribed radius, QC only
    _, near = ndimage.distance_transform_edt(owner < 0, return_indices=True)  # nearest centreline voxel of every voxel
    mi = np.nonzero(mask)
    ni = (near[0][mi], near[1][mi], near[2][mi])
    own_b, own_p = owner[ni], pos[ni]
    del near

    lobe = None
    if lobe_img is not None:
        lobe = (np.asanyarray(lobe_img.dataobj), np.linalg.inv(lobe_img.affine))
    by_distal = {r["distal_node"]: r["branch_id"] for r in rows}

    out = {}
    for r in rows:
        bid, k = r["branch_id"], order[r["branch_id"]]
        v = vox_paths[bid]
        n = len(v)
        counts = np.bincount(own_p[own_b == k], minlength=n)[:n].astype(float)       # lumen voxels owned by each point
        pts_mm = _smooth(np.asarray(res.branch_paths[bid], float), sigma_samples)
        pl = _point_lengths(pts_mm)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts_mm, axis=0), axis=1))]) if n > 1 else np.zeros(n)
        total = float(s[-1]) if n else 0.0
        r_in = edt[v[:, 0], v[:, 1], v[:, 2]]
        t0 = max(p["trim_min_mm"], p["trim_radius_factor"] * float(r_in[0])) if n else 0.0
        t1 = max(p["trim_min_mm"], p["trim_radius_factor"] * float(r_in[-1])) if n else 0.0
        use = (s >= t0) & (s <= total - t1)
        usable = float(pl[use].sum())
        flags = []
        m = {"branch_id": bid, "generation": r["generation"], "kind": r["kind"], "side": r["side"], "length_mm": r["length_mm"],
             "lumen_volume_mm3": round(float(counts.sum()) * vox_mm3, 2), "trim_proximal_mm": round(t0, 2), "trim_distal_mm": round(t1, 2),
             "usable_length_mm": round(usable, 2), "mean_area_mm2": None, "area_equivalent_diameter_mm": None,
             "inscribed_diameter_median_mm": None, "calibre_ratio": None, "intratapering_pct_per_mm": None, "n_taper_windows": 0,
             "intertapering_pct": None, "parent_branch_id": by_distal.get(r["proximal_node"]), "lobe": None, "lobe_fraction": None}
        if r["kind"] == "terminal":
            flags.append("terminal_branch")
        if usable <= 0 or counts[use].sum() <= 0:
            flags.append("not_measured")
        else:
            area = float(counts[use].sum()) * vox_mm3 / usable
            d_eq = 2.0 * math.sqrt(area / math.pi)
            d_in = 2.0 * float(np.median(r_in[use]))
            m.update(mean_area_mm2=round(area, 3), area_equivalent_diameter_mm=round(d_eq, 3), inscribed_diameter_median_mm=round(d_in, 3),
                     calibre_ratio=round(d_eq / d_in, 3) if d_in > 0 else None)
            if usable < p["min_usable_mm"]:
                flags.append("short_usable_length")
            if d_eq < p["small_voxel_factor"] * coarsest:
                flags.append("small_relative_to_voxel")
            if d_in > 0 and d_eq / d_in > p["non_circular_ratio"]:
                flags.append("non_circular_or_merged_lumen")
            # diameter profile in consecutive windows of window_mm along the trimmed part
            lo, hi = t0, total - t1
            nwin = int((hi - lo) // p["window_mm"])
            xs, ys = [], []
            for j in range(nwin):
                a, b = lo + j * p["window_mm"], lo + (j + 1) * p["window_mm"]
                w = (s >= a) & (s < b)
                if pl[w].sum() > 0 and counts[w].sum() > 0:
                    xs.append((a + b) / 2)
                    ys.append(math.log(2.0 * math.sqrt(float(counts[w].sum()) * vox_mm3 / float(pl[w].sum()) / math.pi)))
            m["n_taper_windows"] = len(xs)
            if len(xs) >= p["min_windows"]:
                m["intratapering_pct_per_mm"] = round(-100.0 * float(np.polyfit(xs, ys, 1)[0]), 4)
        if lobe is not None and n:
            arr, linv = lobe
            w = np.asarray(res.branch_paths[bid], float)
            ijk = np.round((linv @ np.c_[w, np.ones(len(w))].T)[:3].T).astype(int)
            ok = np.all((ijk >= 0) & (ijk < np.array(arr.shape[:3])), axis=1)
            lab = np.zeros(len(w), int)
            lab[ok] = arr[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
            inside = lab[lab > 0]
            if len(inside) * 2 >= len(w):
                top = int(np.bincount(inside).argmax())
                m["lobe"] = (lobe_names or {}).get(top, str(top))
                m["lobe_fraction"] = round(float((lab == top).sum()) / len(w), 3)
            else:
                m["lobe"] = "central"
                m["lobe_fraction"] = round(1.0 - len(inside) / len(w), 3)
        m["flags"] = flags
        out[bid] = m

    for m in out.values():                                                   # relative change against the parent branch
        par = out.get(m["parent_branch_id"]) if m["parent_branch_id"] is not None else None
        dp = par["area_equivalent_diameter_mm"] if par else None
        if dp and m["area_equivalent_diameter_mm"]:
            m["intertapering_pct"] = round(100.0 * (dp - m["area_equivalent_diameter_mm"]) / dp, 3)
        elif m["generation"] not in (None, 0):
            m["flags"].append("no_parent_diameter")
    res_rows = list(out.values())
    for m in res_rows:
        m["flags"] = ";".join(m["flags"])
    return res_rows


def case_quality(mask_img: nib.Nifti1Image, res: TreeResult, params: dict | None = None,
                 lung_img: nib.Nifti1Image | None = None) -> dict:
    """Case-level quality flags. Each flag is a reason for caution, not a measurement and not a diagnosis."""
    p = {**DEFAULTS, **(params or {})}
    flags, detail = [], {}
    sp = np.sqrt((np.asarray(mask_img.affine)[:3, :3] ** 2).sum(axis=0))
    detail["spacing_mm_from_affine"] = [round(float(x), 4) for x in sp]
    if not np.all(np.isfinite(sp)) or np.any(sp <= 0):
        flags.append("missing_or_invalid_spacing")
    else:
        if np.allclose(sp, 1.0) and np.allclose(np.asarray(mask_img.affine)[:3, :3], np.eye(3)):
            flags.append("spacing_may_be_a_default (1 mm identity affine): check the source header")
        if float(sp.max() / sp.min()) > p["anisotropy_flag"]:
            flags.append(f"anisotropic_voxels (coarsest / finest spacing {float(sp.max() / sp.min()):.2f})")
        if float(sp.max()) > p["thick_slice_mm"]:
            flags.append(f"thick_slices ({float(sp.max()):.2f} mm): small airways are affected by partial volume")
    pre = res.summary["preprocess"]
    detail["mask_components"] = pre["components"]
    detail["discarded_volume_ml"], detail["kept_volume_ml"] = pre["discarded_volume_ml"], pre["kept_volume_ml"]
    if pre["kept_volume_ml"] > 0 and pre["discarded_volume_ml"] / pre["kept_volume_ml"] > p["discarded_fraction_flag"]:
        flags.append(f"broken_airway_mask ({pre['components']} pieces; {pre['discarded_volume_ml']} mL outside the largest piece were not measured)")

    def touching(img):
        can = nib.as_closest_canonical(img)
        a = np.asanyarray(can.dataobj) > 0
        return [name for name, sl in (("right", a[-1]), ("left", a[0]), ("anterior", a[:, -1]), ("posterior", a[:, 0]),
                                      ("superior", a[:, :, -1]), ("inferior", a[:, :, 0])) if sl.any()]
    t = touching(mask_img)
    detail["airway_mask_touches_image_edge"] = t
    if t:
        flags.append("airway_mask_touches_image_edge (" + ", ".join(t) + "): the airway tree may be cut off there")
    if lung_img is not None:
        tl = touching(lung_img)
        detail["lung_mask_touches_image_edge"] = tl
        if tl:
            flags.append("truncated_scan (lung outline touches the image edge: " + ", ".join(tl) + ")")
    for f in res.summary["flags"]:
        flags.append("incomplete_or_ambiguous_centreline: " + f)
    if res.summary["n_tree_branches"] < 3:
        flags.append("incomplete_or_ambiguous_centreline: fewer than 3 tree branches")
    return {"flags": flags, "detail": detail}


FEATURE_GENERATIONS = range(1, 9)
BLOCKING = ("not_measured", "short_usable_length", "non_circular_or_merged_lumen")


def case_features(rows: list[dict]) -> list[dict]:
    """Case-level summaries, long format: feature, group, value, n_branches. Branches with a blocking flag are left out;
    `small_relative_to_voxel` and `terminal_branch` branches are kept and counted separately."""
    def clean(rs):
        return [r for r in rs if not any(b in r["flags"].split(";") for b in BLOCKING)]

    def med(vals):
        vals = [v for v in vals if v is not None]
        return (round(float(np.median(vals)), 4), len(vals)) if vals else (None, 0)

    out = []
    groups = [(f"generation_{g}", [r for r in rows if r["generation"] == g]) for g in FEATURE_GENERATIONS]
    groups += [(f"lobe_{lb}", [r for r in rows if r["lobe"] == lb]) for lb in sorted({r["lobe"] for r in rows if r["lobe"]})]
    groups += [("all_measured", rows)]
    for name, rs in groups:
        c = clean(rs)
        for feat, key in (("median_area_equivalent_diameter_mm", "area_equivalent_diameter_mm"), ("median_mean_area_mm2", "mean_area_mm2"),
                          ("median_intratapering_pct_per_mm", "intratapering_pct_per_mm"), ("median_intertapering_pct", "intertapering_pct")):
            v, n = med([r[key] for r in c])
            out.append({"feature": feat, "group": name, "value": v, "n_branches": n})
        out.append({"feature": "n_branches_total", "group": name, "value": len(rs), "n_branches": len(rs)})
        out.append({"feature": "n_branches_small_relative_to_voxel", "group": name,
                    "value": sum("small_relative_to_voxel" in r["flags"].split(";") for r in rs), "n_branches": len(rs)})
    return out
