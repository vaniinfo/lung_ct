"""Checklist measurements (pipeline step 1, part 2): contrast, effusions, heart size, aortic calcium,
ground-glass and consolidation proxies outside the gravity-dependent band.

All arrays are reoriented to RAS (+x patient right, +y front, +z head) before any direction-dependent
step, so "posterior", "right" and "upper" never depend on how the file stores its axes.
Predicted masks only (TotalSegmentator); nothing here reads reference annotations.
Evidence row ids: E70-E96 (E83/E89/E91 = per-lobe distributions, v3_4) (E01-E42 lungs/airways, E50-E68 classifier).
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

RAS = nib.orientations.axcodes2ornt(("R", "A", "S"))


def _to_ras(a: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return nib.orientations.apply_orientation(a, nib.orientations.io_orientation(affine))


def _from_ras(a: np.ndarray, affine: np.ndarray) -> np.ndarray:
    back = nib.orientations.ornt_transform(RAS, nib.orientations.io_orientation(affine))
    return nib.orientations.apply_orientation(a, back)


def _ras_zooms(affine: np.ndarray) -> np.ndarray:
    z = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    o = nib.orientations.io_orientation(affine)
    out = np.zeros(3)
    for i in range(3):
        out[int(o[i, 0])] = z[i]
    return out


def _load(seg_dir: Path, rel: str, ref: nib.Nifti1Image) -> np.ndarray | None:
    p = seg_dir / f"{rel}.nii.gz"
    if not p.exists():
        return None
    img = nib.load(str(p))
    if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
        raise ValueError(f"{rel} is not on the CT grid")
    return np.asanyarray(img.dataobj) > 0


def dependent_band(lab: np.ndarray, frac: float) -> np.ndarray:
    """RAS label array (1 left, 2 right): True in the most posterior `frac` of each lung, per axial slice."""
    dep = np.zeros(lab.shape, bool)
    yy = np.arange(lab.shape[1])
    for z in range(lab.shape[2]):
        sl = lab[:, :, z]
        if not sl.any():
            continue
        for s in (1, 2):
            m = sl == s
            ys = np.where(m.any(axis=0))[0]
            if ys.size == 0:
                continue
            cut = ys.min() + frac * (ys.max() - ys.min())
            dep[:, :, z] |= m & (yy[None, :] <= cut)
    return dep


def location(mask: np.ndarray, lab: np.ndarray, lobe_idx: np.ndarray, lobe_names: list[str]) -> dict | None:
    """Side share, equal-volume zone shares and leading lobe of a RAS mask inside the lungs."""
    n = int(mask.sum())
    if n == 0:
        return None
    right = float((mask & (lab == 2)).sum()) / n
    per_z = (lab > 0).sum(axis=(0, 1)).cumsum()
    tot = per_z[-1]
    z1 = int(np.searchsorted(per_z, tot / 3))
    z2 = int(np.searchsorted(per_z, 2 * tot / 3))
    mz = mask.sum(axis=(0, 1))
    zones = {"lower": float(mz[:z1 + 1].sum()) / n, "middle": float(mz[z1 + 1:z2 + 1].sum()) / n,
             "upper": float(mz[z2 + 1:].sum()) / n}
    counts = np.bincount(lobe_idx[mask], minlength=len(lobe_names) + 1)[1:]
    side = "right" if right >= 0.75 else "left" if right <= 0.25 else "both"
    top_zone = max(zones, key=zones.get)
    zone = top_zone if zones[top_zone] >= 0.5 else "no single zone"
    return {"side": side, "right_share_pct": round(100 * right), "zone": zone,
            "zone_shares_pct": {k: round(100 * v) for k, v in zones.items()},
            "leading_lobe": lobe_names[int(counts.argmax())], "leading_lobe_share_pct": round(100 * counts.max() / n)}


def fissure_distance_mm(lobe_idx: np.ndarray, lobe_names: list[str], zooms) -> np.ndarray | None:
    """Distance (mm) of every voxel to the nearest boundary between two lobes of the SAME lung (a predicted fissure).

    lobe_idx: 0 outside, i+1 for lobe_names[i]; short names start with R or L. Returns None when no such boundary
    exists (e.g. a lobe mask is missing). Computed inside the lungs' bounding box; inf elsewhere."""
    side = np.zeros(len(lobe_names) + 1, np.uint8)
    for i, n in enumerate(lobe_names, start=1):
        side[i] = 1 if n.upper().startswith("L") else 2
    if not (lobe_idx > 0).any():
        return None
    idx = np.argwhere(lobe_idx > 0)
    lo, hi = idx.min(0), idx.max(0) + 1
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    li = lobe_idx[sl]
    edge = np.zeros(li.shape, bool)
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax], b[ax] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        d = (li[a] != li[b]) & (li[a] > 0) & (li[b] > 0) & (side[li[a]] == side[li[b]])
        edge[a] |= d
        edge[b] |= d
    if not edge.any():
        return None
    out = np.full(lobe_idx.shape, np.inf, np.float32)
    out[sl] = ndimage.distance_transform_edt(~edge, sampling=[float(z) for z in zooms]).astype(np.float32)
    return out


def lobe_distribution(mask: np.ndarray, lobe_idx: np.ndarray, lobe_names: list[str], vml: float,
                      fissure_mm: np.ndarray | None = None, near_mm: float | None = None) -> dict:
    """Full per-lobe distribution of a selected region (a boolean mask). Nothing is thresholded or dropped here.

    Three different denominators, each named in its key:
      pct_of_lobe    region voxels in the lobe / ALL voxels of that lobe mask
      pct_of_region  region voxels in the lobe / all voxels of the selected region
      near_fissure_pct_of_region  region voxels within near_mm of a predicted fissure / all voxels of the region
    A lobe appears in the result even when it holds no selected voxel."""
    n = int(mask.sum())
    nl = len(lobe_names) + 1
    c_reg = np.bincount(lobe_idx[mask], minlength=nl) if n else np.zeros(nl, int)
    c_all = np.bincount(lobe_idx.ravel(), minlength=nl)
    near = fissure_mm is not None and near_mm is not None
    c_near = (np.bincount(lobe_idx[mask & (fissure_mm <= near_mm)], minlength=nl) if n else np.zeros(nl, int)) if near else None
    lobes = {}
    for i, nm in enumerate(lobe_names, start=1):
        lobes[nm] = {"ml": round(float(c_reg[i]) * vml, 2), "voxels": int(c_reg[i]),
                     "pct_of_lobe": round(100 * float(c_reg[i]) / float(c_all[i]), 2) if c_all[i] else None,
                     "pct_of_region": round(100 * float(c_reg[i]) / n, 1) if n else None}
        if near:
            lobes[nm]["near_fissure_pct_of_lobe_part"] = round(100 * float(c_near[i]) / float(c_reg[i]), 1) if c_reg[i] else None
    return {"region_ml": round(n * vml, 2), "region_voxels": n, "lobes": lobes,
            "outside_lobes_pct_of_region": round(100 * float(c_reg[0]) / n, 1) if n else None,
            "near_fissure_mm": near_mm if near else None,
            "near_fissure_pct_of_region": (round(100 * float(c_near.sum()) / n, 1) if n else None) if near else None}


E88_NAME_NEUTRAL = "Dense-lung candidates next to predicted pleural fluid (spatial tag, not a diagnosis)"
E88_NOTE_NEUTRAL = ("spatial tag only: not named compressed lung or atelectasis. Still left out of the historical baseline value E85, "
                    "which is kept unchanged; included in the retained inventory (E92-E94)")
DENSE_INVENTORY_VERSION = "dense_lung_inventory_v2"
DENSE_TAGS = ("no_spatial_tag", "fluid_adjacent", "posterior", "fluid_adjacent_and_posterior")
DENSE_INVENTORY_NOTE = ("HU-selected dense-lung candidates, not a finding. Nothing is excluded by location: 'fluid_adjacent' and "
                        "'posterior' are spatial tags only and name no diagnosis. Attenuation on a non-contrast scan cannot separate "
                        "consolidation, atelectasis, dependent density, tumour, haemorrhage or fluid. No presence / absence limit is "
                        "applied to these rows")


def dense_lung_inventory_rows(d, near_eff, dep, lobe_idx, short, vml, fis, cfg, band, nmm, dep_fraction, lobe_dist=None) -> list[dict]:
    """Evidence rows E92-E94 (dense_lung_inventory_v2): every consolidation-range voxel that is left after lung-edge
    erosion, vessel / airway removal and opening is KEPT. Being within nmm of the predicted pleural fluid and lying in the
    posterior band are recorded as spatial tags; they no longer remove anything and no longer name a diagnosis.

    d: region before the two location steps (RAS); near_eff: within nmm of predicted pleural fluid or None; dep: posterior
    band. The untagged part equals the historical baseline region (E85), which is computed elsewhere and not changed."""
    from scipy import ndimage as _nd
    near = near_eff if near_eff is not None else np.zeros(d.shape, bool)
    masks = {"no_spatial_tag": d & ~near & ~dep, "fluid_adjacent": d & near & ~dep,
             "posterior": d & dep & ~near, "fluid_adjacent_and_posterior": d & near & dep}
    assert int(sum(int(x.sum()) for x in masks.values())) == int(d.sum()), "tags must partition the region"
    near_mm = (lobe_dist or {}).get("near_fissure_mm")
    tags = {}
    for t in DENSE_TAGS:
        x = masks[t]
        n = int(_nd.label(x)[1]) if x.any() else 0
        c = np.bincount(lobe_idx[x], minlength=len(short) + 1) if x.any() else np.zeros(len(short) + 1, int)
        tags[t] = {"ml": round(float(x.sum()) * vml, 1), "pieces": n,
                   "per_lobe_ml": {nm: round(float(c[i + 1]) * vml, 1) for i, nm in enumerate(short)},
                   "outside_lobes_ml": round(float(c[0]) * vml, 1)}
    src = (f"HU {band[0]} to {band[1]} inside predicted lungs (eroded, minus vessels/airways, opened); "
           "no exclusion by distance to pleural fluid or by posterior position")
    rel = "low - HU band inside predicted lungs; candidates only, what they represent is not determined; not validated"
    total = round(float(d.sum()) * vml, 1)
    val = {"version": DENSE_INVENTORY_VERSION, "total_ml": total, "tags": tags,
           "tag_definitions": {"no_spatial_tag": "neither of the two tags (= the historical baseline region, E85)",
                               "fluid_adjacent": f"within {nmm} mm of the predicted pleural-fluid outline" if nmm else "not used (no distance configured)",
                               "posterior": f"in the most posterior {int(100 * dep_fraction)}% of the lung's front-to-back extent, per axial slice and lung",
                               "fluid_adjacent_and_posterior": "both"},
           "pleural_fluid_outline_present": bool(near.any())}

    def r(rid, name, value, unit):
        return {"id": rid, "group": "lung density", "name": name, "value": value, "unit": unit, "reliability": rel, "source": src, "note": DENSE_INVENTORY_NOTE}
    return [r("E92", "Dense-lung candidates (consolidation HU range), all retained", total, "mL"),
            r("E93", "Dense-lung candidates, by lobe", lobe_distribution(d, lobe_idx, short, vml, fis, near_mm) if cfg.get("by_lobe", True) else None, None),
            r("E94", "Dense-lung candidates, spatial tags (not diagnoses)", val, None)]


LOBE_DIST_NOTE = ("full distribution, nothing dropped. pct_of_lobe = share of that whole lobe; pct_of_region = share of the "
                  "selected region; near_fissure = within the stated distance of a predicted boundary between two lobes of "
                  "the same lung. HU-selected region, not a confirmed finding; lobes come from TotalSegmentator outlines")


def image_noise(hu: np.ndarray, ct: nib.Nifti1Image, seg_dir: Path, ck: dict) -> tuple[np.ndarray, list[dict], dict | None]:
    """Blood noise (SD of HU in the eroded aorta). Grainy scans (sharp kernel, tiny pixels) push plain
    blood above the calcium threshold and grainy lung below -950 HU. If SD > limit, the HU-threshold
    measurements use a Gaussian-smoothed copy; if still above the limit they are not reported.
    Returns (image for threshold measurements, evidence rows [E69], info)."""
    nz = ck.get("noise")
    if not nz:
        return hu, [], None
    aorta = _load(seg_dir, "total/aorta", ct)
    rel = ck["reliability"]["noise"]
    if aorta is None or not aorta.any():
        return hu, [{"id": "E69", "group": "image quality", "name": "Image noise (blood SD in aorta)", "value": None,
                     "unit": "HU", "reliability": rel, "source": "aorta mask missing"}], None
    core = ndimage.binary_erosion(aorta, iterations=3)
    core = core if core.sum() > 200 else aorta
    v = hu[core].astype(np.float32)
    sd = float(1.4826 * np.median(np.abs(v - np.median(v))))  # robust SD: calcified plaques do not inflate it
    info = {"sd_hu": round(sd, 1), "smoothed": False, "sd_after_hu": None, "too_noisy": False, "blocked": []}
    out = hu
    if nz.get("mode") == "not_measured":  # v3: never smooth; skip measurements the grain would distort
        info["blocked"] = sorted(k for k, lim in nz["limits_hu"].items() if sd > lim)
        note = ("normal" if not info["blocked"] else "grainy scan: not measured: " + ", ".join(info["blocked"]))
        row = {"id": "E69", "group": "image quality", "name": "Image noise (blood SD in aorta)", "value": info["sd_hu"],
               "unit": "HU", "reliability": rel, "source": "robust SD (1.4826 x median absolute deviation) of CT HU in the eroded aorta mask",
               "note": "limits " + ", ".join(f"{k} {v}" for k, v in nz["limits_hu"].items()) + f" HU; {note}"}
        return out, [row], info
    if sd > nz["blood_sd_limit_hu"]:
        zooms = np.sqrt((ct.affine[:3, :3] ** 2).sum(axis=0))
        out = ndimage.gaussian_filter(hu.astype(np.float32), sigma=[nz["smoothing_sigma_mm"] / z for z in zooms])
        sd2 = float(out[core].std())
        info.update(smoothed=True, sd_after_hu=round(sd2, 1), too_noisy=sd2 > nz["blood_sd_limit_hu"])
    note = ("normal" if not info["smoothed"] else
            f"grainy scan: smoothed (Gaussian {nz['smoothing_sigma_mm']} mm) before threshold measurements, "
            f"SD after {info['sd_after_hu']} HU" + ("; STILL too noisy, emphysema and calcium not reported"
                                                    if info["too_noisy"] else ""))
    row = {"id": "E69", "group": "image quality", "name": "Image noise (blood SD in aorta)", "value": info["sd_hu"],
           "unit": "HU", "reliability": rel, "source": "robust SD (1.4826 x median absolute deviation) of CT HU in the eroded aorta mask",
           "note": f"limit {nz['blood_sd_limit_hu']} HU; {note}"}
    return out, [row], info


def blocked(noise: dict | None, measure: str) -> bool:
    """True if image noise rules out this measurement (v2: still noisy after smoothing; v3: over its limit)."""
    if not noise:
        return False
    if noise.get("blocked") is not None and noise.get("blocked"):
        return measure in noise["blocked"]
    return bool(noise.get("too_noisy")) and measure in ("emphysema", "calcium")


def low_attenuation(hu_r: np.ndarray, lung: np.ndarray, airway: np.ndarray | None, lobe_idx: np.ndarray,
                    short: list[str], vml: float, thr: float, min_voxels: int) -> dict:
    """LAA below `thr` HU inside the eroded lung minus airways: whole-lung %, % per lobe, and clusters
    (26-connected groups of >= min_voxels voxels) per litre of lung."""
    core = ndimage.binary_erosion(lung, iterations=1)
    if airway is not None and airway.any():
        core &= ~ndimage.binary_dilation(airway, iterations=1)
    low = core & (hu_r < thr)
    n_core = int(core.sum())
    nl = len(short) + 1
    c_all = np.bincount(lobe_idx[core], minlength=nl)
    c_low = np.bincount(lobe_idx[low], minlength=nl)
    del core
    per_lobe = {nm: round(100 * float(c_low[i]) / float(c_all[i]), 1)
                for i, nm in enumerate(short, start=1) if c_all[i]}
    n_low = int(low.sum())
    lbl, n = ndimage.label(low, structure=np.ones((3, 3, 3), bool))
    del low
    sizes = np.bincount(lbl.ravel())[1:] if n else np.array([], int)
    del lbl
    big = int((sizes >= min_voxels).sum())
    return {"pct": round(100 * n_low / n_core, 2) if n_core else None, "per_lobe": per_lobe,
            "clusters_per_litre": round(big / (n_core * vml / 1000), 1) if n_core else None,
            "largest_cluster_ml": round(float(sizes.max()) * vml, 2) if n else 0.0}


THICKNESS_GRADES_MM = {"small_below": 10.0, "large_above": 20.0}   # small < 10, moderate 10-20, large > 20 mm


def greatest_thickness_mm(mask: np.ndarray | None, affine: np.ndarray) -> float | None:
    """Diameter (mm) of the largest sphere that fits inside the mask ("greatest thickness", evidence row E79).

    How it is calculated: a 3-D Euclidean distance transform is taken inside the predicted mask, with the voxel spacing
    from the affine, so every inside voxel knows its distance in mm to the nearest voxel outside the mask. The largest
    of these distances is the radius of the biggest sphere that fits; the result is twice that radius.
    0.0 for an empty mask, None if the mask is missing.

    What it is NOT: a caliper measurement. A reader measures fluid, on one chosen image and direction, at a chosen place.
    This number is a property of an automatic outline in 3-D (it can come from a spot where two layers of outline touch,
    and it is quantised by the voxel size). It has not been compared with any clinician measurement, so it must not be
    reported as a pericardial-fluid thickness or converted to a size grade without validation."""
    if mask is None:
        return None
    if not mask.any():
        return 0.0
    idx = np.argwhere(mask)
    lo, hi = np.maximum(idx.min(0) - 1, 0), np.minimum(idx.max(0) + 2, mask.shape)
    sub = np.pad(mask[tuple(slice(a, b) for a, b in zip(lo, hi))], 1)
    zooms = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))
    return float(2.0 * ndimage.distance_transform_edt(sub, sampling=zooms).max())


def thickness_grade(mm: float | None, grades: dict | None = None) -> str | None:
    g = grades or THICKNESS_GRADES_MM
    if mm is None:
        return None
    if mm <= 0:
        return "no fluid outlined"
    return "small" if mm < g["small_below"] else ("large" if mm > g["large_above"] else "moderate")


def pericardial_thickness_row(pc: np.ndarray | None, affine: np.ndarray, ck: dict) -> dict:
    """Evidence row E79 (shown next to the volume E73; not used in any decision yet)."""
    g = ck.get("measurements", {}).get("pericardial_thickness_grades_mm", THICKNESS_GRADES_MM)
    mm = greatest_thickness_mm(pc, affine)
    grade = thickness_grade(mm, g)
    if ck.get("measurements", {}).get("pericardial_thickness_show_grade", True) is False:   # v3_5: no size wording
        return {"id": "E79", "group": "effusion", "name": "Pericardial fluid outline, largest inscribed sphere diameter",
                "value": None if mm is None else round(mm, 1), "unit": "mm",
                "reliability": "property of the automatic outline (largest sphere that fits inside it); NOT a caliper measurement; not validated",
                "source": "TotalSegmentator 2.18 prediction (pleural_pericard_effusion task), 3-D distance transform" if pc is not None else "mask not available",
                **({"note": ("no fluid outlined" if not mm else "largest sphere that fits inside the predicted outline; not a clinician's thickness "
                                                               "measurement and not a size grade; nearly every scan has a few mm outlined")
                             + "; shown, not used in the answer"} if grade else {})}
    return {"id": "E79", "group": "effusion", "name": "Pericardial fluid, greatest thickness",
            "value": None if mm is None else round(mm, 1), "unit": "mm",
            "reliability": ck.get("reliability", {}).get(
                "pericardial_thickness", "low - largest sphere inside the automatic outline; not validated against a radiologist's measurement"),
            "source": "TotalSegmentator 2.18 prediction (pleural_pericard_effusion task), distance transform"
                      if pc is not None else "mask not available",
            **({"note": ("no fluid outlined" if not mm else
                         f"thickest spot of the outlined fluid. If this is an effusion, {mm:.0f} mm would be graded {grade} "
                         f"(small under {g['small_below']:g} mm, moderate up to {g['large_above']:g} mm, large above). "
                         "Nearly every scan has a few mm outlined, so this number alone does not show an effusion")
                        + "; shown, not used in the answer"} if grade else {})}


def measure_conditions(hu: np.ndarray, ct: nib.Nifti1Image, lab: np.ndarray, lobe_masks: dict,
                       lobe_short: dict, seg_dir: Path, ck: dict, noise: dict | None = None,
                       lobe_dist: dict | None = None) -> tuple[list[dict], dict]:
    """Return (evidence rows E70-E92, arrays in the CT's own orientation for figures).

    lobe_dist ({"near_fissure_mm": x}; default: the config's measurements.lobe_distribution, absent before v3_4) adds
    the per-lobe distribution rows E83 / E89 / E91. They are extra rows only: no other row depends on them."""
    m, rel = ck["measurements"], ck["reliability"]
    lobe_dist = m.get("lobe_distribution") if lobe_dist is None else lobe_dist
    aff = ct.affine
    vml = float(abs(np.linalg.det(aff[:3, :3]))) / 1000.0
    sx = _ras_zooms(aff)[0]
    rows: list[dict] = []
    calcium_mask = None

    def row(rid, group, name, value, unit, reliability, source, note=None):
        rows.append({"id": rid, "group": group, "name": name, "value": value, "unit": unit,
                     "reliability": reliability, "source": source, **({"note": note} if note else {})})

    hu_r = _to_ras(hu, aff)
    lab_r = _to_ras(lab, aff)
    lobe_names = list(lobe_masks)
    lobe_idx = np.zeros(lab.shape, np.uint8)
    for i, n in enumerate(lobe_names, start=1):
        lobe_idx[lobe_masks[n]] = i
    lobe_idx = _to_ras(lobe_idx, aff)
    short = [lobe_short[n] for n in lobe_names]
    lung = lab_r > 0
    seg_src = "TotalSegmentator 2.18 prediction"

    # contrast (E70)
    aorta = _load(seg_dir, "total/aorta", ct)
    contrast = "unknown"
    if aorta is not None and aorta.any():
        core = ndimage.binary_erosion(aorta, iterations=2)
        core = core if core.sum() > 50 else aorta
        med = float(np.median(hu[core]))
        c = m["contrast_aorta_hu"]
        contrast = "non-contrast" if med < c["non_contrast_below"] else "contrast" if med > c["contrast_above"] else "unclear"
        row("E70", "contrast", "Blood density in the aorta (median)", round(med), "HU", rel["contrast"],
            "CT HU in eroded aorta mask", f"scan type: {contrast}")
    else:
        row("E70", "contrast", "Blood density in the aorta", None, "HU", rel["contrast"], "aorta mask missing",
            "scan type: unknown")

    # pleural effusion by side (E71, E72), pericardial effusion (E73)
    pe = _load(seg_dir, "pleural_pericard_effusion/pleural_effusion", ct)  # also used for E88
    if pe is not None and lung.any():
        pe_r = _to_ras(pe, aff)
        xs = np.arange(lab_r.shape[0])
        cl = ndimage.center_of_mass(lab_r == 1)[0]
        cr = ndimage.center_of_mass(lab_r == 2)[0]
        right_side = (xs > (cl + cr) / 2)[:, None, None]
        row("E71", "effusion", "Pleural effusion, left", round(float((pe_r & ~right_side).sum()) * vml, 1), "mL",
            rel["effusion_masks"], seg_src + " (pleural_pericard_effusion task)")
        row("E72", "effusion", "Pleural effusion, right", round(float((pe_r & right_side).sum()) * vml, 1), "mL",
            rel["effusion_masks"], seg_src + " (pleural_pericard_effusion task)")
    else:
        row("E71", "effusion", "Pleural effusion", None, "mL", rel["effusion_masks"], "mask not available")
    pc = _load(seg_dir, "pleural_pericard_effusion/pericardial_effusion", ct)
    row("E73", "effusion", "Pericardial effusion", None if pc is None else round(float(pc.sum()) * vml, 1), "mL",
        rel["effusion_masks"], seg_src + " (pleural_pericard_effusion task)" if pc is not None else "mask not available")
    rows.append(pericardial_thickness_row(pc, aff, ck))

    # heart size (E74-E77)
    heart = _load(seg_dir, "total/heart", ct)
    if heart is not None and heart.any():
        h_r = _to_ras(heart, aff)

        def widest(mask):
            best, kz = 0.0, None
            for z in np.where(mask.any(axis=(0, 1)))[0]:
                xs_ = np.where(mask[:, :, z].any(axis=1))[0]
                w = (xs_.max() - xs_.min() + 1) * sx
                if w > best:
                    best, kz = w, int(z)
            return best, kz
        hw, _ = widest(h_r)
        tw, _ = widest(lung)
        row("E74", "heart", "Widest heart width (axial)", round(hw), "mm", rel["cardiothoracic_ratio"], seg_src)
        row("E75", "heart", "Widest inner chest width (lungs, axial)", round(tw), "mm", rel["cardiothoracic_ratio"], seg_src)
        row("E76", "heart", "Cardiothoracic ratio (axial CT)", round(hw / tw, 2) if tw else None, "ratio",
            rel["cardiothoracic_ratio"], "E74 / E75", "axial CT ratio, not the chest X-ray ratio")
        row("E77", "heart", "Heart volume", round(float(heart.sum()) * vml), "mL", rel["heart_volume"], seg_src)
    else:
        row("E76", "heart", "Cardiothoracic ratio", None, "ratio", rel["cardiothoracic_ratio"], "heart mask missing")

    # aortic wall calcium (E78), non-contrast only
    if blocked(noise, "calcium"):
        row("E78", "vessels", "Calcium at the aorta wall", None, "mL", rel["aortic_calcium"],
            "not measured: image too grainy (see E69)")
    elif aorta is not None and aorta.any() and contrast == "non-contrast":
        a_d = ndimage.binary_dilation(aorta, iterations=m["aortic_calcium_mask_dilation_voxels"])
        cand = a_d & (hu >= m["aortic_calcium_hu"])
        lbl, n = ndimage.label(cand)
        keep = 0
        if n:
            sizes = np.bincount(lbl.ravel())[1:]
            peak = ndimage.maximum(hu, lbl, index=np.arange(1, n + 1))
            ok = (sizes >= m["aortic_calcium_min_component_voxels"]) & (np.asarray(peak) >= m.get("aortic_calcium_min_peak_hu", 130))
            keep = int(sizes[ok].sum())
            calcium_mask = np.isin(lbl, np.flatnonzero(ok) + 1)
        row("E78", "vessels", "Calcium at the aorta wall (>= 130 HU spots with a peak >= 300 HU)", round(keep * vml, 3), "mL",
            rel["aortic_calcium"], "CT HU at the predicted aorta (dilated 1 voxel)")
    else:
        row("E78", "vessels", "Calcium at the aorta wall", None, "mL", rel["aortic_calcium"],
            "not measured: " + ("contrast scan" if contrast == "contrast" else "aorta mask missing or scan type unclear"))

    # density types outside the dependent band (E80-E90)
    core = ndimage.binary_erosion(lung, iterations=m["density_lung_erosion_voxels"])
    ex = np.zeros(lung.shape, bool)
    for r_ in m["density_exclude"]:
        e = _load(seg_dir, r_, ct)
        if e is not None:
            ex |= _to_ras(e, aff)
    if ex.any():
        ex = ndimage.binary_dilation(ex, iterations=m["density_exclude_dilation_voxels"])
    base = core & ~ex
    del ex, core
    dep = dependent_band(lab_r, m["dependent_band_fraction"])
    lung_ml = float(lung.sum()) * vml
    out_arrays = {}
    dep_dense = np.zeros(lung.shape, bool)

    # v3: lung within N mm of a pleural effusion is usually compressed (passive atelectasis), not consolidation
    near_eff = None
    nmm = m.get("effusion_adjacent_mm")
    if nmm and pe is not None and pe.any():
        pe_r2 = _to_ras(pe, aff)
        zr = _ras_zooms(aff)
        idx = np.argwhere(pe_r2)
        pad = np.ceil(nmm / zr).astype(int) + 1
        lo = np.maximum(idx.min(0) - pad, 0)
        hi = np.minimum(idx.max(0) + pad + 1, pe_r2.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        dist = ndimage.distance_transform_edt(~pe_r2[sl], sampling=zr)
        near_eff = np.zeros(lung.shape, bool)
        near_eff[sl] = dist <= nmm
        del dist, pe_r2

    if m.get("lung_inflation"):  # v3: breath-in check (E95, E96)
        li = m["lung_inflation"]
        med = float(np.median(hu_r[lung])) if lung.any() else None
        poor = (med is not None and med > li["median_hu_above"]) or lung_ml < li["volume_ml_below"]
        row("E95", "image quality", "Total lung volume (inflation check)", round(lung_ml), "mL",
            rel["lung_inflation"], seg_src, f"poor inflation if below {li['volume_ml_below']} mL")
        row("E96", "image quality", "Median lung density (inflation check)", None if med is None else round(med),
            "HU", rel["lung_inflation"], "CT HU inside predicted lungs",
            f"poor inflation if above {li['median_hu_above']} HU; inflation: {'POOR' if poor else 'adequate'}")

    fis = fissure_distance_mm(lobe_idx, short, _ras_zooms(aff)) if lobe_dist else None
    rel_ld = rel.get("lobe_distribution", "per-lobe split of an HU-selected region by predicted lobe outlines; not validated")

    def dist_row(rid, name, mask, source):
        if not lobe_dist:
            return
        row(rid, "lung density", name, None if mask is None else lobe_distribution(
            mask, lobe_idx, short, vml, fis, lobe_dist.get("near_fissure_mm")), None, rel_ld, source, LOBE_DIST_NOTE)

    for key, band, rid in (("ground_glass", m["ground_glass_hu"], 80), ("consolidation", m["consolidation_hu"], 85)):
        if blocked(noise, key):
            dist_row("E83" if key == "ground_glass" else "E89", ("Ground-glass-range" if key == "ground_glass"
                     else "Consolidation-range") + " region, by lobe", None, "not measured: image too grainy (see E69)")
            name = "Ground-glass-range density" if key == "ground_glass" else "Consolidation-range density"
            for k_, suffix in ((0, "non-dependent lung"), (1, "% of lung"), (2, "location")):
                row(f"E{rid + k_}", "lung density", f"{name}, {suffix}", None, None, rel["density_type"],
                    "not measured: image too grainy (see E69)")
            out_arrays[key] = np.zeros(hu.shape, np.uint8)
            continue
        d = base & (hu_r >= band[0]) & (hu_r < band[1])
        if m["density_opening_voxels"]:
            d = ndimage.binary_opening(d, iterations=m["density_opening_voxels"])
        inv_cfg = m.get("dense_lung_inventory")
        if inv_cfg and key == "consolidation":      # v3_6: keep everything, tag by location (rows E92-E94); baseline rows below are unchanged
            rows.extend(dense_lung_inventory_rows(d, near_eff, dep, lobe_idx, short, vml, fis, inv_cfg, band, nmm,
                                                  m["dependent_band_fraction"], lobe_dist))
            out_arrays["dense_candidates"] = _from_ras(d, aff)
        if near_eff is not None and key == "consolidation":
            adj = d & near_eff
            out_arrays["compressed"] = _from_ras(adj, aff)
            row("E88", "lung density", E88_NAME_NEUTRAL if inv_cfg else "Dense lung next to pleural fluid (likely compressed lung / atelectasis)",
                round(float(adj.sum()) * vml), "mL", rel["density_type"],
                f"consolidation-range lung within {nmm} mm of the predicted pleural effusion",
                E88_NOTE_NEUTRAL if inv_cfg else "counted as atelectasis, not consolidation")
            dist_row("E91", "Dense lung next to pleural fluid, by lobe", adj,
                     f"consolidation-range lung within {nmm} mm of the predicted pleural effusion")
            d &= ~near_eff
        nd = d & ~dep
        dep_dense |= d & dep
        ml = float(nd.sum()) * vml
        name = "Ground-glass-range density" if key == "ground_glass" else "Consolidation-range density"
        src = f"HU {band[0]} to {band[1]} inside predicted lungs (eroded, minus vessels/airways), outside the dependent band"
        row(f"E{rid}", "lung density", f"{name}, non-dependent lung", round(ml), "mL", rel["density_type"], src)
        row(f"E{rid + 1}", "lung density", f"{name}, % of lung", round(100 * ml / lung_ml, 2) if lung_ml else None,
            "% of lung", rel["density_type"], src)
        loc = location(nd, lab_r, lobe_idx, short)
        row(f"E{rid + 2}", "lung density", f"{name}, location", loc, None, rel["density_type"], src)
        dist_row("E83" if key == "ground_glass" else "E89", f"{name} region, by lobe", nd, src)
        out_arrays[key] = _from_ras(nd, aff)
    if nmm and not any(r_["id"] == "E88" for r_ in rows):
        row("E88", "lung density", E88_NAME_NEUTRAL if m.get("dense_lung_inventory") else "Dense lung next to pleural fluid (likely compressed lung / atelectasis)", 0, "mL",
            rel["density_type"], "no pleural effusion found" if (pe is None or not pe.any()) else "not measured")
    row("E90", "lung density", "Dense lung in the dependent (back) band, both ranges",
        round(float(dep_dense.sum()) * vml), "mL", rel["dependent_density"],
        f"most posterior {int(100 * m['dependent_band_fraction'])}% of each lung per slice",
        "context for atelectasis; usually normal in a supine patient")
    out_arrays["dependent"] = _from_ras(dep & lung, aff)
    if calcium_mask is not None:
        out_arrays["calcium"] = calcium_mask  # already in the CT's own orientation
    del base, dep, dep_dense, near_eff, fis
    if m.get("laa_cluster_min_voxels"):  # v3: emphysema measured here (airways excluded) - replaces E30 value
        if blocked(noise, "emphysema"):
            laa = None
        else:
            aw = _load(seg_dir, "lung_vessels/lung_airways", ct)
            laa = low_attenuation(hu_r, lung, None if aw is None else _to_ras(aw, aff), lobe_idx, short, vml,
                                  m["emphysema_hu"], m["laa_cluster_min_voxels"])
        src = "CT HU inside predicted lungs (eroded 1 voxel, airways removed)"
        row("E31", "emphysema", "Lung below -950 HU, airways excluded (LAA-950)", None if laa is None else laa["pct"],
            "% of lung", rel["emphysema_laa950"], src if laa else "not measured: image too grainy (see E69)")
        row("E32", "emphysema", "LAA-950 by lobe", None if laa is None else laa["per_lobe"], "%", rel["laa_clusters"], src,
            "report only")
        row("E33", "emphysema", "Dark clusters (>= %d voxels) per litre of lung" % m["laa_cluster_min_voxels"],
            None if laa is None else laa["clusters_per_litre"], "per L", rel["laa_clusters"], src,
            None if laa is None else f"largest cluster {laa['largest_cluster_ml']} mL; report only, no threshold yet")
    return rows, {"contrast": contrast, "arrays": out_arrays}
