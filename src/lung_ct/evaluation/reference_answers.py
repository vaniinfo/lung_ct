"""Derive reference answers to the VLM questions from expert masks.

Answers use the same field names and enums as lung_ct.vlm.prompting, so any model whose
output follows that schema (MedGemma, Qwen3-VL, ...) can be scored the same way.

Rules and thresholds live in configs/reference_answer_rules_v1.json (PROVISIONAL).
Geometry rules: side is decided from world coordinates (RAS), never from array index;
cranio-caudal zones use world z; distances are in mm using voxel spacing.

Reference masks are reference, not predictions. Nothing here is ever given to a model.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from lung_ct.qc.laterality import label_centroids_ras

RULES_PATH = Path(__file__).resolve().parents[3] / "configs" / "reference_answer_rules_v1.json"
LOBES = ("lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
         "lung_upper_lobe_left", "lung_lower_lobe_left")


@lru_cache(maxsize=4)
def load_rules(path: str | None = None) -> dict:
    p = Path(path) if path else RULES_PATH
    raw = p.read_bytes()
    rules = json.loads(raw)
    rules["_sha256"] = hashlib.sha256(raw).hexdigest()
    rules["_path"] = str(p)
    return rules


def _same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image, tol: float = 1e-3) -> bool:
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=tol)


def assign_sides(lung_img: nib.Nifti1Image) -> dict:
    """Map lung labels to sides from world x (RAS+: +x = patient right)."""
    data = np.asanyarray(lung_img.dataobj)
    labels = sorted(int(v) for v in np.unique(data) if v > 0)
    if len(labels) != 2:
        raise ValueError(f"expected 2 lung labels, found {labels}")
    c = label_centroids_ras(lung_img, tuple(labels))
    left, right = sorted(labels, key=lambda lab: c[lab][0])
    return {"left_label": left, "right_label": right,
            "centroid_x_mm": {"left": c[left][0], "right": c[right][0]}}


def _world_z(idx: tuple[np.ndarray, np.ndarray, np.ndarray], affine: np.ndarray) -> np.ndarray:
    a = affine[2]
    return a[0] * idx[0] + a[1] * idx[1] + a[2] * idx[2] + a[3]


def _depth_fraction(side_mask: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Distance to the lung edge (mm) / max distance, inside one lung; 0 outside."""
    out = np.zeros(side_mask.shape, dtype=np.float32)
    if not side_mask.any():
        return out
    sl = ndimage.find_objects(side_mask.astype(np.uint8))[0]
    sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
    d = ndimage.distance_transform_edt(side_mask[sl], sampling=spacing).astype(np.float32)
    dmax = float(d.max())
    if dmax > 0:
        out[sl] = d / dmax
    return out


def _ctss_bin(pct: float, bins: list) -> int:
    if pct <= 0:
        return 0
    for lo, hi, score in bins[1:]:
        if lo < pct <= hi:
            return score
    return bins[-1][2]


def reference_answers(infection_img: nib.Nifti1Image, lung_img: nib.Nifti1Image,
                      lobe_imgs: dict[str, nib.Nifti1Image] | None = None,
                      rules: dict | None = None) -> dict:
    """Reference values for opacity_presence and distribution.{laterality,craniocaudal,axial}.

    Also returns the measurements behind each answer, and (if lobe masks on the same grid
    are supplied) a near-reference CT severity score.
    """
    rules = rules or load_rules()
    if not _same_grid(infection_img, lung_img):
        raise ValueError("infection and lung masks are not on the same grid")
    spacing = np.sqrt((lung_img.affine[:3, :3] ** 2).sum(axis=0))
    vml = float(abs(np.linalg.det(lung_img.affine[:3, :3]))) / 1000.0

    sides = assign_sides(lung_img)
    lung = np.asanyarray(lung_img.dataobj)
    inf_all = np.asanyarray(infection_img.dataobj) > 0
    lung_any = lung > 0
    inf = inf_all & lung_any
    side_masks = {s: lung == sides[f"{s}_label"] for s in ("left", "right")}

    m = {"voxel_ml": vml,
         "infection_ml_inside_lung": float(inf.sum() * vml),
         "infection_ml_outside_lung": float((inf_all & ~lung_any).sum() * vml),
         "sides": sides}
    for s, sm in side_masks.items():
        n_lung, n_inf = int(sm.sum()), int((inf & sm).sum())
        m[s] = {"lung_ml": n_lung * vml, "infection_ml": n_inf * vml,
                "infection_pct_of_lung": 100.0 * n_inf / n_lung if n_lung else None}

    ans: dict = {"distribution": {}}
    present = m["infection_ml_inside_lung"] >= rules["opacity_presence"]["min_infection_ml_inside_lung"]
    ans["opacity_presence"] = "present" if present else "absent"
    if not present:
        ans["distribution"] = {k: "not_applicable" for k in ("laterality", "craniocaudal", "axial")}
        return {"answers": ans, "measurements": m, "rules_version": rules["rules_version"],
                "rules_sha256": rules["_sha256"]}

    # laterality
    l_ml, r_ml = m["left"]["infection_ml"], m["right"]["infection_ml"]
    tot = l_ml + r_ml
    small = min(l_ml, r_ml) / tot
    thr = rules["laterality"]["unilateral_fraction_threshold"]
    ans["distribution"]["laterality"] = ("bilateral" if small >= thr
                                         else ("right" if r_ml > l_ml else "left"))
    m["laterality"] = {"smaller_side_share": small}

    # cranio-caudal zones along world z (RAS +z = superior): lower, middle, upper
    #   equal_height: thirds of the lungs' z extent (v1)
    #   equal_volume: boundaries at the 1/3 and 2/3 quantiles of lung-voxel z, so each zone
    #                 holds a third of the lung volume (v2)
    method = rules["craniocaudal"].get("zone_method", "equal_height")
    lz = _world_z(np.nonzero(lung_any), lung_img.affine)
    zmin, zmax = float(lz.min()), float(lz.max())
    if method == "equal_height":
        cuts = np.linspace(zmin, zmax, 4)[1:-1]
    elif method == "equal_volume":
        cuts = np.quantile(lz, [1 / 3, 2 / 3])
    else:
        raise ValueError(f"unknown craniocaudal zone_method {method!r}")
    del lz
    iz = _world_z(np.nonzero(inf), lung_img.affine)
    zone = np.clip(np.digitize(iz, cuts), 0, 2)
    shares = {name: float((zone == i).mean()) for i, name in enumerate(("lower", "middle", "upper"))}
    best = max(shares, key=shares.get)
    ans["distribution"]["craniocaudal"] = (
        f"{best}_predominant" if shares[best] >= rules["craniocaudal"]["predominance_min_share"]
        else "no_predominance")
    m["craniocaudal"] = {"zone_method": method, "lung_z_range_mm": [zmin, zmax],
                         "zone_cuts_z_mm": [float(c) for c in cuts],
                         "infection_share_by_zone": shares}

    # axial: share of infection in the outer depth fraction of its own lung
    a = rules["axial"]
    periph_n = 0
    for s, sm in side_masks.items():
        frac = _depth_fraction(sm, spacing)
        periph_n += int((inf & sm & (frac <= a["peripheral_depth_fraction"])).sum())
    p_share = periph_n / int(inf.sum())
    ans["distribution"]["axial"] = ("peripheral" if p_share >= a["peripheral_if_share_at_least"]
                                    else "central" if p_share <= a["central_if_share_at_most"]
                                    else "mixed")
    m["axial"] = {"peripheral_share": p_share}

    # near-reference CT severity score from predicted lobes
    if lobe_imgs:
        bins = rules["ct_severity_score"]["bins_pct"]
        lobes = {}
        for name, img in lobe_imgs.items():
            if not _same_grid(img, lung_img):
                lobes[name] = {"error": "lobe mask not on the CT grid"}
                continue
            lm = np.asanyarray(img.dataobj) > 0
            n = int(lm.sum())
            pct = 100.0 * int((inf_all & lm).sum()) / n if n else None
            lobes[name] = {"lobe_ml": n * vml, "infection_pct": pct,
                           "score": _ctss_bin(pct, bins) if pct is not None else None}
        scores = [v.get("score") for v in lobes.values()]
        m["ct_severity_score"] = {
            "source": "expert infection mask x TotalSegmentator predicted lobes (near-reference)",
            "lobes": lobes,
            "total": sum(scores) if all(isinstance(x, int) for x in scores) and len(scores) == 5 else None}

    return {"answers": ans, "measurements": m, "rules_version": rules["rules_version"],
            "rules_sha256": rules["_sha256"]}


def load_case_reference(case_id: str, root: Path, rules: dict | None = None) -> dict:
    """Load expert masks (and predicted lobes if present) for a case and derive answers."""
    raw = root / "data" / "raw" / "covid_ct"
    inf_p, lung_p = raw / "Infection_Mask" / f"{case_id}.nii.gz", raw / "Lung_Mask" / f"{case_id}.nii.gz"
    lobe_dir = root / "data" / "derived" / case_id / "segmentation" / "total"
    lobes = {n: nib.load(str(lobe_dir / f"{n}.nii.gz")) for n in LOBES
             if (lobe_dir / f"{n}.nii.gz").exists()}
    ref = reference_answers(nib.load(str(inf_p)), nib.load(str(lung_p)),
                            lobes if len(lobes) == 5 else None, rules)
    ref["case_id"] = case_id
    ref["inputs"] = {"infection_mask": str(inf_p.relative_to(root)),
                     "lung_mask": str(lung_p.relative_to(root)),
                     "lobe_masks": str(lobe_dir.relative_to(root)) if len(lobes) == 5 else None}
    return ref
