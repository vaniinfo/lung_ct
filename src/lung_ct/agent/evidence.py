"""Pipeline step 1: build the evidence table for one case from the CT and PREDICTED masks.

Nothing here reads the expert (reference) masks: the evidence is what the agent may cite.
Each row has an id (E..), a value, a unit, a source and a reliability tag taken from
configs/agent_evidence_v1.json, so the report writer can weigh it.

Distribution answers (laterality / cranio-caudal / axial) are derived from the dense-lung
proxy with the SAME rules and code used to derive reference answers from expert masks
(lung_ct.evaluation.reference_answers, rules v2), so evidence and reference are comparable.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from lung_ct.evaluation.reference_answers import load_rules, reference_answers

LOBES = {  # TotalSegmentator name -> (side, label used for the side mask, short name)
    "lung_upper_lobe_right": ("right", 2, "RUL"),
    "lung_middle_lobe_right": ("right", 2, "RML"),
    "lung_lower_lobe_right": ("right", 2, "RLL"),
    "lung_upper_lobe_left": ("left", 1, "LUL"),
    "lung_lower_lobe_left": ("left", 1, "LLL"),
}


def load_config(root: Path, path: str | None = None) -> dict:
    p = Path(path) if path else root / "configs" / "agent_evidence_v1.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _mask(path: Path, ref: nib.Nifti1Image) -> np.ndarray:
    img = nib.load(str(path))
    if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
        raise ValueError(f"{path.name} is not on the CT grid")
    return np.asanyarray(img.dataobj) > 0


def opacity_proxy(hu: np.ndarray, lung: np.ndarray, exclude: np.ndarray, cfg: dict) -> np.ndarray:
    c = cfg["opacity_proxy"]
    core = ndimage.binary_erosion(lung, iterations=c["lung_erosion_voxels"])
    ex = ndimage.binary_dilation(exclude, iterations=c["exclude_dilation_voxels"]) if exclude.any() else exclude
    lo, hi = c["hu_band"]
    m = core & ~ex & (hu >= lo) & (hu <= hi)
    return ndimage.binary_opening(m, iterations=c["opening_voxels"]) if c["opening_voxels"] else m


def build_evidence(root: Path, case_id: str, ct_path: Path, seg_dir: Path,
                   airway_summary: Path | None, cfg: dict,
                   classifier: dict | None = None,
                   checklist_cfg: dict | None = None) -> tuple[dict, nib.Nifti1Image, dict]:
    """Return (evidence dict, CT image, arrays for figures)."""
    rules = load_rules(str(root / cfg["reference_rules"]))
    rel = cfg["reliability"]
    ct = nib.load(str(ct_path))
    hu = np.asanyarray(ct.dataobj).astype(np.float32)
    vml = float(abs(np.linalg.det(ct.affine[:3, :3]))) / 1000.0

    lab = np.zeros(ct.shape, np.uint8)
    lobe_masks = {}
    for name, (_, side_label, _) in LOBES.items():
        m = _mask(seg_dir / "total" / f"{name}.nii.gz", ct)
        lobe_masks[name] = m
        lab[m] = side_label
    lung = lab > 0
    exclude = np.zeros(ct.shape, bool)
    for rel_path in cfg["opacity_proxy"]["exclude"]:
        p = seg_dir / f"{rel_path}.nii.gz"
        if p.exists():
            exclude |= _mask(p, ct)
    prox = opacity_proxy(hu, lung, exclude, cfg)

    lab_img = nib.Nifti1Image(lab, ct.affine)
    prox_img = nib.Nifti1Image(prox.astype(np.uint8), ct.affine)
    lobe_imgs = {n: nib.Nifti1Image(m.astype(np.uint8), ct.affine) for n, m in lobe_masks.items()}
    derived = reference_answers(prox_img, lab_img, lobe_imgs, rules)
    meas = derived["measurements"]

    rows = []

    def row(rid, group, name, value, unit, reliability, source, note=None):
        rows.append({"id": rid, "group": group, "name": name, "value": value, "unit": unit,
                     "reliability": reliability, "source": source, **({"note": note} if note else {})})

    seg_src = "TotalSegmentator 2.18 prediction"
    left_ml, right_ml = meas["left"]["lung_ml"], meas["right"]["lung_ml"]
    row("E01", "lungs", "Left lung volume", round(left_ml), "mL", rel["lung_volumes"], seg_src)
    row("E02", "lungs", "Right lung volume", round(right_ml), "mL", rel["lung_volumes"], seg_src)
    for i, (name, (side, _, short)) in enumerate(LOBES.items(), start=3):
        row(f"E{i:02d}", "lungs", f"{short} volume", round(int(lobe_masks[name].sum()) * vml),
            "mL", rel["lobe_volumes"], seg_src)

    prox_src = (f"dense-lung proxy, HU {cfg['opacity_proxy']['hu_band'][0]} to "
                f"{cfg['opacity_proxy']['hu_band'][1]} inside predicted lungs minus vessels/airways")
    tot_ml = meas["infection_ml_inside_lung"]
    row("E10", "dense lung", "Dense lung tissue, total", round(tot_ml), "mL", rel["opacity_proxy_amount"],
        prox_src, cfg["opacity_proxy"]["note"])
    row("E11", "dense lung", "Dense lung tissue, left lung", round(meas["left"]["infection_ml"]), "mL",
        rel["opacity_proxy_amount"], prox_src,
        f"{meas['left']['infection_pct_of_lung']:.1f}% of left lung")
    row("E12", "dense lung", "Dense lung tissue, right lung", round(meas["right"]["infection_ml"]), "mL",
        rel["opacity_proxy_amount"], prox_src,
        f"{meas['right']['infection_pct_of_lung']:.1f}% of right lung")
    r_share = meas["right"]["infection_ml"] / tot_ml if tot_ml else None
    row("E13", "distribution", "Share of dense tissue in the right lung",
        round(100 * r_share) if r_share is not None else None, "%", rel["opacity_proxy_distribution"],
        prox_src, f"derived laterality (rules v2, 5% rule): {derived['answers']['distribution'].get('laterality')}")
    sh = meas.get("craniocaudal", {}).get("infection_share_by_zone", {})
    row("E14", "distribution", "Share of dense tissue by zone (equal-volume thirds)",
        {k: round(100 * v) for k, v in sh.items()}, "%", rel["opacity_proxy_distribution"], prox_src,
        f"derived cranio-caudal: {derived['answers']['distribution'].get('craniocaudal')}")
    ps = meas.get("axial", {}).get("peripheral_share")
    row("E15", "distribution", "Share of dense tissue in the outer third of each lung",
        round(100 * ps) if ps is not None else None, "%", rel["opacity_proxy_distribution"], prox_src,
        f"derived axial: {derived['answers']['distribution'].get('axial')}")
    ctss = meas.get("ct_severity_score", {})
    for i, (name, (_, _, short)) in enumerate(LOBES.items(), start=20):
        lo = ctss.get("lobes", {}).get(name, {})
        row(f"E{i:02d}", "severity", f"{short}: % of lobe with dense tissue",
            None if lo.get("infection_pct") is None else round(lo["infection_pct"], 1), "%",
            rel["opacity_proxy_amount"], prox_src, f"lobe score {lo.get('score')} (0-5)")
    row("E25", "severity", "Dense-tissue severity score (sum of 5 lobe scores, 0-25)", ctss.get("total"),
        "points", rel["opacity_proxy_amount"], prox_src,
        "same 0-5 per-lobe bins as the CT severity score, applied to the proxy; not a radiologist score")

    hu_thr, noise_rows, noise = hu, [], None
    if checklist_cfg is not None:
        from lung_ct.agent.conditions import image_noise
        hu_thr, noise_rows, noise = image_noise(hu, ct, seg_dir, checklist_cfg)
    thr = cfg["emphysema_threshold_hu"]
    from lung_ct.agent.conditions import blocked
    if blocked(noise, "emphysema"):
        row("E30", "emphysema", f"Lung below {thr} HU (LAA-950)", None, "% of lung", rel["emphysema_laa950"],
            "not measured: image too grainy (see E69)")
    else:
        laa = 100.0 * float((hu_thr[lung] < thr).mean())
        row("E30", "emphysema", f"Lung below {thr} HU (LAA-950)", round(laa, 1), "% of lung",
            rel["emphysema_laa950"], "CT HU inside predicted lungs"
            + (" (smoothed copy, grainy scan - see E69)" if noise and noise["smoothed"] else ""))
    rows.extend(noise_rows)

    if airway_summary and airway_summary.exists():
        a = json.loads(airway_summary.read_text(encoding="utf-8"))["summary"]
        row("E40", "airways", "Airway tree branches (total / terminal)",
            f"{a['n_tree_branches']} / {a['n_terminal_branches']}", "branches", rel["airway_tree"],
            "lung_ct airway tree v2 on TotalSegmentator airways (NOT AirQuant)")
        row("E41", "airways", "Deepest airway generation detected", a["max_generation"], "generation",
            rel["airway_tree"], "lung_ct airway tree v2")
        row("E42", "airways", "Airway branches by side (left / right)",
            f"{a['branches_by_side'].get('left')} / {a['branches_by_side'].get('right')}", "branches",
            rel["airway_tree"], "lung_ct airway tree v2",
            "; ".join(a.get("flags", [])) or None)
    else:
        row("E40", "airways", "Airway tree", None, None, rel["airway_tree"], "not available for this case")

    if classifier and any(not np.isfinite(f["p_present"]) for f in classifier["findings"]):
        classifier = None  # never cite non-finite scores; step 2 shown as not run
    if classifier:
        for i, f in enumerate(classifier["findings"]):
            row(f"E{50 + i:02d}", "classifier", f"{f['label']} (classifier probability)", f["p_present"],
                "p", classifier["reliability"], f"{classifier['model']} zero-shot",
                None if f.get("ct_rate_label", True) else "prompt outside the model's 18 training labels")
            rows[-1]["ckey"] = f["key"]
    else:
        row("E50", "classifier", "Finding classifier (pipeline step 2)", None, None, "n/a",
            "NOT RUN for this case (COLIPRI)")

    extra = {}
    if checklist_cfg is not None:
        from lung_ct.agent.conditions import measure_conditions
        short = {n: v[2] for n, v in LOBES.items()}
        crow, extra = measure_conditions(hu_thr, ct, lab, lobe_masks, short, seg_dir, checklist_cfg, noise)
        rows.extend(crow)

    ev = {"case_id": case_id, "config_version": cfg["config_version"], "config_status": cfg["status"],
          "pilot_validation": cfg["pilot_validation"], "rows": rows,
          "derived_answers": derived["answers"], "rules_version": derived["rules_version"],
          "classifier": {"model": classifier["model"], "high_threshold": classifier["high_threshold"],
                         "low_threshold": classifier["low_threshold"]} if classifier else None,
          "inputs": {"ct": str(ct_path.relative_to(root)), "segmentation": str(seg_dir.relative_to(root)),
                     "airway_summary": str(airway_summary.relative_to(root)) if airway_summary else None},
          "reference_masks_used": False}
    arrays = {"hu": hu, "lab": lab, "proxy": prox, **extra.get("arrays", {})}
    if extra:
        ev["scan_type"] = extra["contrast"]
        ev["checklist_config"] = checklist_cfg["config_version"]
        ev["image_noise"] = noise
    return ev, ct, arrays
