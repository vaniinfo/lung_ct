"""Hybrid experiment (core env): put numbers and location on HoloCT finding masks with our own anatomy.

For every mask HoloCT drew (runs/P3_holoct_*/<case>/findings_grounded/mask_XXX.nii.gz):
  HoloCT finding mask  ->  intersect with each TotalSegmentator lobe  ->  count voxels  ->  mL and % of the lobe
plus how much of the mask lies outside the lungs, inside the TotalSegmentator pleural / pericardial fluid
masks, and the median HU inside it (a plausibility check: fluid ~0-20 HU, consolidation ~-100..60 HU,
ground-glass ~-700..-300 HU, emphysema < -950 HU).

Geometry is checked first: the HoloCT mask, the CT and the lobe masks must have the same shape and affine;
otherwise the case is reported as failed and nothing is measured. Volume in mL = voxels x voxel volume / 1000.
Exploratory: no thresholds, no decisions, nothing is fed back into the checklist. Research use only.

Usage: python scripts/holoct_hybrid.py --run runs/P3_holoct_meeting4_<time>
Writes runs/E9_holoct_hybrid_<time>/{hybrid.json,hybrid.md}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.evidence import LOBES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    out = ROOT / "runs" / f"E9_holoct_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    result = {"holoct_run": run.name, "cases": []}
    md = ["# Hybrid: HoloCT finding masks measured with TotalSegmentator lobes", "",
          f"HoloCT run: `{run.name}`. Exploratory; not a diagnosis.", ""]
    for gdir in sorted(run.glob("*/findings_grounded")):
        cid = gdir.parent.name
        rec = {"case_id": cid, "status": "ok", "findings": []}
        result["cases"].append(rec)
        md += [f"## {cid}", ""]
        try:
            res = json.loads((gdir / "result.json").read_text(encoding="utf-8"))
            ct = nib.load(str(ROOT / "data/raw/ct_rate/volumes" / f"{cid}.nii.gz"))
            hu = np.asanyarray(ct.dataobj).astype(np.float32)
            seg = ROOT / "data" / "derived" / cid / "segmentation"
            vox_ml = abs(np.linalg.det(ct.affine[:3, :3])) / 1000.0

            def load(path):
                img = nib.load(str(path))
                if img.shape != ct.shape or not np.allclose(img.affine, ct.affine, atol=1e-3):
                    raise ValueError(f"geometry differs from the CT: {Path(path).name} shape {img.shape} vs {ct.shape}")
                return np.asanyarray(img.dataobj) > 0

            lobes = {short: load(seg / "total" / f"{name}.nii.gz") for name, (_, _, short) in LOBES.items()}
            lung = np.zeros(ct.shape, bool)
            for m in lobes.values():
                lung |= m
            pleural = load(seg / "pleural_pericard_effusion" / "pleural_effusion.nii.gz")
            pericard = load(seg / "pleural_pericard_effusion" / "pericardial_effusion.nii.gz")
            lobe_ml = {k: float(v.sum() * vox_ml) for k, v in lobes.items()}
            rec["lobe_volume_ml"] = {k: round(v, 1) for k, v in lobe_ml.items()}
            rec["tseg_pleural_effusion_ml"] = round(float(pleural.sum() * vox_ml), 1)
            if not res["masks"]:
                md += ["HoloCT drew no masks for this case.", ""]
            for m in res["masks"]:
                mask = load(gdir / m["file"])
                n = int(mask.sum())
                f = {"label": m["label"], "file": m["file"], "voxels": n, "total_ml": round(n * vox_ml, 2)}
                if n == 0:
                    f["note"] = "empty mask"
                else:
                    f["median_hu"] = round(float(np.median(hu[mask])), 0)
                    f["per_lobe"] = {k: {"ml": round(float((mask & v).sum() * vox_ml), 2),
                                         "pct_of_lobe": round(100.0 * float((mask & v).sum() * vox_ml) / lobe_ml[k], 2) if lobe_ml[k] else None}
                                     for k, v in lobes.items()}
                    in_lung = float((mask & lung).sum() * vox_ml)
                    f["in_lung_ml"] = round(in_lung, 2)
                    f["pct_of_all_lung"] = round(100.0 * in_lung / sum(lobe_ml.values()), 2)
                    f["outside_lung_ml"] = round(n * vox_ml - in_lung, 2)
                    f["in_tseg_pleural_fluid_ml"] = round(float((mask & pleural).sum() * vox_ml), 2)
                    f["in_tseg_pericardial_fluid_ml"] = round(float((mask & pericard).sum() * vox_ml), 2)
                    side_r = sum(f["per_lobe"][k]["ml"] for k in ("RUL", "RML", "RLL"))
                    f["right_share_pct_of_in_lung"] = round(100.0 * side_r / in_lung, 0) if in_lung > 0 else None
                rec["findings"].append(f)
                md += [f"**{m['label']}** (`{m['file']}`)", ""]
                if n == 0:
                    md += ["- Empty mask: HoloCT named the finding but drew nothing.", ""]
                    continue
                md += [f"- Total {f['total_ml']:g} mL, median density {f['median_hu']:g} HU; in lung {f['in_lung_ml']:g} mL "
                       f"({f['pct_of_all_lung']:g}% of all lung), outside the lung outlines {f['outside_lung_ml']:g} mL; "
                       f"inside TotalSegmentator pleural fluid {f['in_tseg_pleural_fluid_ml']:g} mL, pericardial fluid {f['in_tseg_pericardial_fluid_ml']:g} mL.",
                       "", "| lobe | mL | % of that lobe |", "|---|---|---|"]
                md += [f"| {k} | {v['ml']:g} | {v['pct_of_lobe']:g} |" for k, v in f["per_lobe"].items()]
                md.append("")
        except Exception as e:  # failed cases are reported, never dropped
            rec["status"] = f"FAILED: {type(e).__name__}: {e}"
            md += [f"FAILED: {type(e).__name__}: {e}", ""]
    (out / "hybrid.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "hybrid.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
