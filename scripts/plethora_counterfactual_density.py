"""Counterfactual input test for the traced PleThora scans (segmentation env). Diagnostic only; nothing here is a method.

Question: is a missed reference layer missed because of its DENSITY or because of its SHAPE (thin layer)?
Test: copy the CT, set every voxel inside the first-reviewer reference outline to water-like density (10 HU plus the
scan's own noise level), leave everything else untouched, and run the unchanged pleural_pericard_effusion task again.
  - outline found now  -> the model can segment a layer of this shape and place; density kept it from doing so.
  - still not found    -> shape / thickness (or place) is enough to make the model miss it.
The edited CT exists only in memory and is never written. Seed fixed.

Usage (Git Bash): environments/segmentation/Scripts/python.exe scripts/plethora_counterfactual_density.py --out runs/E16_.../counterfactual_density.json LUNG1-013 ...
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOTALSEG_HOME_DIR", str(ROOT / "models" / "totalsegmentator"))
TASK = "pleural_pericard_effusion"
P = ROOT / "data" / "raw" / "plethora"
FLUID_HU, SEED = 10.0, 20261123


def main() -> int:
    import nibabel as nib
    import numpy as np
    from totalsegmentator.python_api import totalsegmentator

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = Path(os.environ["TOTALSEG_HOME_DIR"]) / "config.json"
    if not cfg.exists() or json.loads(cfg.read_text()).get("send_usage_stats") is not False:
        raise SystemExit(f"{cfg}: usage statistics must be switched off before running")
    res = {}
    for pid in args.cases:
        ct = nib.load(P / "volumes" / f"{pid}.nii.gz")
        a = np.asarray(ct.dataobj).astype(np.float32)
        ref_img = nib.load(P / "effusions" / "Effusions" / pid / f"{pid}_effusion_first_reviewer.nii.gz")
        rc, cc = nib.as_closest_canonical(ref_img), nib.as_closest_canonical(ct)
        assert rc.shape == cc.shape and np.allclose(rc.affine, cc.affine, atol=1e-2)
        back = nib.orientations.ornt_transform(nib.orientations.io_orientation(cc.affine), nib.orientations.io_orientation(ct.affine))
        ref = nib.orientations.apply_orientation(np.asarray(rc.dataobj) > 0, back)
        vml = float(np.prod(ct.header.get_zooms()[:3])) / 1000.0
        soft = a[(a > 20) & (a < 80)]
        noise = float(np.std(soft)) if soft.size > 1000 else 15.0     # rough noise level from soft tissue of this scan
        rng = np.random.default_rng(SEED)
        b = a.copy()
        b[ref] = FLUID_HU + rng.normal(0, noise, int(ref.sum()))
        with tempfile.TemporaryDirectory(dir=str(ROOT / "data" / "derived" / "plethora_trace")) as td:
            totalsegmentator(nib.Nifti1Image(b.astype(np.int16), ct.affine), Path(td), quiet=True, task=TASK, fast=False, device="gpu")
            seg = np.asarray(nib.load(Path(td) / "pleural_effusion.nii.gz").dataobj) > 0
        inter = float((seg & ref).sum())
        res[pid] = {"reference_ml": round(float(ref.sum()) * vml, 1), "original_mean_hu_in_reference": round(float(a[ref].mean()), 0),
                    "noise_sd_used_hu": round(noise, 1), "tool_ml_after_edit": round(float(seg.sum()) * vml, 1),
                    "fraction_of_reference_outlined_after_edit": round(inter / max(1.0, float(ref.sum())), 3),
                    "dice_after_edit": round(2 * inter / max(1.0, float(seg.sum() + ref.sum())), 3)}
        print(pid, res[pid], flush=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fluid_hu": FLUID_HU, "seed": SEED, "note": "edited CT kept in memory only; diagnostic, not a method", "cases": res}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
