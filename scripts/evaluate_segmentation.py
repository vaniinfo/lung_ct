"""Phase 2 (core env): evaluate TotalSegmentator outputs against reference lung masks.

Reads a segmentation run (scripts/segment_totalseg.py), and for each case computes
predicted lobe volumes, predicted left/right lung vs reference (Dice, volume error),
predicted laterality, airway/vessel volumes, and segmentation overlay QC PNGs.

Usage:
    python scripts/evaluate_segmentation.py --seg-run runs/P2_seg_1cases_<ts>
"""
import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lung_ct.evaluation.lung_seg import LOBES, evaluate_lung_segmentation, same_grid  # noqa: E402
from lung_ct.qc.qc_views import generate_qc_views  # noqa: E402

DATA_DIR = ROOT / "data" / "raw" / "covid_ct"
VESSELS = ["lung_airways", "lung_airways_wall", "lung_arteries", "lung_veins"]
LOBE_COLORS = {  # distinguishable, colour-blind-aware
    "lung_upper_lobe_left": (0.0, 0.45, 0.70),
    "lung_lower_lobe_left": (0.80, 0.47, 0.65),
    "lung_upper_lobe_right": (0.90, 0.62, 0.0),
    "lung_middle_lobe_right": (0.0, 0.62, 0.45),
    "lung_lower_lobe_right": (0.84, 0.37, 0.0),
}
LEFT_LABELS = [LOBES.index(n) + 1 for n in LOBES if n.endswith("_left")]
RIGHT_LABELS = [LOBES.index(n) + 1 for n in LOBES if n.endswith("_right")]


def _load(path: Path):
    return nib.load(str(path)) if path.exists() else None


def evaluate_case(seg_rec: dict, qc_dir: Path) -> dict:
    cid = seg_rec["case_id"]
    seg_dir = ROOT / seg_rec["output_dir"]
    ct_img = nib.load(str(ROOT / seg_rec["ct_file"]))
    ref_img = nib.load(str(DATA_DIR / "Lung_Mask" / f"{cid}.nii.gz"))
    if not same_grid(ct_img, ref_img):
        raise ValueError("reference lung mask not on CT grid")

    lobe_imgs = {n: _load(seg_dir / "total" / f"{n}.nii.gz") for n in LOBES}
    res = evaluate_lung_segmentation(ref_img, lobe_imgs)

    vox_ml = res["voxel_volume_ml"]
    vessels = {}
    for n in VESSELS:
        img = _load(seg_dir / "lung_vessels" / f"{n}.nii.gz")
        if img is None:
            vessels[n] = {"status": "missing"}
            res["issues"].append(f"{n}: file missing")
        elif not same_grid(img, ct_img):
            vessels[n] = {"status": "geometry_mismatch"}
            res["issues"].append(f"{n}: not on CT grid")
        else:
            k = int(np.count_nonzero(np.asanyarray(img.dataobj)))
            vessels[n] = {"status": "ok" if k else "empty", "voxels": k, "volume_ml": k * vox_ml}
            if not k:
                res["issues"].append(f"{n}: empty mask")
    res["pred_airways_vessels"] = vessels

    # QC overlay of predicted lobes (label value = lobe index + 1)
    combined = np.zeros(ct_img.shape[:3], dtype=np.uint8)
    for i, n in enumerate(LOBES):
        img = lobe_imgs[n]
        if img is not None and same_grid(img, ct_img):
            combined[np.asanyarray(img.dataobj) > 0] = i + 1
    # one colour per lobe; axial/coronal at the volume centre, sagittal through each lung
    lobe_mask = nib.Nifti1Image(combined, ct_img.affine)
    colors = {i + 1: (LOBE_COLORS[n], n.replace("lung_", "").replace("_", " "))
              for i, n in enumerate(LOBES)}
    imgs = generate_qc_views(ct_img, f"{cid}_pred_lobes", qc_dir, mask_img=lobe_mask,
                             window_name="lung", mask_alpha=0.35, label_colors=colors,
                             views=("axial", "coronal"))
    canon = np.asanyarray(nib.as_closest_canonical(lobe_mask).dataobj)
    for side, labs in (("right_lung", RIGHT_LABELS), ("left_lung", LEFT_LABELS)):
        idx = np.argwhere(np.isin(canon, labs))
        if len(idx):
            centre = tuple(int(v) for v in idx.mean(0).round())
            imgs += generate_qc_views(ct_img, f"{cid}_pred_lobes_{side}", qc_dir,
                                      mask_img=lobe_mask, window_name="lung",
                                      mask_alpha=0.35, label_colors=colors,
                                      index_ras=centre, views=("sagittal",))
    res["qc_images"] = [str(p.relative_to(ROOT)) for p in imgs]

    res.update({"case_id": cid, "source": "TotalSegmentator prediction vs expert reference",
                "status": "warn" if res["issues"] else "ok"})
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seg-run", required=True, type=Path)
    args = ap.parse_args()
    seg = json.loads((ROOT / args.seg_run / "run_manifest.json").read_text())

    run_id = f"P2_eval_{len(seg['cases'])}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    qc_dir = run_dir / "qc"
    qc_dir.mkdir(parents=True)
    manifest = {"run_id": run_id, "seg_run": seg["run_id"],
                "seg_provenance": seg["provenance"], "skipped_in_seg": seg["skipped"],
                "started_utc": datetime.now(timezone.utc).isoformat(), "cases": []}

    for rec in seg["cases"]:
        cid = rec["case_id"]
        print(f"[{cid}] ...", end=" ", flush=True)
        if rec["status"] == "failed":
            res = {"case_id": cid, "status": "failed",
                   "issues": ["segmentation failed: " + "; ".join(rec["issues"])]}
        else:
            try:
                res = evaluate_case(rec, qc_dir)
            except Exception as e:
                res = {"case_id": cid, "status": "failed",
                       "issues": [f"{type(e).__name__}: {e}"],
                       "traceback": traceback.format_exc()}
        manifest["cases"].append(res)
        L, R = res.get("lung_left", {}), res.get("lung_right", {})
        print(res["status"],
              f"Dice L={L.get('dice', float('nan')):.4f} R={R.get('dice', float('nan')):.4f}",
              "; ".join(res["issues"]))

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    cols = ["case_id", "status", "dice_left", "dice_right", "dice_total",
            "pred_left_ml", "ref_left_ml", "err_left_pct", "pred_right_ml", "ref_right_ml",
            "err_right_pct", "pred_laterality_verified", "airways_ml", "arteries_ml",
            "veins_ml", "issues"]
    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in manifest["cases"]:
            L, R, T = r.get("lung_left", {}), r.get("lung_right", {}), r.get("lung_total", {})
            v = r.get("pred_airways_vessels", {})
            w.writerow({
                "case_id": r["case_id"], "status": r["status"],
                "dice_left": L.get("dice"), "dice_right": R.get("dice"),
                "dice_total": T.get("dice"),
                "pred_left_ml": L.get("pred_volume_ml"), "ref_left_ml": L.get("ref_volume_ml"),
                "err_left_pct": L.get("volume_error_pct"),
                "pred_right_ml": R.get("pred_volume_ml"), "ref_right_ml": R.get("ref_volume_ml"),
                "err_right_pct": R.get("volume_error_pct"),
                "pred_laterality_verified": r.get("pred_laterality", {}).get("verified"),
                "airways_ml": v.get("lung_airways", {}).get("volume_ml"),
                "arteries_ml": v.get("lung_arteries", {}).get("volume_ml"),
                "veins_ml": v.get("lung_veins", {}).get("volume_ml"),
                "issues": " | ".join(r["issues"]),
            })
    failed = sum(r["status"] == "failed" for r in manifest["cases"])
    print(f"{run_id}: {len(manifest['cases']) - failed} evaluated, {failed} failed -> {run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
