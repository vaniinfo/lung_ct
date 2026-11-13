"""Phase 1 batch ingest + QC over COVID-19-CT-Seg cases (1 -> 5 -> 20 feasibility).

Per case: CT load + HU plausibility, CT/mask geometry match, lung label set and
world-space laterality, reference lung volumes, reference infection burden, QC PNGs.
Every case gets a row in the summary; failures are recorded, never dropped.

Usage:
    python scripts/batch_qc.py --n 5
    python scripts/batch_qc.py --n 20
    python scripts/batch_qc.py --cases coronacases_001 radiopaedia_4_85506_1
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lung_ct.ingest.nifti_loader import check_geometry_match, load_nifti  # noqa: E402
from lung_ct.metrics.volumes import compute_infection_burden, compute_lung_volumes  # noqa: E402
from lung_ct.qc.laterality import check_lung_laterality  # noqa: E402
from lung_ct.qc.qc_views import generate_qc_views  # noqa: E402

DATA_DIR = ROOT / "data" / "raw" / "covid_ct"
CT_DIR = DATA_DIR / "COVID-19-CT-Seg_20cases"
RUNS_DIR = ROOT / "runs"
LABELS = {1: "left_lung", 2: "right_lung"}  # dataset ReadMe convention, verified per case


def provenance() -> dict:
    def _git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except Exception:
            return None

    import matplotlib
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {"nibabel": nib.__version__, "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "dataset": {"source": "https://zenodo.org/records/3757476",
                    "checksums": (DATA_DIR / "checksums_sha256.txt").read_text().splitlines()},
    }


def process_case(case_id: str, qc_dir: Path) -> dict:
    t0 = time.time()
    rec = {"case_id": case_id, "status": "ok", "issues": []}
    ct_path = CT_DIR / f"{case_id}.nii.gz"
    lung_path = DATA_DIR / "Lung_Mask" / f"{case_id}.nii.gz"
    inf_path = DATA_DIR / "Infection_Mask" / f"{case_id}.nii.gz"

    ct_img, ct = load_nifti(ct_path)
    rec["ct"] = {
        "file": str(ct_path.relative_to(ROOT)), "sha256": ct.sha256,
        "shape": list(ct.shape), "spacing_mm": list(ct.spacing_mm),
        "orientation": ct.orientation_code, "voxel_volume_mm3": ct.voxel_volume_mm3,
        "dtype_on_disk": str(ct_img.get_data_dtype()),
        "scl_slope_inter": [float(x) if x is not None and np.isfinite(x) else None
                            for x in ct_img.header.get_slope_inter()],
        "intensity_range": [ct.intensity_min, ct.intensity_max],
        "hu_valid": ct.hu_valid, "hu_reason": ct.hu_reason,
    }
    if not ct.hu_valid:
        rec["issues"].append(f"non-HU intensities: {ct.hu_reason} -> excluded from HU analysis")

    lung_img, lung = load_nifti(lung_path)
    inf_img, inf = load_nifti(inf_path)
    geo = {"ct_lung": check_geometry_match(ct, lung), "ct_infection": check_geometry_match(ct, inf)}
    rec["geometry"] = {k: {"match": v["match"], "issues": v["issues"]} for k, v in geo.items()}
    geometry_ok = all(v["match"] for v in geo.values())
    if not geometry_ok:
        rec["issues"].append("CT/mask geometry mismatch -> volumes and overlays skipped")

    lung_labels = sorted(np.unique(np.asanyarray(lung_img.dataobj)).astype(int).tolist())
    inf_labels = sorted(np.unique(np.asanyarray(inf_img.dataobj)).astype(int).tolist())
    rec["labels"] = {"lung_mask": lung_labels, "infection_mask": inf_labels}
    if lung_labels != [0, 1, 2]:
        rec["issues"].append(f"unexpected lung labels {lung_labels} (expected [0, 1, 2])")
    if inf_labels == [0]:
        rec["issues"].append("infection mask is empty")

    lat = check_lung_laterality(lung_img, left_label=1, right_label=2)
    rec["laterality"] = lat
    if not lat["verified"]:
        rec["issues"].append(f"laterality not verified: {lat['reason']}")

    if geometry_ok:
        vols = compute_lung_volumes(lung_img)
        named = {"total_lung_volume_ml": vols["total_lung_volume_ml"]}
        for lab, name in LABELS.items():
            key = f"{name}_volume_ml" if lat["verified"] else f"label_{lab}_volume_ml"
            named[key] = vols.get(f"label_{lab}_volume_ml", 0.0)
        rec["reference_lung_volumes"] = named
        rec["reference_infection_burden"] = compute_infection_burden(inf_img, lung_img)
        if rec["reference_infection_burden"]["infection_voxels_outside_lung"] > 0:
            rec["issues"].append("infection voxels outside reference lung mask")

        if ct.hu_valid:
            window = "lung"
        elif ct.intensity_min >= 0 and ct.intensity_max <= 255:
            window = "raw8bit_nonHU"
        else:
            window = None
        rec["qc_window"] = window
        if window:
            rec["qc_images"] = [str(p.relative_to(ROOT)) for p in
                                generate_qc_views(ct_img, case_id, qc_dir, mask_img=lung_img,
                                                  window_name=window, mask_color=(0, 0.8, 0))
                                + generate_qc_views(ct_img, f"{case_id}_infection", qc_dir,
                                                    mask_img=inf_img, window_name=window)]
        else:
            rec["qc_images"] = []
            rec["issues"].append("QC PNGs not rendered: no defined window for this intensity scale")

    if rec["issues"]:
        rec["status"] = "warn"
    rec["elapsed_seconds"] = round(time.time() - t0, 2)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--n", type=int, help="first N cases in sorted order")
    g.add_argument("--cases", nargs="+", help="explicit case ids")
    args = ap.parse_args()

    all_cases = sorted(p.name[:-len(".nii.gz")] for p in CT_DIR.glob("*.nii.gz"))
    cases = args.cases if args.cases else all_cases[:args.n]

    run_id = f"P1_qc_{len(cases)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    qc_dir = run_dir / "qc"
    qc_dir.mkdir(parents=True)
    manifest = {"run_id": run_id, "provenance": provenance(), "cases": []}

    for case_id in cases:
        print(f"[{case_id}] ...", end=" ", flush=True)
        try:
            rec = process_case(case_id, qc_dir)
        except Exception as e:  # record, never omit
            rec = {"case_id": case_id, "status": "failed", "issues": [f"{type(e).__name__}: {e}"],
                   "traceback": traceback.format_exc()}
        manifest["cases"].append(rec)
        print(rec["status"], "; ".join(rec["issues"]))

    manifest["provenance"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    cols = ["case_id", "status", "orientation", "shape", "spacing_mm", "intensity_min",
            "intensity_max", "hu_valid", "geometry_ok", "laterality_verified",
            "total_lung_volume_ml", "left_lung_volume_ml", "right_lung_volume_ml",
            "infection_burden_pct", "issues"]
    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in manifest["cases"]:
            ct, v = r.get("ct", {}), r.get("reference_lung_volumes", {})
            w.writerow({
                "case_id": r["case_id"], "status": r["status"],
                "orientation": ct.get("orientation"), "shape": ct.get("shape"),
                "spacing_mm": [round(s, 4) for s in ct.get("spacing_mm", [])],
                "intensity_min": ct.get("intensity_range", [None, None])[0],
                "intensity_max": ct.get("intensity_range", [None, None])[1],
                "hu_valid": ct.get("hu_valid"),
                "geometry_ok": all(g["match"] for g in r.get("geometry", {}).values())
                if "geometry" in r else None,
                "laterality_verified": r.get("laterality", {}).get("verified"),
                "total_lung_volume_ml": v.get("total_lung_volume_ml"),
                "left_lung_volume_ml": v.get("left_lung_volume_ml"),
                "right_lung_volume_ml": v.get("right_lung_volume_ml"),
                "infection_burden_pct": r.get("reference_infection_burden", {})
                .get("infection_burden_pct"),
                "issues": " | ".join(r["issues"]),
            })

    n = {s: sum(r["status"] == s for r in manifest["cases"]) for s in ("ok", "warn", "failed")}
    print(f"\n{run_id}: {n}  ->  {run_dir}")
    return 1 if n["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
