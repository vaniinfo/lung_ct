"""Phase 1+2: Process first COVID CT case - ingest, QC, volumes, infection burden."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lung_ct.ingest.nifti_loader import load_nifti, check_geometry_match
from lung_ct.qc.qc_views import generate_qc_views
from lung_ct.metrics.volumes import (
    compute_lung_volumes,
    compute_infection_burden,
    voxel_volume_from_affine,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "covid_ct"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

CASE_ID = "coronacases_001"


def main():
    start_time = time.time()

    ct_path = DATA_DIR / "COVID-19-CT-Seg_20cases" / f"{CASE_ID}.nii.gz"
    lung_path = DATA_DIR / "Lung_Mask" / f"{CASE_ID}.nii.gz"
    inf_path = DATA_DIR / "Infection_Mask" / f"{CASE_ID}.nii.gz"

    print(f"=== Processing {CASE_ID} ===")

    # Load CT
    print("\n[1] Loading CT volume...")
    ct_img, ct_meta = load_nifti(ct_path)
    print(f"  Shape: {ct_meta.shape}")
    print(f"  Spacing: {ct_meta.spacing_mm} mm")
    print(f"  Origin: {ct_meta.origin}")
    print(f"  Orientation: {ct_meta.orientation_code}")
    print(f"  Voxel volume: {ct_meta.voxel_volume_mm3:.4f} mm^3")
    print(f"  Intensity range: [{ct_meta.intensity_min:.1f}, {ct_meta.intensity_max:.1f}]")
    print(f"  HU valid: {ct_meta.hu_valid} ({ct_meta.hu_reason})")
    print(f"  SHA-256: {ct_meta.sha256[:16]}...")

    # Load lung mask
    print("\n[2] Loading lung mask...")
    lung_img, lung_meta = load_nifti(lung_path)
    print(f"  Shape: {lung_meta.shape}")
    print(f"  Unique values: {sorted(set(lung_img.get_fdata().ravel().astype(int)))}")

    # Load infection mask
    print("\n[3] Loading infection mask...")
    inf_img, inf_meta = load_nifti(inf_path)
    print(f"  Shape: {inf_meta.shape}")
    print(f"  Unique values: {sorted(set(inf_img.get_fdata().ravel().astype(int)))}")

    # Geometry checks
    print("\n[4] Geometry validation...")
    ct_lung_match = check_geometry_match(ct_meta, lung_meta)
    ct_inf_match = check_geometry_match(ct_meta, inf_meta)
    print(f"  CT-Lung match: {ct_lung_match['match']}")
    if not ct_lung_match["match"]:
        for issue in ct_lung_match["issues"]:
            print(f"    ISSUE: {issue}")
    print(f"  CT-Infection match: {ct_inf_match['match']}")
    if not ct_inf_match["match"]:
        for issue in ct_inf_match["issues"]:
            print(f"    ISSUE: {issue}")

    # Create run directory
    run_id = f"E0_{CASE_ID}_{int(time.time())}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # QC views
    print("\n[5] Generating QC views...")
    qc_dir = run_dir / "qc"

    # Lung window with lung mask overlay
    lung_views = generate_qc_views(ct_img, CASE_ID, qc_dir, mask_img=lung_img,
                                    window_name="lung", mask_color=(0, 1, 0))
    print(f"  Lung window views: {len(lung_views)} PNGs")

    # Lung window with infection mask overlay
    inf_views = generate_qc_views(ct_img, f"{CASE_ID}_infection", qc_dir,
                                   mask_img=inf_img, window_name="lung",
                                   mask_color=(1, 0, 0))
    print(f"  Infection overlay views: {len(inf_views)} PNGs")

    # Mediastinum window (no overlay)
    med_views = generate_qc_views(ct_img, CASE_ID, qc_dir, window_name="mediastinum")
    print(f"  Mediastinum window views: {len(med_views)} PNGs")

    # Compute volumes
    print("\n[6] Computing lung volumes from reference masks...")
    lung_vols = compute_lung_volumes(lung_img)
    for k, v in lung_vols.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Compute infection burden
    print("\n[7] Computing reference infection burden...")
    infection = compute_infection_burden(inf_img, lung_img)
    for k, v in infection.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    elapsed = time.time() - start_time

    # Save results
    results = {
        "case_id": CASE_ID,
        "run_id": run_id,
        "ct": {
            "file": str(ct_path),
            "sha256": ct_meta.sha256,
            "shape": list(ct_meta.shape),
            "spacing_mm": list(ct_meta.spacing_mm),
            "orientation": ct_meta.orientation_code,
            "voxel_volume_mm3": ct_meta.voxel_volume_mm3,
            "intensity_range": [ct_meta.intensity_min, ct_meta.intensity_max],
            "hu_valid": ct_meta.hu_valid,
            "hu_reason": ct_meta.hu_reason,
        },
        "geometry_checks": {
            "ct_lung_match": ct_lung_match,
            "ct_infection_match": ct_inf_match,
        },
        "reference_lung_volumes": lung_vols,
        "reference_infection_burden": infection,
        "qc_images": [str(p) for p in lung_views + inf_views + med_views],
        "elapsed_seconds": elapsed,
        "stage_status": {
            "ingest": "complete",
            "qc": "complete",
            "reference_volumes": "complete",
            "segmentation": "pending",
            "vlm": "pending",
        },
    }

    # Serialize numpy types
    def convert(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    results_path = run_dir / "run_manifest.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "lung_volumes": lung_vols,
            "infection_burden": infection,
        }, f, indent=2, default=convert)

    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"Run directory: {run_dir}")
    print(f"Results: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
