"""Exploratory airway calibre and tapering per branch (core env). Measurement stage only: no detection, no diagnosis.

Stage order (kept separate):
  1. airway segmentation   scripts/segment_totalseg.py   -> lung_vessels/lung_airways.nii.gz   (model prediction)
  2. airway tree           lung_ct.airway.tree           -> branches, generation, side, length_mm, tortuosity
  3. THIS SCRIPT           lung_ct.airway.calibre        -> lumen area, equivalent diameter, tapering, lobe, quality flags
  4. group comparison      scripts/airway_group_compare.py (separate; needs labels and patient-level splits)

Our own Python implementation ("lung_ct airway calibre v1"). NOT AirQuant, no MATLAB, no claim of equivalence.
NOT VALIDATED: these are not validated airway diameters. See docs/AIRWAY_BRONCHIECTASIS_MEASURES.md.

The tree stage is run here with its unchanged code and default parameters, so branch ids, generations and lengths are the
same as in a runs/P5_airway_* baseline for the same mask. Nothing in a P5 run is read or written. Compare with:
  n_tree_branches, max_generation and total_centreline_mm in summary.csv of both runs must be equal.

Per case -> runs/P6_airway_measures_<n>cases_<ts>/<case>/:
  airway_branch_measures.csv     one row per tree branch (units in the column names: mm, mm2, mm3, percent)
  airway_case_features.csv       per generation / per lobe medians (long format)
  airway_measures_summary.json   tree summary, quality flags, provenance
Run level: run_manifest.json, summary.csv, case_features.csv, failures.csv

Usage:
    python scripts/airway_measure.py --seg-run runs/P2_seg_10cases_<ts> [--cases coronacases_001]
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from lung_ct.airway import calibre  # noqa: E402
from lung_ct.airway.tree import DEFAULTS as TREE_DEFAULTS, METHOD as TREE_METHOD, analyze_airway_tree  # noqa: E402

MASK = "lung_vessels/lung_airways.nii.gz"
LOBES = {1: ("RUL", "total/lung_upper_lobe_right.nii.gz"), 2: ("RML", "total/lung_middle_lobe_right.nii.gz"),
         3: ("RLL", "total/lung_lower_lobe_right.nii.gz"), 4: ("LUL", "total/lung_upper_lobe_left.nii.gz"),
         5: ("LLL", "total/lung_lower_lobe_left.nii.gz")}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return c + ("+dirty" if d else "")
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def lobe_label_image(seg_dir: Path):
    """Integer lobe labels on the grid of the first lobe mask (1 RUL, 2 RML, 3 RLL, 4 LUL, 5 LLL). None if a mask is missing."""
    imgs = {}
    for k, (_, rel) in LOBES.items():
        p = seg_dir / rel
        if not p.exists():
            return None, f"lobe mask missing: {rel}"
        imgs[k] = nib.load(str(p))
    ref = imgs[1]
    lab = np.zeros(ref.shape, np.uint8)
    for k, im in imgs.items():
        if im.shape != ref.shape or not np.allclose(im.affine, ref.affine, atol=1e-3):
            return None, "lobe masks are not on one grid"
        lab[np.asanyarray(im.dataobj) > 0] = k
    return nib.Nifti1Image(lab, ref.affine), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seg-run", required=True, type=Path)
    ap.add_argument("--cases", nargs="+")
    args = ap.parse_args()
    seg = json.loads((ROOT / args.seg_run / "run_manifest.json").read_text())
    cases = [c for c in seg["cases"] if not args.cases or c["case_id"] in args.cases]
    import importlib.metadata as md
    run_id = f"P6_airway_measures_{len(cases)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    manifest = {"run_id": run_id, "provenance": {
        "started_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(), "python": platform.python_version(),
        "packages": {p: md.version(p) for p in ("numpy", "scipy", "scikit-image", "skan", "networkx", "nibabel")},
        "seg_run": str(args.seg_run), "input_mask": "TotalSegmentator lung_vessels/lung_airways (prediction)",
        "tree_method": TREE_METHOD, "tree_params": TREE_DEFAULTS, "calibre_method": calibre.METHOD, "calibre_params": calibre.DEFAULTS,
        "units": {"lengths and diameters": "mm", "areas": "mm2", "volumes": "mm3", "tapering": "percent per mm (intra), percent (inter)"},
        "status": "mask-derived estimates from a predicted airway mask; not airway-diameter measurements on the CT and not validated; not a bronchiectasis detector",
        "not_computed": ["wall_thickness", "broncho_arterial_ratio (no reliable airway-artery pairing)", "any diagnosis"]}, "cases": []}
    summary_rows, feature_rows, failures = [], [], []
    for c in cases:
        cid = c["case_id"]
        print(f"[{cid}] ...", end=" ", flush=True)
        cdir = run_dir / cid
        cdir.mkdir()
        seg_dir = ROOT / c["output_dir"]
        mpath = seg_dir / MASK
        rec = {"case_id": cid, "mask": str(mpath.relative_to(ROOT)), "status": "ok"}
        try:
            if c["status"] != "ok":
                raise RuntimeError(f"segmentation status {c['status']}")
            rec["mask_sha256"] = sha256_file(mpath)
            img = nib.load(str(mpath))
            t0 = time.time()
            res = analyze_airway_tree(img)                      # unchanged baseline code and defaults
            lobe_img, lobe_issue = lobe_label_image(seg_dir)
            rows = calibre.measure_branches(img, res, lobe_img=lobe_img, lobe_names={k: v[0] for k, v in LOBES.items()})
            lung_img = None if lobe_img is None else nib.Nifti1Image((np.asanyarray(lobe_img.dataobj) > 0).astype(np.uint8), lobe_img.affine)
            q = calibre.case_quality(img, res, lung_img=lung_img)
            if lobe_issue:
                q["flags"].append(f"lobe not assigned ({lobe_issue})")
            feats = calibre.case_features(rows)
            rec.update(elapsed_seconds=round(time.time() - t0, 1), tree_summary=res.summary, quality=q,
                       n_branches_measured=sum(r["area_equivalent_diameter_mm"] is not None for r in rows))
            if rows:
                with open(cdir / "airway_branch_measures.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
            with open(cdir / "airway_case_features.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["feature", "group", "value", "n_branches"])
                w.writeheader()
                w.writerows(feats)
            (cdir / "airway_measures_summary.json").write_text(json.dumps({**rec, "provenance": manifest["provenance"]}, indent=2))
            feature_rows += [{"case_id": cid, **fr} for fr in feats]
            s = res.summary
            flagged = lambda name: sum(name in r["flags"].split(";") for r in rows)      # noqa: E731
            summary_rows.append({"case_id": cid, "n_tree_branches": s["n_tree_branches"], "max_generation": s["max_generation"],
                                 "total_centreline_mm": s["total_centreline_length_mm"], "n_branches_measured": rec["n_branches_measured"],
                                 "n_small_relative_to_voxel": flagged("small_relative_to_voxel"), "n_non_circular_or_merged": flagged("non_circular_or_merged_lumen"),
                                 "n_short_usable_length": flagged("short_usable_length"), "n_not_measured": flagged("not_measured"),
                                 "spacing_mm": "x".join(str(x) for x in q["detail"]["spacing_mm_from_affine"]),
                                 "n_case_flags": len(q["flags"]), "case_flags": " | ".join(q["flags"]), "elapsed_seconds": rec["elapsed_seconds"]})
            print("ok", rec["n_branches_measured"], "of", len(rows), "branches measured;", len(q["flags"]), "case flag(s)", f"{rec['elapsed_seconds']}s")
        except Exception as e:  # noqa: BLE001  recorded, never dropped
            rec.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            failures.append({"case_id": cid, "reason": rec["error"]})
            (cdir / "airway_measures_summary.json").write_text(json.dumps(rec, indent=2))
            print("FAILED", rec["error"])
        manifest["cases"].append(rec)
    manifest["provenance"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    for name, rws, cols in (("summary.csv", summary_rows, list(summary_rows[0]) if summary_rows else ["case_id"]),
                            ("case_features.csv", feature_rows, ["case_id", "feature", "group", "value", "n_branches"]),
                            ("failures.csv", failures, ["case_id", "reason"])):
        with open(run_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rws)
    print(f"done: {len(summary_rows)} ok, {len(failures)} failed -> {run_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
