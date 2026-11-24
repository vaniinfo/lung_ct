"""Phase 2 (segmentation env): run TotalSegmentator lung tasks, write masks + provenance.

Runs in environments/segmentation (TotalSegmentator + CUDA torch). Only produces masks
and a per-case seg_manifest.json; volumes/Dice are computed in the core env by
scripts/evaluate_segmentation.py.

Tasks (normal resolution, not --fast):
  total        roi_subset = 5 lung lobes + trachea
  lung_vessels lung_airways, lung_airways_wall, lung_arteries, lung_veins

By default only HU-valid cases from a Phase 1 QC run are segmented; non-HU
(display-scaled) cases are skipped and listed, unless --include-non-hu.

Usage (Git Bash):
    environments/segmentation/Scripts/python.exe scripts/segment_totalseg.py \
        --qc-run runs/P1_qc_20cases_20261107_223515 --n 1
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Windows: each spawned nnU-Net worker loads OpenBLAS with one buffer per CPU thread;
# with many workers this exhausts the commit limit ("OpenBLAS error: Memory allocation
# still failed"). Cap BLAS/OpenMP threads before numpy is imported (inherited by workers).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Weights + config on D:, usage statistics disabled in that config. Must precede import.
os.environ.setdefault("TOTALSEG_HOME_DIR", str(ROOT / "models" / "totalsegmentator"))

DERIVED_DIR = ROOT / "data" / "derived"

EXPECTED = {
    "total": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
              "lung_middle_lobe_right", "lung_lower_lobe_right", "trachea"],
    "lung_vessels": ["lung_airways", "lung_airways_wall", "lung_arteries", "lung_veins"],
}
WEIGHT_DATASETS = {  # model folders these tasks use
    "total": "Dataset291_TotalSegmentator_part1_organs_1559subj",
    "lung_vessels": "Dataset117_lung_airways_arteries_veins_282subj",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_hashes() -> dict:
    root = Path(os.environ["TOTALSEG_HOME_DIR"]) / "nnunet" / "results"
    out = {}
    for task, ds in WEIGHT_DATASETS.items():
        ckpts = sorted((root / ds).rglob("checkpoint_final.pth"))
        out[task] = {str(c.relative_to(root)): sha256(c) for c in ckpts} or "MISSING"
    return out


def check_class_map():
    """Fail explicitly if the installed version lacks the expected classes.

    A task listed with names None (added by --extra-task) takes all of its classes.
    """
    from totalsegmentator.map_to_binary import class_map
    for task in list(EXPECTED):
        if EXPECTED[task] is None:
            EXPECTED[task] = sorted(class_map[task].values())
    for task, names in EXPECTED.items():
        available = set(class_map[task].values())
        missing = [n for n in names if n not in available]
        if missing:
            raise RuntimeError(f"Installed TotalSegmentator task '{task}' lacks {missing}")


def segment_case(case_id: str, ct_path: Path) -> dict:
    import torch
    from totalsegmentator.python_api import totalsegmentator

    seg_dir = DERIVED_DIR / case_id / "segmentation"
    rec = {"case_id": case_id, "ct_file": str(ct_path.relative_to(ROOT)),
           "ct_sha256": sha256(ct_path), "output_dir": str(seg_dir.relative_to(ROOT)),
           "tasks": {}, "status": "ok", "issues": []}

    for task in EXPECTED:
        out = seg_dir / task
        if out.exists():  # never mix outputs of an earlier (possibly interrupted) run
            for f in out.glob("*.nii.gz"):
                f.unlink()
        settings = {"task": task, "fast": False, "device": "gpu",
                    "roi_subset": EXPECTED[task] if task == "total" else None}
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        totalsegmentator(ct_path, out, quiet=True, **settings)
        files = sorted(p.name[:-7] for p in out.glob("*.nii.gz"))
        missing = [n for n in EXPECTED[task] if n not in files]
        rec["tasks"][task] = {
            "settings": settings,
            "elapsed_seconds": round(time.time() - t0, 1),
            "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "outputs": files,
            "missing_outputs": missing,
        }
        if missing:
            rec["status"] = "failed"
            rec["issues"].append(f"{task}: missing outputs {missing}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extra-total", nargs="*", default=[], metavar="CLASS",
                    help="extra classes for the 'total' task roi_subset (e.g. heart liver aorta)")
    ap.add_argument("--extra-task", nargs="*", default=[], metavar="TASK",
                    help="extra TotalSegmentator tasks, all classes (e.g. pleural_pericard_effusion)")
    ap.add_argument("--qc-run", required=True, type=Path,
                    help="Phase 1 run dir with run_manifest.json (source of HU validity)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--n", type=int, help="first N eligible cases")
    g.add_argument("--cases", nargs="+")
    ap.add_argument("--include-non-hu", action="store_true",
                    help="also segment display-scaled cases (out-of-distribution)")
    args = ap.parse_args()

    qc = json.loads((ROOT / args.qc_run / "run_manifest.json").read_text())
    qc_cases = {c["case_id"]: c for c in qc["cases"]}
    order = args.cases if args.cases else sorted(qc_cases)
    eligible, skipped = [], []
    for cid in order:
        c = qc_cases.get(cid)
        if c is None or c["status"] == "failed":
            skipped.append({"case_id": cid, "reason": "absent or failed in Phase 1 QC"})
        elif not c["ct"]["hu_valid"] and not args.include_non_hu:
            skipped.append({"case_id": cid, "reason": f"non-HU: {c['ct']['hu_reason']}"})
        else:
            eligible.append(cid)
    if args.n:
        eligible = eligible[:args.n]

    import importlib.metadata as md
    import torch
    EXPECTED["total"] = EXPECTED["total"] + [c for c in args.extra_total if c not in EXPECTED["total"]]
    for t in args.extra_task:
        EXPECTED.setdefault(t, None)
    check_class_map()

    run_id = f"P2_seg_{len(eligible)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "provenance": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "qc_run": str(args.qc_run),
            "python": platform.python_version(),
            "packages": {p: md.version(p) for p in ("TotalSegmentator", "nnunetv2", "torch",
                                                     "nibabel", "numpy")},
            "gpu": torch.cuda.get_device_name(0),
            "totalseg_home": os.environ["TOTALSEG_HOME_DIR"],
            "weights_sha256": weight_hashes(),
            "expected_outputs": {k: list(v) for k, v in EXPECTED.items()},
        },
        "skipped": skipped,
        "cases": [],
    }
    print(f"{run_id}: {len(eligible)} to segment, {len(skipped)} skipped")

    for cid in eligible:
        print(f"[{cid}] ...", end=" ", flush=True)
        ct_path = ROOT / qc_cases[cid]["ct"]["file"]
        try:
            rec = segment_case(cid, ct_path)
        except Exception as e:
            rec = {"case_id": cid, "status": "failed", "issues": [f"{type(e).__name__}: {e}"],
                   "traceback": traceback.format_exc()}
        manifest["cases"].append(rec)
        seg_dir = DERIVED_DIR / cid / "segmentation"
        seg_dir.mkdir(parents=True, exist_ok=True)
        (seg_dir / "seg_manifest.json").write_text(json.dumps(
            {**rec, "run_id": run_id, "provenance": manifest["provenance"]}, indent=2))
        t = rec.get("tasks", {})
        print(rec["status"], {k: f"{v['elapsed_seconds']}s/{v['peak_gpu_memory_mb']}MB"
                              for k, v in t.items()}, "; ".join(rec["issues"]))

    manifest["provenance"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    failed = sum(c["status"] == "failed" for c in manifest["cases"])
    print(f"done: {len(manifest['cases']) - failed} ok, {failed} failed -> {run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
