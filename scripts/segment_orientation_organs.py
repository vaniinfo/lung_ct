"""Organ masks for the left / right check of scans stored outside the repository (segmentation env).

For every HU-valid scan of a QC run, TotalSegmentator 'total' is run with its FAST setting (3 mm) for the organs that
scripts/anatomy_contrast_check.py needs: five lung lobes, trachea, heart, liver, aorta. The masks are used for organ
positions (orientation) and the aorta density only: never for a volume, a measurement or a checklist answer.

Writes <seg-root>/<scan>/total/<organ>.nii.gz and <seg-root>/segment_records.jsonl (one line per scan; resumable).
Default seg-root: <data_root of the QC run>/derived.

Usage:
  environments/segmentation/Scripts/python.exe scripts/segment_orientation_organs.py --qc-run runs/P1_qc_ctrate_calib400_<time>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # Windows nnU-Net workers: see scripts/segment_totalseg.py
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOTALSEG_HOME_DIR", str(ROOT / "models" / "totalsegmentator"))

ORGANS = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
          "trachea", "heart", "liver", "aorta"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qc-run", required=True)
    ap.add_argument("--seg-root", default=None)
    ap.add_argument("--n", type=int, default=None, help="first N eligible scans only")
    args = ap.parse_args()
    qc = json.loads((ROOT / args.qc_run / "run_manifest.json").read_text(encoding="utf-8"))
    seg_root = Path(args.seg_root or (Path(qc["data_root"]) / "derived"))
    seg_root.mkdir(parents=True, exist_ok=True)
    rec_path = seg_root / "segment_records.jsonl"
    done = set()
    if rec_path.exists():
        for line in rec_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["status"] == "ok":
                done.add(r["case_id"])
    todo = [c for c in qc["cases"] if c["status"] != "failed" and c["ct"]["hu_valid"]]
    skipped = [c["case_id"] for c in qc["cases"] if c not in todo]
    todo = todo[:args.n] if args.n else todo
    print(f"{len(todo)} scans to segment ({len(done & {c['case_id'] for c in todo})} already done), {len(skipped)} skipped (failed or not HU)", flush=True)
    import torch
    from totalsegmentator.python_api import totalsegmentator
    bad = 0
    for i, c in enumerate(todo):
        cid, out = c["case_id"], seg_root / c["case_id"] / "total"
        if cid in done and all((out / f"{o}.nii.gz").exists() for o in ORGANS):
            continue
        rec = {"case_id": cid, "status": "ok", "settings": {"task": "total", "fast": True, "roi_subset": ORGANS, "device": "gpu"}}
        t0 = time.time()
        try:
            if out.exists():                                 # never mix outputs of an interrupted run
                for f in out.glob("*.nii.gz"):
                    f.unlink()
            totalsegmentator(Path(c["ct"]["file"]), out, quiet=True, task="total", fast=True, roi_subset=ORGANS, device="gpu")
            missing = [o for o in ORGANS if not (out / f"{o}.nii.gz").exists()]
            if missing:
                rec.update(status="failed", error=f"missing outputs {missing}")
        except Exception as e:  # recorded, never dropped
            rec.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
            torch.cuda.empty_cache()
        rec["seconds"] = round(time.time() - t0, 1)
        bad += rec["status"] != "ok"
        with open(rec_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i + 1}/{len(todo)} {cid}] {rec['status']} {rec['seconds']} s {rec.get('error', '')}", flush=True)
    print(f"done: {len(todo) - bad} ok, {bad} failed -> {seg_root}", flush=True)
    return 0                                                 # failures are in the records; the anatomy check reports them per scan


if __name__ == "__main__":
    sys.exit(main())
