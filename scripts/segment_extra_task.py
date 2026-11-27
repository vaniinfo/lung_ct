"""Run ONE extra TotalSegmentator task on cases that are already segmented (segmentation env).

Unlike segment_totalseg.py this touches nothing that exists: it only writes
data/derived/<case>/segmentation/<task>/ and seg_manifest_<task>.json next to the original manifest.
Weights are downloaded on first use into models/totalsegmentator (usage statistics off); no image leaves the machine.

Usage (Git Bash):
    environments/segmentation/Scripts/python.exe scripts/segment_extra_task.py --task lung_nodules \
        --cases valid_301_a_2 valid_541_a_2
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
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # Windows nnU-Net workers, see segment_totalseg.py
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOTALSEG_HOME_DIR", str(ROOT / "models" / "totalsegmentator"))
DERIVED = ROOT / "data" / "derived"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--volumes", default="data/raw/ct_rate/volumes", help="folder with <case>.nii.gz (HU-valid, already QC'd)")
    args = ap.parse_args()

    import importlib.metadata as md
    import torch
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator
    expected = sorted(class_map[args.task].values())
    cfg = Path(os.environ["TOTALSEG_HOME_DIR"]) / "config.json"
    if not cfg.exists() or json.loads(cfg.read_text()).get("send_usage_stats") is not False:
        raise SystemExit(f"{cfg}: usage statistics must be switched off before running")

    run_id = f"P2_seg_{args.task}_{len(args.cases)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    man = {"run_id": run_id, "task": args.task, "expected_outputs": expected, "started_utc": datetime.now(timezone.utc).isoformat(),
           "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
           "packages": {p: md.version(p) for p in ("TotalSegmentator", "nnunetv2", "torch", "nibabel", "numpy")},
           "totalseg_home": os.environ["TOTALSEG_HOME_DIR"], "cases": []}
    for cid in args.cases:
        print(f"[{cid}] ...", end=" ", flush=True)
        rec = {"case_id": cid, "status": "ok", "issues": []}
        try:
            ct = ROOT / args.volumes / f"{cid}.nii.gz"
            seg = DERIVED / cid / "segmentation"
            if not (seg / "seg_manifest.json").exists():
                raise FileNotFoundError("case has no base segmentation (run segment_totalseg.py first)")
            out = seg / args.task
            if out.exists():  # never mix outputs of an earlier run of this task
                for f in out.glob("*.nii.gz"):
                    f.unlink()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            totalsegmentator(ct, out, quiet=True, task=args.task, fast=False, device="gpu")
            files = sorted(p.name[:-7] for p in out.glob("*.nii.gz"))
            rec.update(ct_file=str(ct.relative_to(ROOT)), ct_sha256=sha256(ct), output_dir=str(out.relative_to(ROOT)),
                       elapsed_seconds=round(time.time() - t0, 1), peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 2**20),
                       outputs={f: sha256(out / f"{f}.nii.gz") for f in files})
            missing = [n for n in expected if n not in files]
            if missing:
                rec.update(status="failed", issues=[f"missing outputs {missing}"])
            (seg / f"seg_manifest_{args.task}.json").write_text(json.dumps({**rec, "run_id": run_id, "packages": man["packages"]}, indent=2))
        except Exception as e:  # reported, never dropped
            rec.update(status="failed", issues=[f"{type(e).__name__}: {e}"], traceback=traceback.format_exc())
        man["cases"].append(rec)
        print(rec["status"], rec.get("elapsed_seconds"), "s", rec.get("peak_gpu_memory_mb"), "MB", "; ".join(rec["issues"]))
    root = Path(os.environ["TOTALSEG_HOME_DIR"]) / "nnunet" / "results"
    man["weights_sha256"] = {str(c.relative_to(root)): sha256(c) for d in root.glob("Dataset*") for c in d.rglob("*.pth")
                             if args.task.replace("_", "") in d.name.replace("_", "").lower()} or "not matched by folder name"
    man["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_manifest.json").write_text(json.dumps(man, indent=2))
    failed = sum(c["status"] == "failed" for c in man["cases"])
    print(f"done: {len(man['cases']) - failed} ok, {failed} failed -> {run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
