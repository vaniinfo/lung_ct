"""Run the TotalSegmentator pleural_pericard_effusion task on the fetched PleThora scans (segmentation env, GPU).

Same task, same weights, same settings (fast=False) as the lung_ct agent uses for evidence rows E71 / E72 / E73.
Nothing is tuned here. Writes data/derived/plethora/<PatientID>/pleural_pericard_effusion/*.nii.gz and a run manifest.
Resumable: a scan with a finished record is skipped.

Usage (Git Bash): environments/segmentation/Scripts/python.exe scripts/plethora_segment.py
"""
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
TASK = "pleural_pericard_effusion"
P = ROOT / "data" / "raw" / "plethora"
OUT = ROOT / "data" / "derived" / "plethora"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import importlib.metadata as md
    import torch
    from totalsegmentator.python_api import totalsegmentator
    cfg = Path(os.environ["TOTALSEG_HOME_DIR"]) / "config.json"
    if not cfg.exists() or json.loads(cfg.read_text()).get("send_usage_stats") is not False:
        raise SystemExit(f"{cfg}: usage statistics must be switched off before running")
    cases = [c["patient_id"] for c in json.loads((P / "selection.json").read_text(encoding="utf-8"))["cases"]]
    OUT.mkdir(parents=True, exist_ok=True)
    man_path = OUT / "seg_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {
        "task": TASK, "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
        "packages": {p: md.version(p) for p in ("TotalSegmentator", "nnunetv2", "torch", "nibabel", "numpy")}, "settings": {"fast": False, "device": "gpu"}, "cases": {}}
    for k, pid in enumerate(cases, 1):
        if man["cases"].get(pid, {}).get("status") == "ok":
            continue
        rec = {"status": "ok", "issues": []}
        try:
            ct = P / "volumes" / f"{pid}.nii.gz"
            if not ct.exists():
                raise FileNotFoundError("scan was not fetched")
            out = OUT / pid / TASK
            if out.exists():
                for f in out.glob("*.nii.gz"):
                    f.unlink()
            t0 = time.time()
            totalsegmentator(ct, out, quiet=True, task=TASK, fast=False, device="gpu")
            files = sorted(p.name[:-7] for p in out.glob("*.nii.gz"))
            rec.update(ct_sha256=sha256(ct), elapsed_seconds=round(time.time() - t0, 1), outputs={f: sha256(out / f"{f}.nii.gz") for f in files})
            missing = [n for n in ("pleural_effusion", "pericardial_effusion") if n not in files]
            if missing:
                rec.update(status="failed", issues=[f"missing outputs {missing}"])
        except Exception as e:  # reported, never dropped
            rec.update(status="failed", issues=[f"{type(e).__name__}: {e}"], traceback=traceback.format_exc())
        man["cases"][pid] = rec
        man["updated_utc"] = datetime.now(timezone.utc).isoformat()
        man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
        print(f"[{k}/{len(cases)}] {pid} {rec['status']} {rec.get('elapsed_seconds')} s {'; '.join(rec['issues'])}", flush=True)
    root = Path(os.environ["TOTALSEG_HOME_DIR"]) / "nnunet" / "results" / "Dataset315_thoraxCT"
    man["weights_sha256"] = {str(c.relative_to(root)): sha256(c) for c in root.rglob("*.pth")}
    man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    failed = [p for p, r in man["cases"].items() if r["status"] != "ok"]
    print(f"done: {len(man['cases']) - len(failed)} ok, {len(failed)} failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
