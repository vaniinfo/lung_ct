"""Fetch the CT scans needed for the PleThora pleural-effusion check and convert them to NIfTI (core env).

PleThora (Kiser et al. 2020, TCIA, CC BY 3.0) = expert pleural-effusion outlines for 78 scans of the public
NSCLC-Radiomics collection (CC BY-NC 3.0), plus a spreadsheet that codes effusion present / absent for 402 scans.
This script downloads ONLY public data and sends nothing but public series identifiers to TCIA.

Selection (fixed before any result was seen):
  - every scan with Effusion.Event = 1 (78)
  - a random sample of scans with Effusion.Event = 0 (seed 20261123, default 78)

Per scan: download the DICOM series, read it with SimpleITK (GDCM, rescale slope/intercept applied -> HU),
write data/raw/plethora/volumes/<PatientID>.nii.gz, record geometry and hashes, delete the DICOM copy.

Usage: python -I scripts/plethora_fetch.py [--negatives 78] [--only LUNG1-001 ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "raw" / "plethora"
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
SEED = 20261123
UA = {"User-Agent": "research-client"}


def num(s: str) -> float:
    return float(s.replace(",", "")) if s.strip() else 0.0


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def get(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"   retry {i + 1}: {e}", flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError("unreachable")


def selection(n_neg: int) -> list[dict]:
    rows = list(csv.DictReader(open(P / "download" / "Thoracic-and-Pleural-Effusion-Segmentations-April-2020.csv", encoding="utf-8-sig")))
    pos = [r for r in rows if r["Effusion.Event"] == "1"]
    neg = sorted((r for r in rows if r["Effusion.Event"] == "0"), key=lambda r: r["PatientID"])
    neg = sorted(random.Random(SEED).sample(neg, n_neg), key=lambda r: r["PatientID"])
    out = []
    for r, g in [(r, "effusion") for r in pos] + [(r, "no_effusion") for r in neg]:
        out.append({"patient_id": r["PatientID"], "group": g, "reference_right_ml": num(r["Right.Pleural.Effusion.Volume"]),
                    "reference_left_ml": num(r["Left.Pleural.Effusion.Volume"]), "csv_dim": [int(r["Dim.x"]), int(r["Dim.y"]), int(r["Dim.z"])]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--negatives", type=int, default=78)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    import SimpleITK as sitk
    sel = selection(args.negatives)
    (P / "selection.json").write_text(json.dumps({"seed": SEED, "rule": __doc__.split("Selection")[1].split("Per scan")[0].strip(), "cases": sel}, indent=1), encoding="utf-8")
    if args.only:
        sel = [s for s in sel if s["patient_id"] in args.only]
    series = json.loads(get(f"{API}/getSeries?Collection=NSCLC-Radiomics&Modality=CT&format=json"))
    by_pat: dict[str, list[dict]] = {}
    for s in series:
        by_pat.setdefault(s["PatientID"], []).append(s)
    vol, tmp = P / "volumes", P / "dicom_tmp"
    vol.mkdir(exist_ok=True)
    log_path = P / "fetch_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    for k, s in enumerate(sel, 1):
        pid = s["patient_id"]
        out = vol / f"{pid}.nii.gz"
        if out.exists() and log.get(pid, {}).get("status") == "ok":
            continue
        rec = {"patient_id": pid, "group": s["group"], "status": "ok", "issues": [], "fetched_utc": datetime.now(timezone.utc).isoformat()}
        try:
            cands = by_pat.get(pid, [])
            match = [c for c in cands if int(c.get("ImageCount", 0)) == s["csv_dim"][2]] or cands
            if len(match) != 1:
                rec["issues"].append(f"{len(cands)} CT series, {len(match)} match the slice count in the PleThora spreadsheet")
            if not match:
                raise RuntimeError("no CT series for this patient")
            uid = match[0]["SeriesInstanceUID"]
            d = tmp / pid
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            blob = get(f"{API}/getImage?SeriesInstanceUID={uid}")
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for m in z.infolist():                       # flat extraction; ignore any path inside the archive
                    if not m.is_dir():
                        (d / Path(m.filename).name).write_bytes(z.read(m))
            rd = sitk.ImageSeriesReader()
            ids = rd.GetGDCMSeriesIDs(str(d))
            if len(ids) != 1:
                rec["issues"].append(f"{len(ids)} series ids in the download")
            rd.SetFileNames(rd.GetGDCMSeriesFileNames(str(d), uid if uid in ids else ids[0]))
            img = rd.Execute()
            sitk.WriteImage(img, str(out), useCompression=True)
            a = sitk.GetArrayViewFromImage(img)
            rec.update(series_uid=uid, dicom_zip_bytes=len(blob), dicom_zip_sha256=hashlib.sha256(blob).hexdigest(), n_dicom_files=len(rd.GetFileNames()),
                       size=list(img.GetSize()), spacing_mm=[round(x, 5) for x in img.GetSpacing()], origin_lps=[round(x, 3) for x in img.GetOrigin()],
                       direction_lps=[round(x, 6) for x in img.GetDirection()], hu_min=int(a.min()), hu_max=int(a.max()),
                       nifti=str(out.relative_to(ROOT).as_posix()), nifti_bytes=out.stat().st_size, nifti_sha256=sha(out),
                       manufacturer=match[0].get("Manufacturer"), model=match[0].get("ManufacturerModelName"))
            if list(img.GetSize()) != s["csv_dim"]:
                rec["issues"].append(f"size {list(img.GetSize())} differs from the PleThora spreadsheet {s['csv_dim']}")
            shutil.rmtree(d)
        except Exception as e:  # noqa: BLE001
            rec.update(status="failed", error=f"{type(e).__name__}: {e}")
        log[pid] = rec
        log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
        print(f"[{k}/{len(sel)}] {pid} {s['group']}: {rec['status']} {rec.get('size', '')} {rec.get('issues') or ''} {rec.get('error', '')}", flush=True)
    bad = [p for p, r in log.items() if r["status"] != "ok"]
    print(f"done: {sum(r['status'] == 'ok' for r in log.values())} ok, {len(bad)} failed {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
