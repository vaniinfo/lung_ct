"""Source-CT audit for selected PleThora scans (core env): DICOM headers and DICOM -> NIfTI fidelity.

Downloads the public DICOM series again from TCIA (only the public series identifier is sent), checks the archive hash
against fetch_log.json, records the acquisition headers, rebuilds HU from the stored pixel values and compares every
voxel with data/raw/plethora/volumes/<PatientID>.nii.gz. The DICOM copy is deleted afterwards; only headers are kept.

Usage: python -I scripts/plethora_source_audit.py --out runs/E16_.../source_audit.json LUNG1-013 ...
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "raw" / "plethora"
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
TAGS = ("Manufacturer", "ManufacturerModelName", "SoftwareVersions", "Modality", "ImageType", "StudyDate", "PatientPosition", "SliceThickness", "PixelSpacing",
        "KVP", "XRayTubeCurrent", "Exposure", "ConvolutionKernel", "ReconstructionDiameter", "FilterType", "ContrastBolusAgent", "ContrastBolusRoute",
        "RescaleSlope", "RescaleIntercept", "RescaleType", "BitsStored", "PixelRepresentation", "PixelPaddingValue", "ImageOrientationPatient",
        "StudyDescription", "SeriesDescription", "BodyPartExamined", "GantryDetectorTilt", "PatientSex")


def get(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client"}), timeout=600) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"   retry {i + 1}: {e}", flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    log = json.loads((P / "fetch_log.json").read_text(encoding="utf-8"))
    res = {}
    for pid in args.cases:
        rec = {"patient_id": pid, "issues": []}
        d = P / "dicom_audit" / pid
        try:
            blob = get(f"{API}/getImage?SeriesInstanceUID={log[pid]['series_uid']}")
            rec["archive_sha256_matches_fetch_log"] = hashlib.sha256(blob).hexdigest() == log[pid]["dicom_zip_sha256"]
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            zipfile.ZipFile(io.BytesIO(blob)).extractall(d)
            ds = [pydicom.dcmread(f) for f in sorted(d.rglob("*.dcm"))]
            ds.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            h = ds[0]
            rec["headers"] = {t: (str(h.get(t)) if h.get(t) is not None else None) for t in TAGS}
            z = np.array([float(x.ImagePositionPatient[2]) for x in ds])
            dz = np.diff(z)
            rec["n_slices"] = len(ds)
            rec["slice_step_mm"] = {"min": round(float(dz.min()), 3), "max": round(float(dz.max()), 3)}
            rec["same_headers_on_all_slices"] = {t: len({str(x.get(t)) for x in ds}) == 1 for t in ("RescaleSlope", "RescaleIntercept", "PixelSpacing", "ImageOrientationPatient", "ConvolutionKernel", "KVP")}
            hu = np.stack([x.pixel_array.astype(np.float32) * float(x.RescaleSlope) + float(x.RescaleIntercept) for x in ds], -1)  # (row, col, slice)
            img = nib.load(P / "volumes" / f"{pid}.nii.gz")
            a = np.asarray(nib.as_closest_canonical(img).dataobj).astype(np.float32)   # R A S
            # DICOM rows run anterior -> posterior, columns right -> left for an axial HFS series (checked against the header below)
            iop = [round(float(v)) for v in h.ImageOrientationPatient]
            rec["image_orientation_patient"] = iop
            if iop != [1, 0, 0, 0, 1, 0]:
                rec["issues"].append(f"unexpected ImageOrientationPatient {iop}; voxel comparison not done")
            else:
                dic_ras = hu.transpose(1, 0, 2)[::-1, ::-1, :]   # (col=L, row=P, slice=S) -> flip to R, A
                same = dic_ras.shape == a.shape
                rec["nifti_vs_dicom"] = {"same_shape": bool(same), "max_abs_difference_hu": float(np.abs(dic_ras - a).max()) if same else None,
                                         "voxels_different": int((dic_ras != a).sum()) if same else None}
            rec["hu"] = {"min": float(hu.min()), "max": float(hu.max()), "air_mode_outside_body": float(np.bincount((hu[:20, :20, :].astype(int) + 2000).ravel()).argmax() - 2000),
                         "fraction_at_minimum": round(float((hu == hu.min()).mean()), 3)}
            sp = [float(v) for v in h.PixelSpacing]
            rec["spacing_matches_nifti"] = bool(np.allclose(sorted(img.header.get_zooms()[:3]), sorted([sp[0], sp[1], float(np.median(dz))]), atol=1e-3))
        except Exception as e:  # noqa: BLE001  reported, never dropped
            rec["issues"].append(f"{type(e).__name__}: {e}")
        finally:
            if d.exists():
                shutil.rmtree(d)
        res[pid] = rec
        print(pid, json.dumps({k: v for k, v in rec.items() if k != "headers"}), flush=True)
        print("   ", {k: v for k, v in rec.get("headers", {}).items() if v not in (None, "None")}, flush=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
