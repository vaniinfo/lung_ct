"""Fetch the tumour outline (GTV-1) of selected NSCLC-Radiomics scans from the public DICOM SEG series (core env).

Used only by the PleThora failure analysis, to see whether a predicted pleural-effusion mask sits on tumour.
PleThora's reference outlines exclude effusion that overlaps this GTV. Sends only public series identifiers to TCIA.
Writes data/raw/plethora/gtv/<PatientID>_gtv.nii.gz on the grid of data/raw/plethora/volumes/<PatientID>.nii.gz
(frames are placed by their ImagePositionPatient z, never by frame order) and gtv_fetch_log.json.

Usage: python -I scripts/plethora_fetch_gtv.py LUNG1-001 LUNG1-002 ...
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "raw" / "plethora"
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client"}), timeout=600) as r:
        return r.read()


def main() -> int:
    import pydicom
    import SimpleITK as sitk
    (P / "gtv").mkdir(exist_ok=True)
    logp = P / "gtv_fetch_log.json"
    log = json.loads(logp.read_text(encoding="utf-8")) if logp.exists() else {}
    for pid in sys.argv[1:]:
        if log.get(pid, {}).get("status") == "ok":
            continue
        rec = {"status": "ok"}
        try:
            ser = [s for s in json.loads(get(f"{API}/getSeries?Collection=NSCLC-Radiomics&PatientID={pid}&format=json")) if s.get("Modality") == "SEG"]
            if len(ser) != 1:
                raise RuntimeError(f"{len(ser)} SEG series")
            blob = get(f"{API}/getImage?SeriesInstanceUID={ser[0]['SeriesInstanceUID']}")
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".dcm")]
                ds = pydicom.dcmread(io.BytesIO(z.read(names[0])))
            labels = {int(s.SegmentNumber): str(s.SegmentLabel) for s in ds.SegmentSequence}
            desc = {int(s.SegmentNumber): str(getattr(s, "SegmentDescription", "")) for s in ds.SegmentSequence}
            want = [n for n, lab in labels.items() if lab.upper().replace("_", "-") in ("GTV-1", "GTV1", "NEOPLASM, PRIMARY")]
            rec["gtv_segment"] = [{"number": n, "label": labels[n], "description": desc[n]} for n in want]
            if len(want) != 1:
                raise RuntimeError(f"{len(want)} primary-tumour segments; segments: {sorted(labels.values())}")
            ct = sitk.ReadImage(str(P / "volumes" / f"{pid}.nii.gz"))
            nx, ny, nz = ct.GetSize()
            if (ds.Rows, ds.Columns) != (ny, nx):
                raise RuntimeError(f"SEG frame size {(ds.Rows, ds.Columns)} differs from the CT {(ny, nx)}")
            sh = ds.SharedFunctionalGroupsSequence[0]
            iop = [float(v) for v in sh.PlaneOrientationSequence[0].ImageOrientationPatient]
            if not np.allclose(iop, ct.GetDirection()[0:1] + ct.GetDirection()[3:4] + ct.GetDirection()[6:7] + ct.GetDirection()[1:2] + ct.GetDirection()[4:5] + ct.GetDirection()[7:8], atol=1e-3):
                raise RuntimeError(f"SEG orientation {iop} differs from the CT direction {ct.GetDirection()}")
            frames = ds.pixel_array
            frames = frames[None] if frames.ndim == 2 else frames
            out = np.zeros((nz, ny, nx), np.uint8)
            used, off = 0, []
            for i, fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
                if int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber) not in want:
                    continue
                pos = [float(v) for v in fg.PlanePositionSequence[0].ImagePositionPatient]
                idx = ct.TransformPhysicalPointToContinuousIndex(pos)
                off.append(max(abs(idx[0]), abs(idx[1]), abs(idx[2] - round(idx[2]))))
                k = int(round(idx[2]))
                if not 0 <= k < nz:
                    raise RuntimeError(f"frame {i} lies outside the CT (slice index {idx[2]:.2f})")
                out[k] |= (frames[i] > 0).astype(np.uint8)
                used += 1
            if max(off, default=0) > 0.25:
                raise RuntimeError(f"SEG frames are not on the CT grid (largest index offset {max(off):.2f})")
            img = sitk.GetImageFromArray(out)
            img.CopyInformation(ct)
            f = P / "gtv" / f"{pid}_gtv.nii.gz"
            sitk.WriteImage(img, str(f), useCompression=True)
            sp = ct.GetSpacing()
            rec.update(series_uid=ser[0]["SeriesInstanceUID"], seg_zip_sha256=hashlib.sha256(blob).hexdigest(), segments=sorted(labels.values()), gtv_frames=used,
                       largest_index_offset=round(max(off, default=0), 4), gtv_ml=round(float(out.sum()) * sp[0] * sp[1] * sp[2] / 1000, 1), file=f.relative_to(ROOT).as_posix())
        except Exception as e:  # noqa: BLE001
            rec = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        log[pid] = rec
        logp.write_text(json.dumps(log, indent=1), encoding="utf-8")
        print(pid, rec, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
