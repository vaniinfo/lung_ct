"""Read the label files of the published TotalSegmentator effusion training set without downloading the 9 GB archive
(core env), and describe the pleural-effusion outlines the model was trained on.

Source: Zenodo record 20272295, Dataset315_pleural_pericard_effusion.zip (CC BY 4.0). The archive is read with HTTP
byte-range requests: the file list first, then only dataset.json and the label files. Nothing is uploaded.
The labels go to data/raw/ts_effusion_training_labels/ (own folder; they are data, never executed).

Per training case: grid, spacing, pleural-effusion volume, largest sphere that fits inside the outline, slices.
Purpose: (1) do the labels contain thin layers like the ones the model misses on PleThora? (2) does any training
case have the grid of the NSCLC-Radiomics scans (overlap screening)?

Usage: python -I scripts/ts_effusion_training_labels.py --out runs/E16_.../training_labels
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
URL = "https://zenodo.org/api/records/20272295/files/Dataset315_pleural_pericard_effusion.zip/content"
DEST = ROOT / "data" / "raw" / "ts_effusion_training_labels"


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, fetched in byte ranges with a small read-ahead cache."""

    def __init__(self, url: str, block: int = 1 << 22):
        self.url, self.block, self.pos, self.cache, self.fetched = url, block, 0, (0, b""), 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": "research-client"}), timeout=120) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, a: int, b: int) -> bytes:
        for i in range(5):
            try:
                rq = urllib.request.Request(self.url, headers={"User-Agent": "research-client", "Range": f"bytes={a}-{b - 1}"})
                with urllib.request.urlopen(rq, timeout=300) as r:
                    d = r.read()
                self.fetched += len(d)
                return d
            except Exception as e:  # noqa: BLE001
                if i == 4:
                    raise
                print(f"   retry {i + 1}: {e}", flush=True)
                time.sleep(5 * (i + 1))
        raise RuntimeError("unreachable")

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        c0, c = self.cache
        if not (c0 <= self.pos and self.pos + n <= c0 + len(c)):
            ln = max(n, self.block)
            c0 = self.pos
            c = self._get(c0, min(self.size, c0 + ln))
            self.cache = (c0, c)
        o = self.pos - c0
        self.pos += n
        return c[o:o + n]

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    rf = RangeFile(URL)
    rf.seek(max(0, rf.size - (1 << 22)))   # central directory sits at the end
    rf.read(rf.size - rf.pos)
    zf = zipfile.ZipFile(rf)
    infos = zf.infolist()
    tops = sorted({"/".join(i.filename.split("/")[:2]) for i in infos})
    labels = [i for i in infos if "labelsTr/" in i.filename and i.filename.endswith(".nii.gz")]
    images = [i for i in infos if "imagesTr/" in i.filename and i.filename.endswith(".nii.gz")]
    meta = {"url": URL, "archive_bytes": rf.size, "entries": len(infos), "folders": tops, "label_files": len(labels), "image_files": len(images),
            "label_bytes_compressed": sum(i.compress_size for i in labels), "image_bytes_compressed": sum(i.compress_size for i in images)}
    print(json.dumps(meta, indent=1), flush=True)
    for i in infos:
        if i.filename.endswith(".json") and i.file_size < 5_000_000:
            (DEST / Path(i.filename).name).write_bytes(zf.read(i))
    rows = []
    for k, i in enumerate(sorted(labels, key=lambda x: x.filename), 1):
        dst = DEST / Path(i.filename).name
        if not dst.exists():
            rf.block = max(1 << 16, i.compress_size + 4096)
            dst.write_bytes(zf.read(i))
        img = nib.load(dst)
        a = np.asarray(img.dataobj)
        zooms = [float(z) for z in img.header.get_zooms()[:3]]
        vml = float(np.prod(zooms)) / 1000.0
        pe = a == 2
        r = {"case": dst.name[:-7], "shape": "x".join(map(str, a.shape)), "spacing_mm": "x".join(f"{z:.3f}" for z in zooms), "axcodes": "".join(nib.aff2axcodes(img.affine)),
             "labels_present": " ".join(str(int(v)) for v in np.unique(a)), "lung_ml": round(float((a == 1).sum()) * vml, 0),
             "pleural_ml": round(float(pe.sum()) * vml, 1), "pericardial_ml": round(float((a == 3).sum()) * vml, 1)}
        if pe.any():
            sl = ndi.find_objects(pe.astype(np.uint8))[0]
            sub = np.pad(pe[sl], 1)
            r["pleural_largest_inscribed_sphere_diameter_mm"] = round(2 * float(ndi.distance_transform_edt(sub, sampling=zooms).max()), 1)
            zax = int(np.argmax(zooms)) if max(zooms) > 1.5 * min(zooms) else 2
            r["pleural_extent_along_thickest_axis_mm"] = round(float(np.any(pe, axis=tuple(x for x in range(3) if x != zax)).sum()) * zooms[zax], 0)
        rows.append(r)
        if k % 50 == 0:
            print(f"[{k}/{len(labels)}] fetched {rf.fetched / 1e6:.0f} MB", flush=True)
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(out / "training_label_attributes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    meta["bytes_fetched"] = rf.fetched
    (out / "training_archive_listing.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("done", len(rows), "label files;", f"{rf.fetched / 1e6:.0f} MB fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
