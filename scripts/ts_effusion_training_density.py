"""Density inside the pleural-effusion labels of the published TotalSegmentator training set, for a sample (core env).

Reads single training images out of the Zenodo archive by HTTP byte range, keeps them in memory only, and measures
the HU inside the pleural label (whole label and the part at least 2 mm from its edge). Nothing is stored but the
table; nothing is uploaded. Sample (fixed rule, no result seen first): the N training cases whose pleural label is
thinnest (largest inscribed sphere), plus every K-th of the remaining labelled cases in file-name order.

Usage: python -I scripts/ts_effusion_training_density.py --attrs runs/E16_.../training_labels/training_label_attributes.csv --out runs/E16_.../training_labels
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ts_effusion_training_labels import DEST, URL, RangeFile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--attrs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinnest", type=int, default=20)
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(ROOT / args.attrs, encoding="utf-8")) if r.get("pleural_largest_inscribed_sphere_diameter_mm")]
    rows.sort(key=lambda r: (float(r["pleural_largest_inscribed_sphere_diameter_mm"]), r["case"]))
    thin = rows[:args.thinnest]
    rest = sorted(rows[args.thinnest:], key=lambda r: r["case"])[::args.every]
    sel = [(r, "thinnest") for r in thin] + [(r, "systematic_sample") for r in rest]
    rf = RangeFile(URL)
    rf.seek(max(0, rf.size - (1 << 22)))
    rf.read(rf.size - rf.pos)
    zf = zipfile.ZipFile(rf)
    out_rows = []
    for k, (r, why) in enumerate(sel, 1):
        info = zf.getinfo(f"imagesTr/{r['case']}_0000.nii.gz")
        rf.block = info.compress_size + 8192
        img = nib.Nifti1Image.from_bytes(gzip.decompress(zf.read(info)))
        lab = nib.load(DEST / f"{r['case']}.nii.gz")
        assert img.shape == lab.shape and np.allclose(img.affine, lab.affine, atol=1e-3), r["case"]
        a = np.asarray(img.dataobj).astype(np.float32)
        pe = np.asarray(lab.dataobj) == 2
        z = [float(v) for v in lab.header.get_zooms()[:3]]
        inner = ndi.distance_transform_edt(np.pad(pe, 1), sampling=z)[1:-1, 1:-1, 1:-1] >= 2.0
        v, c = a[pe], a[inner]

        def frac(x, lo, hi):
            return round(float(((x >= lo) & (x < hi)).mean()), 3) if x.size else None
        o = {"case": r["case"], "selected_as": why, "pleural_ml": r["pleural_ml"], "largest_inscribed_sphere_diameter_mm": r["pleural_largest_inscribed_sphere_diameter_mm"],
             "label_mean_hu": round(float(v.mean()), 0), "label_median_hu": round(float(np.median(v)), 0), "label_fluid_range_fraction": frac(v, -30, 40),
             "label_fat_range_fraction": frac(v, -150, -30), "label_below_-150_fraction": frac(v, -5000, -150),
             "inner_part_fraction": round(float(inner.sum()) / float(pe.sum()), 3), "inner_mean_hu": round(float(c.mean()), 0) if c.size else None,
             "inner_fluid_range_fraction": frac(c, -30, 40), "inner_fat_range_fraction": frac(c, -150, -30),
             "image_min_hu": float(a.min()), "image_max_hu": float(a.max())}
        out_rows.append(o)
        print(f"[{k}/{len(sel)}] {json.dumps(o)}", flush=True)
        del a, img
    out = ROOT / args.out
    with open(out / "training_label_density_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print("done", len(out_rows), f"{rf.fetched / 1e6:.0f} MB read, images not stored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
