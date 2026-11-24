"""List the candidates outlined by TotalSegmentator's lung_nodules task (core env). Exploratory: no rule, no answer.

One row per connected outline (26-connectivity): volume (voxel count x voxel volume), equivalent-sphere diameter,
bounding-box extent in mm, lobe (largest overlap, else nearest lobe), centre slice, median HU. A montage of axial
crops is saved per case for visual inspection. The outline is an unvalidated model output: a candidate is not a nodule.

Usage: python scripts/nodule_candidates.py --cases valid_301_a_2 valid_541_a_2 [--grades runs/grades_received]
Writes runs/E11_nodule_candidates_<time>/{candidates.csv,summary.md,<case>.png}
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
LOBES = {"lung_upper_lobe_right": "RUL", "lung_middle_lobe_right": "RML", "lung_lower_lobe_right": "RLL",
         "lung_upper_lobe_left": "LUL", "lung_lower_lobe_left": "LLL"}


def load_on_grid(p: Path, ct) -> np.ndarray:
    img = nib.load(str(p))
    if img.shape != ct.shape or not np.allclose(img.affine, ct.affine, atol=1e-3):
        raise ValueError(f"{p.name} is not on the CT grid")
    return np.asanyarray(img.dataobj)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--grades", default="runs/grades_received", help="folder with reviewer grade CSVs (optional)")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    grades = {}
    for f in glob.glob(str(ROOT / args.grades / "grades_*.csv")):
        for r in csv.DictReader(open(f, newline="", encoding="utf-8-sig")):
            if r["condition_key"] == "lung_nodule":
                grades[r["case_id"]] = (r["doctor_answer"] or "not graded") + (f" ({r['comment']})" if r["comment"] else "")
    out = ROOT / "runs" / f"E11_nodule_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    rows, summ, failed = [], [], []
    for cid in args.cases:
        try:
            ct = nib.load(str(ROOT / "data/raw/ct_rate/volumes" / f"{cid}.nii.gz"))
            seg = ROOT / "data" / "derived" / cid / "segmentation"
            hu = np.asanyarray(ct.dataobj).astype(np.float32)
            nod = load_on_grid(seg / "lung_nodules" / "lung_nodules.nii.gz", ct) > 0
            zooms = np.array(ct.header.get_zooms()[:3], float)
            vox_mm3 = float(abs(np.linalg.det(ct.affine[:3, :3])))
            lobes = np.zeros(ct.shape, np.uint8)
            names = list(LOBES)
            for i, n in enumerate(names, 1):
                lobes[load_on_grid(seg / "total" / f"{n}.nii.gz", ct) > 0] = i
            _, idx = ndi.distance_transform_edt(lobes == 0, sampling=zooms, return_indices=True)
            ax = nib.aff2axcodes(ct.affine)
            lab, n = ndi.label(nod, structure=np.ones((3, 3, 3)))
            case_rows = []
            for k, sl in enumerate(ndi.find_objects(lab), 1):
                m = lab[sl] == k
                nv = int(m.sum())
                vol = nv * vox_mm3
                lv = lobes[sl][m]
                inside = np.bincount(lv, minlength=6)
                c = np.array(ndi.center_of_mass(m)) + [s.start for s in sl]
                ci = tuple(int(round(x)) for x in c)
                if inside[1:].sum() > 0:
                    lobe, how = LOBES[names[int(inside[1:].argmax())]], "inside"
                else:
                    j = tuple(int(idx[d][ci]) for d in range(3))
                    lobe, how = LOBES[names[lobes[j] - 1]], "nearest"
                case_rows.append({"case_id": cid, "candidate": k, "voxels": nv, "volume_mm3": round(vol, 1),
                                  "equiv_diameter_mm": round((6 * vol / math.pi) ** (1 / 3), 1),
                                  "extent_mm": "x".join(f"{(s.stop - s.start) * z:.1f}" for s, z in zip(sl, zooms)),
                                  "lobe": lobe, "lobe_how": how, "share_inside_lobes": round(float(inside[1:].sum()) / nv, 2),
                                  "centre_voxel": "/".join(map(str, ci)), "median_hu": round(float(np.median(hu[sl][m])))})
            case_rows.sort(key=lambda r: -r["volume_mm3"])
            for i, r in enumerate(case_rows, 1):
                r["candidate"] = i
            rows += case_rows
            d = [r["equiv_diameter_mm"] for r in case_rows]
            summ.append({"case_id": cid, "n": len(case_rows), "largest_mm": max(d) if d else None,
                         "n_ge_4mm": sum(x >= 4 for x in d), "n_ge_6mm": sum(x >= 6 for x in d),
                         "lobes": ", ".join(sorted({r["lobe"] for r in case_rows})) or "-",
                         "reviewer": grades.get(cid, "not graded"), "report_label": "Lung nodule" in gt[cid]["labels"], "axes": "".join(ax)})
            # montage: axial slice through each of the largest candidates, lung window, outline in red
            zax = [a in "SI" for a in ax].index(True)
            show = case_rows[:8]
            if show:
                fig, axs = plt.subplots(1, len(show), figsize=(3 * len(show), 3.3), squeeze=False)
                for a, r in zip(axs[0], show):
                    ci = [int(x) for x in r["centre_voxel"].split("/")]
                    half = [int(round(30 / z)) for z in zooms]
                    s = [slice(max(0, ci[d] - half[d]), ci[d] + half[d] + 1) for d in range(3)]
                    s[zax] = ci[zax]
                    other = [d for d in range(3) if d != zax]
                    im, mk = hu[tuple(s)], nod[tuple(s)]
                    a.imshow(im.T, cmap="gray", vmin=-1350, vmax=150, origin="lower", aspect=zooms[other[1]] / zooms[other[0]])
                    a.contour(mk.T.astype(float), levels=[0.5], colors="r", linewidths=0.8)
                    a.set_title(f"#{r['candidate']} {r['lobe']} {r['equiv_diameter_mm']} mm, {r['median_hu']} HU", fontsize=8)
                    a.axis("off")
                fig.suptitle(f"{cid}: lung_nodules candidates (unvalidated outline; crop is not display-oriented)", fontsize=9)
                fig.savefig(out / f"{cid}.png", dpi=110, bbox_inches="tight")
                plt.close(fig)
        except Exception as e:  # reported, never dropped
            failed.append({"case_id": cid, "error": f"{type(e).__name__}: {e}"})
    if rows:
        with open(out / "candidates.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    md = ["# TotalSegmentator lung_nodules: candidates per scan (exploratory, unvalidated)", "",
          "Diameter = sphere with the same volume as the outline. Reviewer = the clinician reviewer's grade for 'Lung nodule'. Report label = CT-RATE, text-derived.", "",
          "| scan | candidates | largest mm | >= 4 mm | >= 6 mm | lobes | reviewer | report label |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {s['case_id']} | {s['n']} | {s['largest_mm'] or '-'} | {s['n_ge_4mm']} | {s['n_ge_6mm']} | {s['lobes']} | {s['reviewer']} | {'yes' if s['report_label'] else ''} |"
           for s in summ]
    md += ["", "| scan | # | lobe | diameter mm | volume mm3 | extent mm | median HU | centre voxel |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['case_id']} | {r['candidate']} | {r['lobe']}{'' if r['lobe_how'] == 'inside' else ' (nearest)'} | {r['equiv_diameter_mm']} | {r['volume_mm3']} | {r['extent_mm']} | {r['median_hu']} | {r['centre_voxel']} |"
           for r in rows]
    md += [f"- FAILED {f['case_id']}: {f['error']}" for f in failed]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
