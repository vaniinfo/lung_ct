"""Describe the traced PleThora scans and render stage overlays (core env). Reads data/derived/plethora_trace/<id>/.

Per scan: what the reference region consists of (HU bands, whole outline and its inner part), how thick it is, what
the model's probabilities are there, and one overlay (largest reference slice): CT, reference outline, the model's
pleural-effusion probability at 1% / 10% / 50%, the model's own 'lung' class.
Descriptive only; the probability levels are not operating points.

Usage: python scripts/plethora_empty_mask_report.py --out runs/E16_... LUNG1-013 ...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "data" / "derived" / "plethora_trace"
BANDS = (("air_or_lung_below_-300", -5000, -300), ("-300_to_-150", -300, -150), ("fat_range_-150_to_-30", -150, -30),
         ("fluid_range_-30_to_40", -30, 40), ("above_40", 40, 5000))


def bands(v: np.ndarray) -> dict:
    return {n: round(float(((v >= a) & (v < b)).mean()), 3) for n, a, b in BANDS} if v.size else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    rows, full = [], {}
    for pid in args.cases:
        d = T / pid
        tr = json.loads((d / "trace.json").read_text(encoding="utf-8"))
        cti = nib.load(d / "ct_crop.nii.gz")
        assert "".join(nib.aff2axcodes(cti.affine)) == "RAS"
        ct = np.asarray(cti.dataobj).astype(np.float32)
        ref = np.asarray(nib.load(d / "ref_crop.nii.gz").dataobj) > 0
        pp = np.asarray(nib.load(d / "p_pleural_pct.nii.gz").dataobj).astype(np.float32) / 100
        pl = np.asarray(nib.load(d / "p_lung_pct.nii.gz").dataobj).astype(np.float32) / 100
        z = [float(v) for v in cti.header.get_zooms()[:3]]
        # inner part: reference voxels at least 2 mm from the outline edge within the axial slice (partial volume removed)
        inpl = np.stack([ndi.distance_transform_edt(ref[:, :, k], sampling=z[:2]) for k in range(ref.shape[2])], -1)
        core = inpl >= 2.0
        thick = 2 * inpl[ref & (inpl >= ndi.maximum_filter(inpl, size=(3, 3, 1)))]   # in-plane ridge = local thickness
        edt3 = ndi.distance_transform_edt(np.pad(ref, 1), sampling=z)
        a = tr["inference"]["as_run"]
        ins = a["inside_reference"]
        # side of each reference voxel: in RAS the x index increases towards the patient's right
        mid = ref.shape[0] / 2
        xs = np.argwhere(ref)[:, 0]
        r = {"patient_id": pid, "reference_ml": tr["reference_ml"], "tool_ml": a["pleural_ml_in_output_file"],
             "reference_right_fraction": round(float((xs >= mid).mean()), 2),
             "reference_slices": int(ref.any((0, 1)).sum()), "largest_inscribed_sphere_diameter_mm": round(2 * float(edt3.max()), 1),
             "axial_thickness_median_mm": round(float(np.median(thick)), 1), "axial_thickness_p90_mm": round(float(np.percentile(thick, 90)), 1),
             "reference_mean_hu": round(float(ct[ref].mean()), 0), "reference_median_hu": round(float(np.median(ct[ref])), 0),
             "inner_part_fraction_of_reference": round(float(core.sum()) / max(1, int(ref.sum())), 3),
             "inner_part_ml": round(float(core.sum()) * float(np.prod(z)) / 1000, 1),
             "inner_part_mean_hu": round(float(ct[core].mean()), 0) if core.any() else None,
             "inner_part_median_hu": round(float(np.median(ct[core])), 0) if core.any() else None,
             "inner_part_fluid_range_fraction": bands(ct[core]).get("fluid_range_-30_to_40"),
             "inner_part_fat_range_fraction": bands(ct[core]).get("fat_range_-150_to_-30"),
             "reference_fluid_range_fraction": bands(ct[ref])["fluid_range_-30_to_40"], "reference_fat_range_fraction": bands(ct[ref])["fat_range_-150_to_-30"],
             "reference_below_-150_fraction": round(bands(ct[ref])["air_or_lung_below_-300"] + bands(ct[ref])["-300_to_-150"], 3),
             "reference_inside_crop_box": tr["crop"]["reference_fraction_inside_box"], "margin_to_box_mm": tr["crop"]["smallest_margin_reference_to_box_face_mm"],
             "p_pleural_max_in_reference": ins["p_pleural_max"], "p_pleural_median_in_reference": ins["p_pleural_median"],
             "reference_fraction_p_at_least_0.1": ins["fraction_with_p_pleural_at_least"]["0.1"],
             "argmax_background_fraction": ins["argmax_class_fraction"]["background"], "argmax_lung_fraction": ins["argmax_class_fraction"]["lung"],
             "argmax_pleural_fraction": ins["argmax_class_fraction"]["pleural_effusion"],
             "p_pleural_max_anywhere": a["max_p_pleural_anywhere"], "ml_p_at_least_0.1_anywhere": a["ml_with_p_pleural_at_least"]["0.1"],
             "output_file_equals_argmax": a["argmax_equals_output_file"], "tool_ml_with_mirroring": tr["inference"]["mirroring_on"]["pleural_ml_in_output_file"],
             "p_pleural_max_in_reference_with_mirroring": tr["inference"]["mirroring_on"]["inside_reference"]["p_pleural_max"]}
        rows.append(r)
        full[pid] = {"hu_bands_reference": bands(ct[ref]), "hu_bands_inner_part": bands(ct[core])}
        # overlay: slice with the largest reference area, plus the slice a quarter of the extent above it
        area = ref.sum((0, 1))
        k0 = int(area.argmax())
        zs = np.flatnonzero(area)
        k1 = int(zs[len(zs) // 2]) if abs(int(zs[len(zs) // 2]) - k0) >= 3 else int(zs[len(zs) // 4])   # a second slice inside the reference
        ks = sorted({k0, k1})
        fig, ax = plt.subplots(len(ks), 2, figsize=(11, 4.6 * len(ks)), squeeze=False)
        for i, k in enumerate(ks):
            img = np.rot90(ct[:, :, k])[:, ::-1]   # RAS: rot90 puts anterior up; the flip puts the patient's right on the left of the picture
            def tf(m):
                return np.rot90(m[:, :, k])[:, ::-1]
            for j in range(2):
                ax[i, j].imshow(img, cmap="gray", vmin=-160, vmax=240, interpolation="nearest")
                ax[i, j].axis("off")
                ax[i, j].text(4, 14, "R", color="yellow", fontsize=12)
                ax[i, j].text(img.shape[1] - 14, 14, "L", color="yellow", fontsize=12)
            ax[i, 0].set_title(f"{pid}  slice {k}  CT only (window -160..240 HU)", fontsize=9)
            ax[i, 1].contour(tf(ref).astype(float), [0.5], colors="cyan", linewidths=0.9)
            for lv, col in ((0.01, "yellow"), (0.1, "orange"), (0.5, "magenta")):
                if tf(pp).max() >= lv:
                    ax[i, 1].contour(tf(pp), [lv], colors=col, linewidths=0.9)
            ax[i, 1].contour(tf(pl), [0.5], colors="lime", linewidths=0.5)
            ax[i, 1].set_title("cyan: reference | model P(pleural effusion) 1% yellow, 10% orange, 50% magenta | green: model 'lung'", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "overlays" / f"{pid}_trace.png", dpi=110)
        plt.close(fig)
        print(pid, {k: r[k] for k in ("reference_ml", "axial_thickness_median_mm", "reference_mean_hu", "inner_part_fraction_of_reference", "inner_part_mean_hu",
                                      "inner_part_fluid_range_fraction", "inner_part_fat_range_fraction", "p_pleural_max_in_reference")}, flush=True)
    with open(out / "trace_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "trace_hu_bands.json").write_text(json.dumps(full, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
