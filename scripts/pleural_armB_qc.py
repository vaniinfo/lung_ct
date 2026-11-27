"""Technical QC of Arm B outputs (core env): source-grid mapping, class identity, voxel volume, orientation overlays.

Per scan in the manifest:
  grid      the predicted masks have the CT's shape and affine (as stored, no reorientation applied to compare)
  class     label 1 ("lung") is lung-sized and made of air-like voxels; label 2 ("pleural effusion") is not air
  mirror    share of label-1 voxels that are air-like (< -500 HU) as stored, and after mirroring the mask left-right
            and front-back; a wrongly oriented output scores lower as stored than one of its mirrors
  volume    voxel volume from the NIfTI header against the value in the run log (written by SimpleITK inside nnU-Net)
  overlay   PNG at the slice with most predicted effusion (or most lung): CT, Arm B lung (green), Arm B effusion
            (magenta), Arm A effusion (yellow), first-reviewer reference (cyan); patient right on the left of the picture
Usage: python scripts/pleural_armB_qc.py --manifest <armB manifest> --armA-manifest <armA manifest> --run-log <run_log json> --out <dir>
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

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--armA-manifest", required=True)
    ap.add_argument("--run-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlays", type=int, default=10**6, help="render at most this many overlays")
    args = ap.parse_args()
    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    arm_a = {c["case_id"]: c["prediction"] for c in json.loads((ROOT / args.armA_manifest).read_text(encoding="utf-8"))["cases"]}
    log = json.loads((ROOT / args.run_log).read_text(encoding="utf-8"))["cases"]
    out = ROOT / args.out
    (out / "qc_overlays").mkdir(parents=True, exist_ok=True)
    rows, n_png = [], 0
    for c in man["cases"]:
        cid = c["case_id"]
        r = {"case_id": cid, "set": c["set"], "run_status": log.get(cid, {}).get("status", "not run")}
        pp = ROOT / c["prediction"]
        if not pp.exists():
            r["qc"] = "no prediction file"
            rows.append(r)
            continue
        ct_raw, eff_raw, lung_raw = nib.load(ROOT / c["ct"]), nib.load(pp), nib.load(pp.parent / "lung.nii.gz")
        r["same_shape_as_ct"] = eff_raw.shape == ct_raw.shape
        r["max_abs_affine_difference"] = float(np.abs(eff_raw.affine - ct_raw.affine).max()) if r["same_shape_as_ct"] else None
        r["grid_ok"] = bool(r["same_shape_as_ct"] and r["max_abs_affine_difference"] < 1e-2)
        ct = nib.as_closest_canonical(ct_raw)
        a = np.asarray(ct.dataobj).astype(np.float32)
        eff = np.asarray(nib.as_closest_canonical(eff_raw).dataobj) > 0
        lung = np.asarray(nib.as_closest_canonical(lung_raw).dataobj) > 0
        vml = float(np.prod(ct.header.get_zooms()[:3])) / 1000.0
        air = a < -500
        nl = max(1, int(lung.sum()))
        r.update(voxel_ml_from_header=vml, lung_l=lung.sum() * vml / 1000, effusion_ml=float(eff.sum()) * vml,
                 lung_ml_in_run_log=log.get(cid, {}).get("label_ml", {}).get("1"), effusion_ml_in_run_log=log.get(cid, {}).get("label_ml", {}).get("2", 0.0),
                 lung_air_like_fraction=float((lung & air).sum()) / nl, lung_air_like_fraction_mirrored_left_right=float((lung[::-1] & air).sum()) / nl,
                 lung_air_like_fraction_mirrored_front_back=float((lung[:, ::-1] & air).sum()) / nl,
                 effusion_air_like_fraction=(float((eff & air).sum()) / int(eff.sum())) if eff.any() else None,
                 effusion_mean_hu=float(a[eff].mean()) if eff.any() else None,
                 right_lung_l=float(lung[:lung.shape[0] // 2].sum()) * vml / 1000, left_lung_l=float(lung[lung.shape[0] // 2:].sum()) * vml / 1000)
        r["volume_matches_run_log"] = bool(r["lung_ml_in_run_log"] is not None and abs(r["lung_l"] * 1000 - r["lung_ml_in_run_log"]) < 0.5
                                           and abs(r["effusion_ml"] - (r["effusion_ml_in_run_log"] or 0.0)) < 0.01)
        r["class_ok"] = bool(1.0 <= r["lung_l"] <= 9.0 and r["lung_air_like_fraction"] > 0.8 and (r["effusion_air_like_fraction"] is None or r["effusion_air_like_fraction"] < 0.2))
        r["orientation_ok"] = bool(r["lung_air_like_fraction"] > r["lung_air_like_fraction_mirrored_left_right"]
                                   and r["lung_air_like_fraction"] > r["lung_air_like_fraction_mirrored_front_back"])
        rows.append(r)
        if n_png < args.overlays:
            n_png += 1
            ref = None
            if c.get("references", {}).get("first"):
                ref = np.asarray(nib.as_closest_canonical(nib.load(ROOT / c["references"]["first"])).dataobj) > 0
            ea = np.asarray(nib.as_closest_canonical(nib.load(ROOT / arm_a[cid])).dataobj) > 0 if (ROOT / arm_a[cid]).exists() else None
            key = eff if eff.any() else ref if ref is not None and ref.any() else lung
            k0 = int(key.sum((0, 1)).argmax())
            k1 = int(lung.sum((0, 1)).argmax())
            ks = [k0] if abs(k0 - k1) < 3 else [k0, k1]
            fig, ax = plt.subplots(1, len(ks), figsize=(6.2 * len(ks), 5.6), squeeze=False)
            for j, k in enumerate(ks):
                def tf(m):
                    return np.rot90(m[:, :, k])[:, ::-1]      # RAS: anterior up, patient right on the left of the picture
                ax[0, j].imshow(tf(a), cmap="gray", vmin=-1000, vmax=300, interpolation="nearest")
                for m, col, lw in ((lung, "lime", 0.6), (ref, "cyan", 1.0), (ea, "yellow", 0.9), (eff, "magenta", 0.9)):
                    if m is not None and m[:, :, k].any():
                        ax[0, j].contour(tf(m).astype(float), [0.5], colors=col, linewidths=lw)
                ax[0, j].text(4, 16, "R", color="yellow", fontsize=13)
                ax[0, j].text(tf(a).shape[1] - 18, 16, "L", color="yellow", fontsize=13)
                ax[0, j].set_title(f"{cid} slice {k} | green B lung, magenta B effusion, yellow A effusion, cyan reference", fontsize=7.5)
                ax[0, j].axis("off")
            fig.tight_layout()
            fig.savefig(out / "qc_overlays" / f"{cid}_armB_qc.png", dpi=100)
            plt.close(fig)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "qc_armB.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    done = [r for r in rows if "grid_ok" in r]
    summ = {"cases": len(rows), "with_prediction": len(done), **{k: sum(bool(r[k]) for r in done) for k in ("grid_ok", "class_ok", "orientation_ok", "volume_matches_run_log")},
            "not_ok": [{"case_id": r["case_id"], **{k: r[k] for k in ("grid_ok", "class_ok", "orientation_ok", "volume_matches_run_log")}} for r in done
                       if not all(r[k] for k in ("grid_ok", "class_ok", "orientation_ok", "volume_matches_run_log"))], "overlays": n_png}
    (out / "qc_armB_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
