"""Review of the Arm B class check (core env). Reads masks; changes nothing.

Two different things are kept apart:
  1. Output-class mapping (which label number is lung, which is pleural effusion). The archive has no label names, so
     the mapping is checked from the outputs on ALL scans: agreement of Arm B label 1 with Arm A's lung label
     (another model, not a reference), and HU content of both labels.
  2. The heuristic check in pleural_armB_qc.py (lung label > 80% air-like, effusion label < 20% air-like), which gave
     warnings on some scans. Each flagged scan is described here: what the non-air part of its "lung" label is
     (HU bands, inside or outside Arm A's lung label, inside the expert effusion outline or tumour outline) and a
     picture at the slice with most non-air "lung". A flag is cleared only by these per-scan facts, never by the
     statement that the heuristic was strict.
Usage: python scripts/pleural_armB_class_review.py --manifest <armB manifest> --armA-manifest <armA manifest> --qc <qc_armB.csv> --out <dir>
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


def load(p):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--armA-manifest", required=True)
    ap.add_argument("--qc", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))["cases"]
    arm_a = {c["case_id"]: c["prediction"] for c in json.loads((ROOT / args.armA_manifest).read_text(encoding="utf-8"))["cases"]}
    flagged = {r["case_id"] for r in csv.DictReader(open(ROOT / args.qc, encoding="utf-8")) if r.get("class_ok") == "False"}
    out = ROOT / args.out
    (out / "class_review").mkdir(parents=True, exist_ok=True)
    rows, panels = [], []
    for c in man:
        cid = c["case_id"]
        ct = load(ROOT / c["ct"]).astype(np.float32)
        pp = ROOT / c["prediction"]
        bl, be = load(pp.parent / "lung.nii.gz") > 0, load(pp) > 0
        al = load((ROOT / arm_a[cid]).parent / "lung_pleural.nii.gz") > 0
        nb = max(1, int(bl.sum()))
        nonair = bl & (ct >= -500)
        nn = max(1, int(nonair.sum()))
        r = {"case_id": cid, "set": c["set"], "flagged_by_heuristic": cid in flagged,
             "b_lung_vs_a_lung_dice": 2 * float((bl & al).sum()) / max(1, int(bl.sum() + al.sum())),
             "b_lung_share_inside_a_lung": float((bl & al).sum()) / nb,
             "b_lung_air_like_share": float((bl & (ct < -500)).sum()) / nb,
             "b_lung_share_minus500_to_minus100": float((bl & (ct >= -500) & (ct < -100)).sum()) / nb,
             "b_lung_share_minus100_or_more": float((bl & (ct >= -100)).sum()) / nb,
             "b_lung_nonair_share_inside_a_lung": float((nonair & al).sum()) / nn,
             "b_effusion_voxels": int(be.sum()),
             "b_effusion_air_like_share": float((be & (ct < -500)).sum()) / int(be.sum()) if be.any() else "",
             "b_effusion_median_hu": float(np.median(ct[be])) if be.any() else ""}
        ref = None
        if c.get("references", {}).get("first"):
            ref = load(ROOT / c["references"]["first"]) > 0
            r["b_lung_nonair_share_inside_expert_effusion_outline"] = float((nonair & ref).sum()) / nn
        if c.get("gtv") and (ROOT / c["gtv"]).exists():
            r["b_lung_nonair_share_inside_tumour_outline"] = float((nonair & (load(ROOT / c["gtv"]) > 0)).sum()) / nn
        rows.append(r)
        if cid in flagged:
            k = int(nonair.sum((0, 1)).argmax())
            panels.append((cid, ct[:, :, k].copy(), bl[:, :, k].copy(), al[:, :, k].copy(), be[:, :, k].copy(), None if ref is None else ref[:, :, k].copy(), k, r))
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "class_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def tf(m):
        return np.rot90(m)[:, ::-1]          # anterior up, patient right on the left of the picture
    for s in range(0, len(panels), 6):
        fig, ax = plt.subplots(2, 3, figsize=(16.5, 11), squeeze=False)
        for a in ax.ravel():
            a.axis("off")
        for a, (cid, im, bl, al, be, ref, k, r) in zip(ax.ravel(), panels[s:s + 6]):
            a.imshow(tf(im), cmap="gray", vmin=-1100, vmax=300, interpolation="nearest")
            for m, col, lw in ((al, "yellow", 0.7), (bl, "lime", 0.8), (ref, "cyan", 0.9), (be, "magenta", 0.9)):
                if m is not None and m.any():
                    a.contour(tf(m).astype(float), [0.5], colors=col, linewidths=lw)
            a.set_title(f"{cid} slice {k} | air-like {r['b_lung_air_like_share']:.2f}, non-air inside A lung {r['b_lung_nonair_share_inside_a_lung']:.2f}", fontsize=9)
            a.text(4, 18, "R", color="yellow", fontsize=12)
        fig.suptitle("green: B label 1 (lung)   yellow: A lung label   magenta: B label 2 (effusion)   cyan: expert effusion outline", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "class_review" / f"flagged_sheet_{s // 6 + 1}.png", dpi=85)
        plt.close(fig)
    d = np.array([r["b_lung_vs_a_lung_dice"] for r in rows])
    ea = [r["b_effusion_air_like_share"] for r in rows if r["b_effusion_air_like_share"] != ""]
    summ = {"scans": len(rows), "flagged": len(flagged),
            "class_mapping_all_scans": {"b_label1_vs_a_lung_dice_min": float(d.min()), "median": float(np.median(d)), "scans_below_0.9": sorted((r["case_id"], round(r["b_lung_vs_a_lung_dice"], 3)) for r in rows if r["b_lung_vs_a_lung_dice"] < 0.9),
                                        "b_label1_air_like_share_median": float(np.median([r["b_lung_air_like_share"] for r in rows])),
                                        "scans_with_b_label2": len(ea), "b_label2_air_like_share_max": max(ea), "b_label2_air_like_share_median": float(np.median(ea)),
                                        "b_label2_median_hu_range": [min(r["b_effusion_median_hu"] for r in rows if r["b_effusion_median_hu"] != ""), max(r["b_effusion_median_hu"] for r in rows if r["b_effusion_median_hu"] != "")]},
            "flagged_scans": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("set", "flagged_by_heuristic")} for r in rows if r["flagged_by_heuristic"]]}
    (out / "class_review_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
