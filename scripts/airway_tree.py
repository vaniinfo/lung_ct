"""Phase 5 (core env): lung volumes, airway centrelines, branch lengths and tortuosity.

Our own Python implementation (nibabel, scikit-image, Skan) - "lung_ct airway tree v2".
NOT AirQuant and no claim of equivalence (AirQuant is a methodological reference only).
Not validated: there is no airway reference in this dataset. Definitions are in
lung_ct/airway/tree.py. The airway mask is a model prediction (TotalSegmentator
lung_vessels task, `lung_airways`), so every metric depends on what that model segmented:
fewer detected branches do not indicate airway loss. Diameter, tapering and wall
thickness are not computed (later, separately validated extensions).

Per case -> runs/P5_airway_<n>cases_<ts>/<case>/:
  airway_branches.csv, airway_summary.json, airway_tree_qc.png
  (CT coronal minimum-intensity slab + centreline | airway mask MIP + centreline)
Run level: run_manifest.json, summary.csv, failures.csv

Usage:
    python scripts/airway_tree.py --seg-run runs/P2_seg_10cases_<ts> \
        [--eval-run runs/P2_eval_10cases_<ts>] [--cases coronacases_001]
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from lung_ct.airway.tree import DEFAULTS, METHOD, analyze_airway_tree, preprocess  # noqa: E402

MASK = "lung_vessels/lung_airways.nii.gz"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
        return c + ("+dirty" if d else "")
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def _overlay(ax, img2d, aspect, pts_by_gen, gmax, title, **imkw):
    ax.imshow(img2d, cmap="gray", aspect=aspect, **imkw)
    cmap = plt.get_cmap("turbo")
    for g in sorted(pts_by_gen):
        pts = np.array(pts_by_gen[g])
        ax.scatter(pts[:, 0], pts[:, 1], s=1.2, color=cmap(0.05 + 0.9 * g / gmax),
                   label=f"gen {g}")
    for txt, xy in (("R", (0.01, 0.5)), ("L", (0.95, 0.5)), ("S", (0.5, 0.96)),
                    ("I", (0.5, 0.01))):
        ax.text(*xy, txt, transform=ax.transAxes, color="yellow", fontsize=11, weight="bold")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def qc_png(ct_img, mask_img, res, out: Path, title: str):
    """Two coronal views with the centreline coloured by generation, radiological display
    (patient Right on screen left, Superior at top), both from canonical RAS:
      left : CT minimum-intensity projection over the anterior-posterior slab spanned by the
             centreline (+-5 mm), window [-1000, -500] HU
      right: airway mask MIP on the isotropic analysis grid
    Centreline voxels are mapped to the CT grid through world (RAS mm) coordinates."""
    gmax = max(res.gen_by_voxel.values(), default=1) or 1
    ct = nib.as_closest_canonical(ct_img)
    sp = np.sqrt((ct.affine[:3, :3] ** 2).sum(axis=0))
    vox = np.array(list(res.gen_by_voxel), dtype=float)
    gens = list(res.gen_by_voxel.values())
    world = res.affine_iso @ np.c_[vox, np.ones(len(vox))].T
    ijk = np.round(np.linalg.inv(ct.affine) @ world).astype(int)[:3].T
    data = np.asanyarray(ct.dataobj)
    j0 = max(int(ijk[:, 1].min() - 5 / sp[1]), 0)
    j1 = min(int(ijk[:, 1].max() + 5 / sp[1]) + 1, data.shape[1])
    minip = data[:, j0:j1, :].min(axis=1).astype(np.float32)          # (I, K)
    I, K = minip.shape
    ct_pts = {}
    for (i, _, k), g in zip(ijk, gens):
        ct_pts.setdefault(int(g), []).append((I - 1 - i, K - 1 - k))
    mask, _, _ = preprocess(mask_img, res.summary["params"]["iso_mm"])
    mip = mask.max(axis=1)
    X, Z = mip.shape
    m_pts = {}
    for (x, _, z), g in res.gen_by_voxel.items():
        m_pts.setdefault(g, []).append((X - 1 - x, Z - 1 - z))

    fig, axs = plt.subplots(1, 2, figsize=(13, 7))
    _overlay(axs[0], minip.T[::-1, ::-1], sp[2] / sp[0], ct_pts, gmax,
             f"CT coronal minIP slab (A-P voxels {j0}-{j1}), window -1000..-500 HU",
             vmin=-1000, vmax=-500)
    _overlay(axs[1], mip.T[::-1, ::-1], 1.0, m_pts, gmax,
             "predicted airway mask MIP (isotropic analysis grid)", vmin=0, vmax=1.6)
    axs[1].legend(markerscale=6, fontsize=6, loc="lower right", ncol=2)
    fig.suptitle(f"{title} - centreline by generation\n{METHOD}", fontsize=9)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def lung_volumes(ev_case):
    """Predicted (TotalSegmentator lobes) and reference (expert mask) lung volumes from the
    Phase 2 evaluation; missing values stay None (never 0)."""
    if not ev_case:
        return {}
    out = {}
    for side in ("lung_left", "lung_right", "lung_total"):
        r = ev_case.get(side) or {}
        out[f"pred_{side}_ml"] = r.get("pred_volume_ml")
        out[f"ref_{side}_ml"] = r.get("ref_volume_ml")
        out[f"dice_{side}"] = r.get("dice")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seg-run", required=True, type=Path)
    ap.add_argument("--eval-run", type=Path, help="Phase 2 evaluation run (lung volumes)")
    ap.add_argument("--cases", nargs="+")
    args = ap.parse_args()
    ev_cases = {}
    if args.eval_run:
        ev = json.loads((ROOT / args.eval_run / "run_manifest.json").read_text())
        ev_cases = {c["case_id"]: c for c in ev["cases"]}

    seg = json.loads((ROOT / args.seg_run / "run_manifest.json").read_text())
    cases = [c for c in seg["cases"] if not args.cases or c["case_id"] in args.cases]
    import importlib.metadata as md
    run_id = f"P5_airway_{len(cases)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "provenance": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(), "python": platform.python_version(),
            "packages": {p: md.version(p) for p in ("numpy", "scipy", "scikit-image",
                                                     "skan", "networkx", "nibabel")},
            "seg_run": str(args.seg_run),
            "eval_run": str(args.eval_run) if args.eval_run else None,
            "input_mask": f"TotalSegmentator {seg['provenance']['packages']['TotalSegmentator']} "
                          f"lung_vessels/lung_airways (prediction)",
            "method": METHOD,
            "not_computed": ["diameter", "tapering", "wall_thickness"],
            "params": DEFAULTS,
        },
        "cases": [],
    }
    summary_rows, failures = [], []
    for c in cases:
        cid = c["case_id"]
        print(f"[{cid}] ...", end=" ", flush=True)
        cdir = run_dir / cid
        cdir.mkdir()
        mpath = ROOT / c["output_dir"] / MASK
        rec = {"case_id": cid, "mask": str(mpath.relative_to(ROOT)), "status": "ok"}
        try:
            if c["status"] != "ok":
                raise RuntimeError(f"segmentation status {c['status']}")
            rec["mask_sha256"] = sha256_file(mpath)
            img = nib.load(str(mpath))
            t0 = time.time()
            res = analyze_airway_tree(img)
            rec["elapsed_seconds"] = round(time.time() - t0, 1)
            rec["summary"] = res.summary
            rec["lung_volumes"] = lung_volumes(ev_cases.get(cid))
            with open(cdir / "airway_branches.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(res.branches[0]))
                w.writeheader()
                w.writerows(res.branches)
            (cdir / "airway_summary.json").write_text(json.dumps(
                {**rec, "provenance": manifest["provenance"]}, indent=2))
            qc_png(nib.load(str(ROOT / c["ct_file"])), img, res, cdir / "airway_tree_qc.png", cid)
            s = res.summary
            summary_rows.append({
                "case_id": cid, **rec["lung_volumes"],
                "n_tree_branches": s["n_tree_branches"],
                "n_terminal": s["n_terminal_branches"], "max_generation": s["max_generation"],
                "branches_left": s["branches_by_side"]["left"],
                "branches_right": s["branches_by_side"]["right"],
                "n_cycles": s["n_cycles"],
                "total_centreline_mm": s["total_centreline_length_mm"],
                "median_tortuosity_internal": s["median_tortuosity_internal"],
                "airway_mask_kept_ml": s["preprocess"]["kept_volume_ml"],
                "airway_mask_discarded_ml": s["preprocess"]["discarded_volume_ml"],
                "flags": "; ".join(s["flags"]), "elapsed_seconds": rec["elapsed_seconds"]})
            print("ok", s["n_tree_branches"], "branches, max gen", s["max_generation"],
                  f"{rec['elapsed_seconds']}s", "; ".join(s["flags"]))
        except Exception as e:
            rec.update(status="failed", error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc())
            failures.append({"case_id": cid, "reason": rec["error"]})
            (cdir / "airway_summary.json").write_text(json.dumps(rec, indent=2))
            print("FAILED", rec["error"])
        manifest["cases"].append(rec)

    manifest["provenance"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    if summary_rows:
        with open(run_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            w.writeheader()
            w.writerows(summary_rows)
    with open(run_dir / "failures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "reason"])
        w.writeheader()
        w.writerows(failures)
    print(f"done: {len(summary_rows)} ok, {len(failures)} failed -> {run_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
