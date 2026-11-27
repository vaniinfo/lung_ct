"""Compare the second segmentation (NV-Segment-CT) with TotalSegmentator on the same scans (core env).

Both are predictions: this measures AGREEMENT between two models, not accuracy. No clinician outline is used.

For every case of the given NV-Segment-CT runs:
  * geometry: same shape and affine as the TotalSegmentator masks, checked before any overlap is counted
  * Dice and volumes for the structures both models outline (lung lobes, trachea, heart, aorta, liver when present)
  * airways: TotalSegmentator lung_airways against NV-Segment-CT 'airway', and trachea + airways against trachea + airway
  * the class 'lung tumor' against TotalSegmentator's lung_nodules mask: volumes, pieces, overlap
  * structures that cannot be inside a chest scan (pelvis, legs, head) but received voxels: listed, not hidden
  * repeatability: a case present in two runs must give identical files

Writes runs/E25_nvsegment_vs_totalseg_<time>/ : agreement.csv, per_case.json, summary.json, SUMMARY.md

Usage:
    .venv/Scripts/python.exe scripts/compare_nvsegment_totalseg.py --runs runs/P2_nvsegment_3cases_... runs/P2_nvsegment_1cases_...
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent

PAIRS = [  # TotalSegmentator file (task/name), NV-Segment-CT label, shown name
    ("total/lung_upper_lobe_left", 28, "Left upper lobe"), ("total/lung_lower_lobe_left", 29, "Left lower lobe"),
    ("total/lung_upper_lobe_right", 30, "Right upper lobe"), ("total/lung_middle_lobe_right", 31, "Right middle lobe"),
    ("total/lung_lower_lobe_right", 32, "Right lower lobe"), ("total/trachea", 57, "Trachea"),
    ("total/heart", 115, "Heart"), ("total/aorta", 6, "Aorta"), ("total/liver", 1, "Liver"),
]
# classes that lie outside any chest scan: voxels here are errors of the model, reported as such
OUTSIDE_CHEST = [15, 22, 33, 34, 35, 58, 59, 60, 61, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 118, 120, 127]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    """2 |A and B| / (|A| + |B|); None when both are empty (not defined, never reported as 1)."""
    s = int(a.sum()) + int(b.sum())
    return None if s == 0 else round(2.0 * int((a & b).sum()) / s, 4)


def ml(mask: np.ndarray, voxel_mm3: float) -> float:
    return round(int(mask.sum()) * voxel_mm3 / 1000.0, 2)


def pieces(mask: np.ndarray) -> int:
    return int(ndimage.label(mask, structure=np.ones((3, 3, 3)))[1])


def same_grid(a, b) -> tuple[bool, float]:
    d = float(np.abs(np.asarray(a.affine, float) - np.asarray(b.affine, float)).max())
    return tuple(a.shape[:3]) == tuple(b.shape[:3]) and d < 1e-3, d


def compare_case(cid: str, case_dir: Path, names: dict) -> tuple[dict, list[dict]]:
    seg = ROOT / "data" / "derived" / cid / "segmentation"
    lab_img = nib.load(str(case_dir / "nvseg_labels.nii.gz"))
    lab = np.asanyarray(lab_img.dataobj)
    vox = float(abs(np.linalg.det(np.asarray(lab_img.affine, float)[:3, :3])))
    rows, info = [], {"case_id": cid, "nvsegment_case_dir": str(case_dir.relative_to(ROOT)), "geometry_max_affine_diff_mm": 0.0, "not_compared": []}

    def ts(rel):
        p = seg / f"{rel}.nii.gz"
        if not p.exists():
            info["not_compared"].append(f"{rel}: no TotalSegmentator file")
            return None
        img = nib.load(str(p))
        ok, d = same_grid(img, lab_img)
        info["geometry_max_affine_diff_mm"] = max(info["geometry_max_affine_diff_mm"], d)
        if not ok:
            raise RuntimeError(f"{cid} {rel}: grid differs from the NV-Segment-CT label map (max affine difference {d})")
        return np.asanyarray(img.dataobj) > 0

    def nv_extra(name):
        img = nib.load(str(case_dir / "extra" / f"{name}.nii.gz"))
        ok, d = same_grid(img, lab_img)
        if not ok:
            raise RuntimeError(f"{cid} extra/{name}: grid differs")
        return np.asanyarray(img.dataobj) > 0

    def row(structure, t, n, note=""):
        rows.append({"case_id": cid, "structure": structure, "totalsegmentator_ml": ml(t, vox), "nvsegment_ml": ml(n, vox),
                     "dice": dice(t, n), "totalsegmentator_empty": not t.any(), "nvsegment_empty": not n.any(), "note": note})

    for rel, label, shown in PAIRS:
        t = ts(rel)
        if t is not None:
            row(shown, t, lab == label)
    lobes_t = [ts(r) for r, _, _ in PAIRS[:5]]
    if all(x is not None for x in lobes_t):
        row("Both lungs (five lobes together)", np.logical_or.reduce(lobes_t), np.isin(lab, [28, 29, 30, 31, 32]))
    aw_t, tr_t, aw_n = ts("lung_vessels/lung_airways"), ts("total/trachea"), nv_extra("airway")
    if aw_t is not None:
        row("Airways (TS lung_airways / NV airway)", aw_t, aw_n, "the two classes are defined differently: see SUMMARY.md")
        if tr_t is not None:
            row("Trachea + airways together", aw_t | tr_t, aw_n | (lab == 57))
        info["airway_pieces"] = {"totalsegmentator_lung_airways": pieces(aw_t), "nvsegment_airway": pieces(aw_n)}
    tum = nv_extra("lung_tumor")
    nod = ts("lung_nodules/lung_nodules")
    lungs_n = np.isin(lab, [28, 29, 30, 31, 32])
    info["class_lung_tumor"] = {"ml": ml(tum, vox), "pieces": pieces(tum), "ml_inside_nvsegment_lobes": ml(tum & lungs_n, vox)}
    if nod is not None:
        row("TS lung_nodules / NV class 'lung tumor'", nod, tum, "different tasks; overlap shown for orientation only")
        info["class_lung_tumor"].update(totalsegmentator_lung_nodules_ml=ml(nod, vox), totalsegmentator_lung_nodules_pieces=pieces(nod))
    counts = np.bincount(lab.ravel(), minlength=256)
    info["outside_chest_classes_with_voxels"] = {names[str(x)]: round(int(counts[x]) * vox / 1000.0, 2) for x in OUTSIDE_CHEST if counts[x]}
    info["other_extra_classes_ml"] = {n: ml(nv_extra(n), vox) for n in ("pancreatic_tumor", "hepatic_vessel", "hepatic_tumor", "colon_cancer_primaries", "bone_lesion")}
    return info, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    args = ap.parse_args()
    out = ROOT / "runs" / f"E25_nvsegment_vs_totalseg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    seen, all_rows, per_case, repeat, failed = {}, [], [], [], []
    for r in args.runs:
        run = r if r.is_absolute() else ROOT / r
        man = json.loads((run / "run_manifest.json").read_text())
        names = man["config"]["class_names"]
        for c in man["cases"]:
            cid = c["case_id"]
            if c["status"] != "ok":
                failed.append({"case_id": cid, "run": run.name, "reason": "; ".join(c["issues"])})
                continue
            if cid in seen:        # the same scan in two runs: files must be identical
                a, b = seen[cid], c["files_sha256"]
                repeat.append({"case_id": cid, "runs": [seen[cid + "/run"], run.name], "identical_files": a == b,
                               "files_that_differ": sorted(k for k in a if a[k] != b.get(k))})
                continue
            seen[cid], seen[cid + "/run"] = c["files_sha256"], run.name
            info, rows = compare_case(cid, run / cid, names)
            info["seconds_total"], info["peak_gpu_memory_mb"] = c["seconds_total"], c["peak_gpu_memory_mb"]
            info["classes_found"], info["classes_empty"] = c["classes_found"], c["classes_empty"]
            per_case.append(info)
            all_rows += rows
    with open(out / "agreement.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    (out / "per_case.json").write_text(json.dumps(per_case, indent=2))
    structures = list(dict.fromkeys(r["structure"] for r in all_rows))
    table = {}
    for s in structures:
        d = [r["dice"] for r in all_rows if r["structure"] == s and r["dice"] is not None]
        table[s] = {"cases": len([r for r in all_rows if r["structure"] == s]), "dice_min": min(d) if d else None,
                    "dice_median": round(float(np.median(d)), 4) if d else None, "dice_max": max(d) if d else None}
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "runs": [str(r) for r in args.runs], "cases": [c["case_id"] for c in per_case],
               "what_this_is": "agreement between two segmentation models; neither is a reference, so this is not accuracy",
               "agreement": table, "repeatability": repeat, "failed": failed}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    md = ["# NV-Segment-CT against TotalSegmentator: agreement on " + str(len(per_case)) + " scans", "",
          "Both are model predictions. Dice (0 = no overlap, 1 = identical) measures how far the two agree, not which one is right.", "",
          "| Structure | Scans | Dice lowest | Dice middle | Dice highest |", "|---|---|---|---|---|"]
    md += [f"| {s} | {t['cases']} | {t['dice_min']} | {t['dice_median']} | {t['dice_max']} |" for s, t in table.items()]
    md += ["", "Airways: TotalSegmentator's `lung_airways` and NV-Segment-CT's `airway` are different class definitions "
           "(how much of the trachea and of the small branches each includes), so a low Dice there is expected and does not rank them.", "",
           "| Scan | Class 'lung tumor' mL (pieces) | TS lung_nodules mL (pieces) | Airway pieces TS / NV | Outside-chest classes with voxels |", "|---|---|---|---|---|"]
    for c in per_case:
        t, a = c["class_lung_tumor"], c.get("airway_pieces", {})
        md.append(f"| {c['case_id']} | {t['ml']} ({t['pieces']}) | {t.get('totalsegmentator_lung_nodules_ml', 'n/a')} ({t.get('totalsegmentator_lung_nodules_pieces', 'n/a')}) | "
                  f"{a.get('totalsegmentator_lung_airways', 'n/a')} / {a.get('nvsegment_airway', 'n/a')} | "
                  f"{', '.join(f'{k} {v} mL' for k, v in c['outside_chest_classes_with_voxels'].items()) or 'none'} |")
    md += ["", "Repeatability: " + ("; ".join(f"{r['case_id']} run twice, files identical: {r['identical_files']}" for r in repeat) or "no scan was run twice"),
           "", "Failed cases: " + (", ".join(f["case_id"] for f in failed) or "none")]
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
