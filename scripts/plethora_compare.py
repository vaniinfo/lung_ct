"""Compare the TotalSegmentator pleural-effusion outline with the PleThora expert outlines (core env).

External check of the measurement behind the lung_ct "pleural effusion" answer (evidence rows E71 + E72, both sides):
  rule (configs/checklist_v3_4.json and v3_5, unchanged):  present >= 20 mL,  none < 5 mL,  in between = undecided.
The limits are read from the config. Nothing is fitted or tuned here.

Reference: PleThora (Kiser et al. 2020). Effusion.Event 1 / 0 = effusion present / absent on the scan;
expert outlines exist for the 78 present scans. The outlines EXCLUDE effusion that overlaps the tumour (GTV), so the
tool can legitimately outline more fluid than the reference on some scans.

Per scan the script checks, before any comparison:
  - geometry: reference and prediction are brought to the CT grid by axis reorientation only (no interpolation);
    shapes and affines must then agree, otherwise the scan is excluded and listed
  - anatomy: the expert thoracic-cavity outline must sit on lung-density voxels of the CT (and do so better than a
    front-back or left-right mirrored copy), which confirms the alignment from the image content
  - intensity: the scan contains air near -1000 HU and values in a CT range

Usage: python -I scripts/plethora_compare.py [--out runs/E14_plethora_pleural_effusion_<ts>]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "raw" / "plethora"
D = ROOT / "data" / "derived" / "plethora"
BANDS = [(0, 5), (5, 20), (20, 50), (50, 150), (150, 1e9)]          # mL of reference effusion; first two = the rule's own limits


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canon(path: Path):
    return nib.as_closest_canonical(nib.load(str(path)))


def arr(img) -> np.ndarray:
    return np.asanyarray(img.dataobj)


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    s = int(a.sum()) + int(b.sum())
    return None if s == 0 else round(2.0 * int((a & b).sum()) / s, 4)


def wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z = 1.96
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * max(0.0, c - h), 1), round(100 * min(1.0, c + h), 1)]


def spearman(x, y) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


def rule(ml: float, present_at: float, absent_below: float) -> str:
    return "present" if ml >= present_at else ("none" if ml < absent_below else "undecided")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = json.loads((ROOT / "configs" / "checklist_v3_5.json").read_text(encoding="utf-8"))
    ck4 = json.loads((ROOT / "configs" / "checklist_v3_4.json").read_text(encoding="utf-8"))
    cond = next(c for c in ck["conditions"] if c["key"] == "pleural_effusion")
    assert cond == next(c for c in ck4["conditions"] if c["key"] == "pleural_effusion")
    hi, lo = float(cond["present_at_least"]), float(cond["absent_below"])
    sel = json.loads((P / "selection.json").read_text(encoding="utf-8"))
    fetch = json.loads((P / "fetch_log.json").read_text(encoding="utf-8"))
    seg = json.loads((D / "seg_manifest.json").read_text(encoding="utf-8"))
    out = ROOT / (args.out or f"runs/E14_plethora_pleural_effusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in sel["cases"]:
        pid = c["patient_id"]
        r = {"patient_id": pid, "group": c["group"], "status": "ok", "issues": [], "csv_reference_ml": round(c["reference_right_ml"] + c["reference_left_ml"], 1)}
        try:
            if fetch.get(pid, {}).get("status") != "ok":
                raise RuntimeError(f"scan not fetched: {fetch.get(pid, {}).get('error', 'no record')}")
            if seg["cases"].get(pid, {}).get("status") != "ok":
                raise RuntimeError(f"segmentation failed: {seg['cases'].get(pid, {}).get('issues', 'no record')}")
            r["issues"] += fetch[pid].get("issues", [])
            if any("non-uniform slice spacing" in i for i in r["issues"]):
                raise RuntimeError("non-uniform slice spacing or missing slices in the DICOM series: volumes in mL would not be reliable")
            ct = canon(P / "volumes" / f"{pid}.nii.gz")
            hu = arr(ct)
            vox_ml = abs(float(np.linalg.det(ct.affine[:3, :3]))) / 1000.0
            r.update(shape="x".join(map(str, ct.shape)), spacing_mm=" ".join(f"{v:.3f}" for v in ct.header.get_zooms()[:3]),
                     hu_p0_1=float(np.percentile(hu[::4, ::4, ::2], 0.1)), hu_max=int(hu.max()))
            if not (-1100 <= r["hu_p0_1"] <= -900):
                r["issues"].append(f"air level {r['hu_p0_1']:.0f} HU is not near -1000")

            def on_grid(path: Path, name: str) -> np.ndarray:
                m = canon(path)
                if m.shape != ct.shape or not np.allclose(m.affine, ct.affine, atol=0.05):
                    raise RuntimeError(f"{name}: grid differs from the CT after reorientation (shape {m.shape} vs {ct.shape})")
                return arr(m)
            pred = on_grid(D / pid / "pleural_pericard_effusion" / "pleural_effusion.nii.gz", "prediction") > 0
            r["tool_ml"] = round(float(pred.sum()) * vox_ml, 1)
            r["tool_answer"] = rule(r["tool_ml"], hi, lo)
            thf = sorted((P / "thorax").rglob(f"{pid}_thor_cav_primary_reviewer.nii.gz"))
            if thf:
                th = on_grid(thf[0], "thoracic cavity") > 0
                lung = hu < -300
                ok_, fb, lr = (float((lung & m).sum()) / max(1, int(m.sum())) for m in (th, th[:, ::-1, :], th[::-1, :, :]))
                r.update(thorax_on_lung_density=round(ok_, 3), thorax_mirrored_front_back=round(fb, 3), thorax_mirrored_left_right=round(lr, 3))
                if not (ok_ > fb and ok_ >= lr - 0.02 and ok_ > 0.35):
                    r["issues"].append("anatomy check: thoracic-cavity outline does not sit on lung density better than a mirrored copy")
                r["tool_ml_inside_reference_thoracic_cavity"] = round(float((pred & th).sum()) * vox_ml, 1)
            else:
                r["issues"].append("no thoracic-cavity outline for the anatomy check")
            refs = {}
            for f in sorted((P / "effusions").rglob(f"{pid}_effusion_*_reviewer.nii.gz")):
                refs[f.name.split("_effusion_")[1].split("_reviewer")[0]] = on_grid(f, f.name) > 0
            if c["group"] == "effusion":
                if not refs:
                    raise RuntimeError("no reference outline")
                vols = {k: round(float(v.sum()) * vox_ml, 1) for k, v in refs.items()}
                # official reference = the reviewer file whose volume equals the PleThora spreadsheet value (their 'primary reviewer')
                best = min(vols, key=lambda k: abs(vols[k] - r["csv_reference_ml"]))
                if abs(vols[best] - r["csv_reference_ml"]) > max(1.0, 0.02 * r["csv_reference_ml"]):
                    r["issues"].append(f"no reviewer file matches the spreadsheet volume {r['csv_reference_ml']} mL (files: {vols}); first reviewer used")
                    best = "first" if "first" in refs else best
                ref = refs[best]
                r.update(reference_reviewer=best, reference_ml=vols[best], reviewer_volumes_ml=json.dumps(vols),
                         reviewer_min_ml=min(vols.values()), reviewer_max_ml=max(vols.values()), dice=dice(pred, ref),
                         reviewer_pair_dice=json.dumps({f"{a}-{b}": dice(refs[a], refs[b]) for i, a in enumerate(sorted(refs)) for b in sorted(refs)[i + 1:]}),
                         tool_ml_overlapping_reference=round(float((pred & ref).sum()) * vox_ml, 1))
            else:
                if refs:
                    r["issues"].append("reference outline exists for a scan coded 'no effusion'")
                r.update(reference_ml=0.0)
        except Exception as e:  # noqa: BLE001  reported, never dropped
            r.update(status="excluded", error=f"{type(e).__name__}: {e}")
        r["issues"] = "; ".join(r["issues"])
        rows.append(r)
        print(pid, r["group"], r["status"], r.get("reference_ml"), r.get("tool_ml"), r.get("tool_answer"), r.get("dice"), r["issues"] or r.get("error", ""), flush=True)

    cols = ["patient_id", "group", "status", "reference_ml", "tool_ml", "tool_answer", "dice", "reference_reviewer", "csv_reference_ml", "reviewer_min_ml", "reviewer_max_ml",
            "reviewer_volumes_ml", "reviewer_pair_dice", "tool_ml_overlapping_reference", "tool_ml_inside_reference_thoracic_cavity", "shape", "spacing_mm", "hu_p0_1", "hu_max",
            "thorax_on_lung_density", "thorax_mirrored_front_back", "thorax_mirrored_left_right", "issues", "error"]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    pos, neg = [r for r in ok if r["group"] == "effusion"], [r for r in ok if r["group"] == "no_effusion"]

    def answers(rs):
        return {a: sum(r["tool_answer"] == a for r in rs) for a in ("present", "undecided", "none")}
    bands = []
    for a, b in BANDS:
        rs = [r for r in pos if a <= r["reference_ml"] < b]
        bands.append({"reference_ml": f"{a:g}-{b:g}" if b < 1e8 else f"{a:g}+", "scans": len(rs), **answers(rs),
                      "median_tool_ml": round(float(np.median([r["tool_ml"] for r in rs])), 1) if rs else None,
                      "median_dice": round(float(np.median([r["dice"] for r in rs if r["dice"] is not None])), 3) if any(r["dice"] is not None for r in rs) else None})
    ref_v, tool_v = np.array([r["reference_ml"] for r in pos]), np.array([r["tool_ml"] for r in pos])
    diff = tool_v - ref_v
    dices = np.array([r["dice"] for r in pos if r["dice"] is not None])
    inter = [v for r in pos for v in json.loads(r["reviewer_pair_dice"]).values() if v is not None]
    inside = sum(r["reviewer_min_ml"] <= r["tool_ml"] <= r["reviewer_max_ml"] for r in pos if r["reviewer_min_ml"] != r["reviewer_max_ml"])
    n_multi = sum(r["reviewer_min_ml"] != r["reviewer_max_ml"] for r in pos)
    ap_, an_ = answers(pos), answers(neg)
    summ = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "what": "external check of the pleural-effusion outline volume; development evidence, single public collection",
        "rule_tested": {"present_at_least_ml": hi, "none_below_ml": lo, "source": "configs/checklist_v3_4.json = checklist_v3_5.json (unchanged, not tuned here)"},
        "selection": {"seed": sel["seed"], "effusion_scans": sum(c["group"] == "effusion" for c in sel["cases"]), "no_effusion_scans": sum(c["group"] == "no_effusion" for c in sel["cases"])},
        "excluded": [{"patient_id": r["patient_id"], "group": r["group"], "why": r["error"]} for r in rows if r["status"] != "ok"],
        "scans_with_issues": [{"patient_id": r["patient_id"], "issues": r["issues"]} for r in ok if r["issues"]],
        "effusion_scans": {"n": len(pos), "tool_answer": ap_, "present_pct": round(100 * ap_["present"] / max(1, len(pos)), 1), "present_ci95": wilson(ap_["present"], len(pos)),
                           "answered_none_pct": round(100 * ap_["none"] / max(1, len(pos)), 1), "by_reference_volume": bands,
                           "tool_zero_ml": sum(r["tool_ml"] == 0 for r in pos)},
        "no_effusion_scans": {"n": len(neg), "tool_answer": an_, "none_pct": round(100 * an_["none"] / max(1, len(neg)), 1), "none_ci95": wilson(an_["none"], len(neg)),
                              "answered_present_pct": round(100 * an_["present"] / max(1, len(neg)), 1),
                              "tool_ml_percentiles_0_50_75_90_100": [round(float(np.percentile([r["tool_ml"] for r in neg], q)), 1) for q in (0, 50, 75, 90, 100)] if neg else None},
        "volume_agreement_effusion_scans": {
            "median_reference_ml": round(float(np.median(ref_v)), 1), "median_tool_ml": round(float(np.median(tool_v)), 1),
            "median_difference_tool_minus_reference_ml": round(float(np.median(diff)), 1), "median_absolute_difference_ml": round(float(np.median(np.abs(diff))), 1),
            "difference_percentiles_5_25_75_95_ml": [round(float(np.percentile(diff, q)), 1) for q in (5, 25, 75, 95)], "spearman": spearman(ref_v, tool_v),
            "tool_within_range_of_reviewers": f"{inside} of {n_multi} scans with more than one distinct reviewer volume"},
        "overlap": {"median_dice_tool_vs_reference": round(float(np.median(dices)), 3), "dice_percentiles_10_25_75_90": [round(float(np.percentile(dices, q)), 3) for q in (10, 25, 75, 90)],
                    "median_dice_between_reviewers_recomputed": round(float(np.median(inter)), 3) if inter else None, "reviewer_pairs": len(inter)},
        "limits": ["one public collection of radiotherapy-planning scans of lung-cancer patients; not CT-RATE, not our clinician's reading",
                   "reference outlines exclude effusion overlapping the tumour, so tool volume can exceed the reference for a valid reason",
                   "the source of the 600 training scans of the TotalSegmentator effusion model is not published; overlap with this collection cannot be excluded",
                   "'no effusion' is the PleThora spreadsheet code; no outline exists for those scans",
                   "says nothing about where 'present' should start clinically; the 20 mL and 5 mL limits were not changed"],
        "inputs": {"selection_sha256": sha(P / "selection.json"), "fetch_log_sha256": sha(P / "fetch_log.json"), "seg_manifest_sha256": sha(D / "seg_manifest.json"),
                   "segmentation_packages": seg.get("packages"), "weights_sha256": seg.get("weights_sha256"),
                   "plethora_files_sha256": {f.name: sha(f) for f in sorted((P / "download").iterdir())}}}
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("excluded", "effusion_scans", "no_effusion_scans", "volume_agreement_effusion_scans", "overlap")}, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
