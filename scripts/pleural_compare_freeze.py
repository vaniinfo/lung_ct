"""Freeze record of the pleural-effusion pipeline comparison (core env). Reads and hashes; changes no result.

Writes <run>/freeze/:
  case_manifest.csv            every scan: set, CT hash, both predictions' hashes, reference file hashes
  manifest_armA.json / manifest_armB.json   copies of the manifests the scores were computed from
  model_identity.json          weights / plans hashes and versions of both pipelines
  preprocessing.json           what each pipeline does to a scan before and after the network (published vs ours)
  environment_core.txt / environment_segmentation.txt / environment_pleural_v1.txt   installed packages of the three Python environments
  count_verification.json      summary counts recomputed from the per-case table
  SHA256SUMS.json              hashes of metrics, tables and report
Usage: python scripts/pleural_compare_freeze.py --run runs/E17_pleural_model_comparison_prep_20261123
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TS = ROOT / "models/totalsegmentator/nnunet/results/Dataset315_thoraxCT/nnUNetTrainer__nnUNetPlans__3d_fullres"
TARGETS = ["LUNG1-013", "LUNG1-113", "LUNG1-170", "LUNG1-195", "LUNG1-249", "LUNG1-299"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = ROOT / args.run
    full, fz = run / "full", run / "freeze"
    fz.mkdir(exist_ok=True)
    ma = json.loads((full / "manifest_armA.json").read_text(encoding="utf-8"))
    mb = json.loads((full / "manifest_armB.json").read_text(encoding="utf-8"))
    shutil.copy(full / "manifest_armA.json", fz / "manifest_armA.json")
    shutil.copy(full / "manifest_armB.json", fz / "manifest_armB.json")
    a_pred = {c["case_id"]: c["prediction"] for c in ma["cases"]}
    rows = []
    for c in mb["cases"]:
        cid = c["case_id"]
        r = {"case_id": cid, "set": c["set"], "ct": c["ct"], "ct_sha256": sha(ROOT / c["ct"]),
             "arm_a_prediction": a_pred[cid], "arm_a_prediction_sha256": sha(ROOT / a_pred[cid]),
             "arm_b_prediction": c["prediction"], "arm_b_prediction_sha256": sha(ROOT / c["prediction"]),
             "arm_b_lung_label_sha256": sha((ROOT / c["prediction"]).parent / "lung.nii.gz"),
             "clinician_grade": c.get("clinician_grade") or "", "report_label_pleural_effusion": c.get("report_label_pleural_effusion")}
        for name, rp in (c.get("references") or {}).items():
            r[f"reference_{name}"], r[f"reference_{name}_sha256"] = rp, sha(ROOT / rp)
        rows.append(r)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(fz / "case_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    log = json.loads((full / "run_log_B_task307_pleuralfinal.json").read_text(encoding="utf-8"))
    relog = json.loads((full / "rerun_LUNG1-104" / "run_log_B_task307_pleuralfinal.json").read_text(encoding="utf-8"))
    ident = {
        "arm_a": {"name": "TotalSegmentator task pleural_pericard_effusion (Dataset315_thoraxCT), nnU-Net v2, fold 0",
                  "files_sha256": {p.name if p.parent == TS else f"{p.parent.name}/{p.name}": sha(p)
                                   for p in [TS / "plans.json", TS / "dataset.json", TS / "dataset_fingerprint.json", TS / "fold_0" / "checkpoint_final.pth"] if p.exists()},
                  "class_names_from_dataset_json": json.loads((TS / "dataset.json").read_text(encoding="utf-8")).get("labels")},
        "arm_b": {"name": "Task307_pleuralfinal, nnU-Net v1 1.7.1, nnUNetTrainerV2__nnUNetPlansv2.1, 3d_fullres, five folds",
                  "source": "Zenodo record 21918112; archive hashes in ../armB_checkpoint_provenance.json",
                  "weights_sha256": log["weights_sha256"], "plans_sha256": log["plans_sha256"], "settings": log["settings"], "plans": log["plans"],
                  "packages": log["packages"], "python": log["python"], "gpu": log["gpu"],
                  "class_names": "not present in the archive; label 1 = lung and label 2 = pleural effusion established from the paper's description and from the outputs (full/class_review_summary.json)",
                  "LUNG1-104": {"main_run": log["cases"]["LUNG1-104"]["issue"], "second_run_saved_and_scored": relog["cases"]["LUNG1-104"]}}}
    (fz / "model_identity.json").write_text(json.dumps(ident, indent=1), encoding="utf-8")

    prep = {
        "arm_a_pipeline_as_implemented": {
            "published_procedure": "TotalSegmentator 2.18 run through its own command for task pleural_pericard_effusion; no step of ours inside it",
            "steps": ["a 6 mm TotalSegmentator model finds the lung lobes", "the CT is cropped to the lobes plus 50 mm", "the crop is resampled to 1.5 mm",
                      "nnU-Net v2, fold 0, no test-time mirroring", "the mask is resampled back to the source grid"],
            "ours": "BLAS threads capped at 1 on Windows; nothing else"},
        "arm_b_pipeline_as_implemented": {
            "published_procedure": ["scans downsampled with the authors' own tool before nnU-Net (tool not published)", "nnU-Net v1 command nnUNet_predict, 3d_fullres, five-fold ensemble",
                                    "the paper does not describe a chest crop; the model was trained and tested on chest CT"],
            "ours": ["whole CT as stored is given to nnU-Net; no crop, no filter", "nnU-Net's own preprocessing from the model's plans: resampling to 2.0 mm, CT normalisation",
                     "the authors' downsampling step is NOT reproduced", "single-process driver on Windows calling the same nnU-Net functions in the same order as predict_cases (bit-identity not verifiable)",
                     "five-fold mean softmax, mirroring on, step 0.5, final checkpoints, no postprocessing, nnU-Net's own export to the source grid"]},
        "difference_that_matters": "Arm A sees only the chest box; Arm B sees the whole scan. On scans much longer than the chest Arm B labels tissue outside the chest (full/armB_outside_chest.csv). "
                                   "The comparison is therefore between two implemented inference pipelines, not between two networks in isolation, and not a reproduction of the published Arm B pipeline."}
    (fz / "preprocessing.json").write_text(json.dumps(prep, indent=1), encoding="utf-8")

    lister = "import importlib.metadata as m\nfor x in sorted({d.metadata['Name'] + '==' + d.version for d in m.distributions()}, key=str.lower): print(x)"
    for name, py in (("core", ROOT / ".venv/Scripts/python.exe"), ("segmentation", ROOT / "environments/segmentation/Scripts/python.exe"),
                     ("pleural_v1", ROOT / "environments/pleural_v1/Scripts/python.exe")):
        txt = subprocess.run([str(py), "-c", lister], capture_output=True, text=True).stdout
        ver = subprocess.run([str(py), "--version"], capture_output=True, text=True).stdout.strip()
        (fz / f"environment_{name}.txt").write_text(f"# {ver} (installed distributions; core = metrics and QC, segmentation = Arm A, pleural_v1 = Arm B)\n{txt}", encoding="utf-8")

    # counts recomputed from the per-case table, compared with the stored summary
    P = list(csv.DictReader(open(full / "paired_reference_positive.csv", encoding="utf-8")))
    S = json.loads((full / "comparison_summary.json").read_text(encoding="utf-8"))["plethora_expert_mask_evaluation"]

    def fl(r, k):
        return float(r[k])

    def recompute(rows_, arm):
        d = np.array([fl(r, f"{arm}_dice") for r in rows_])
        e = np.array([fl(r, f"{arm}_volume_error_ml") for r in rows_])
        return {"n": len(rows_), "empty_predictions": sum(fl(r, f"{arm}_ml") == 0 for r in rows_), "dice_median": float(np.median(d)), "dice_mean": float(d.mean()),
                "dice_at_least_0.7": int((d >= 0.7).sum()), "abs_volume_error_ml_median": float(np.median(np.abs(e))), "abs_volume_error_ml_mean": float(np.abs(e).mean()),
                "signed_volume_error_ml_median": float(np.median(e)), "mask_state": dict(collections.Counter(r[f"{arm}_state"] for r in rows_))}
    checks = {}
    for arm in "ab":
        mine, stored = recompute(P, arm), S["all"][f"arm_{arm}"]
        checks[f"arm_{arm}"] = {k: {"recomputed": mine[k], "stored": stored[k], "equal": (mine[k] == stored[k]) if not isinstance(mine[k], float) else abs(mine[k] - stored[k]) < 1e-9}
                                for k in mine if k != "mask_state"}
        checks[f"arm_{arm}"]["mask_state"] = {"recomputed": mine["mask_state"], "stored": {k: v for k, v in stored["mask_state"].items()},
                                              "equal": all(mine["mask_state"].get(k, 0) == v for k, v in stored["mask_state"].items())}
    lab = collections.Counter("improved" if fl(r, "b_dice") - fl(r, "a_dice") >= 0.10 else "worsened" if fl(r, "b_dice") - fl(r, "a_dice") <= -0.10 else "similar" for r in P)
    checks["labels"] = {"recomputed": dict(lab), "stored": S["all"]["label_counts"], "equal": dict(lab) == S["all"]["label_counts"]}
    ea = sorted(r["case_id"] for r in P if fl(r, "a_ml") == 0)
    eb = sorted(r["case_id"] for r in P if fl(r, "b_ml") == 0)
    by = {r["case_id"]: r for r in P}
    no104 = [r for r in P if r["case_id"] != "LUNG1-104"]
    out = {"written_utc": datetime.now(timezone.utc).isoformat(), "primary_cohort_n": len(P), "summary_vs_per_case_table": checks,
           "all_equal": all(v["equal"] for arm in checks.values() for v in (arm.values() if "equal" not in arm else [arm])),
           "empty_predictions": {"arm_a": ea, "arm_b": eb, "target_misses": TARGETS, "arm_a_empties_are_exactly_the_targets": ea == sorted(TARGETS),
                                 "targets_still_empty_in_b": [c for c in TARGETS if c in eb],
                                 "targets_not_empty_in_b_but_not_recovered": [{"case_id": c, "b_ml": fl(by[c], "b_ml"), "b_dice": fl(by[c], "b_dice"), "reference_ml": fl(by[c], "reference_ml"),
                                                                              "b_share_inside_tumour_outline": by[c]["b_share_in_tumour"]} for c in TARGETS if c not in eb],
                                 "newly_empty_in_b": [{"case_id": c, "a_ml": fl(by[c], "a_ml"), "a_dice": fl(by[c], "a_dice"), "reference_ml": fl(by[c], "reference_ml")} for c in eb if c not in ea],
                                 "targets_max_b_dice": max(fl(by[c], "b_dice") for c in TARGETS)},
           "sensitivity_analysis_excluding_LUNG1-104_not_primary": {"arm_a": recompute(no104, "a"), "arm_b": recompute(no104, "b")}}
    (fz / "count_verification.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    keep = [run / "REPORT.md", *sorted(p for p in full.iterdir() if p.is_file() and p.name != "SHA256SUMS.json"), *sorted(p for p in fz.iterdir() if p.is_file() and p.name != "SHA256SUMS.json"),
            *sorted((full / "rerun_LUNG1-104").glob("*.json")), run / "armB_checkpoint_provenance.json", run / "armB_plans_summary.json", run / "armB_archive_listing.json"]
    (fz / "SHA256SUMS.json").write_text(json.dumps({str(p.relative_to(run)).replace("\\", "/"): sha(p) for p in keep if p.exists()}, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "summary_vs_per_case_table"}, indent=1))
    print("unequal:", [(a, k) for a, d in checks.items() for k, v in (d.items() if "equal" not in d else [("", d)]) if not v["equal"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
