"""Compare airway measurements between labelled bronchiectasis cases and controls (core env). Descriptive only.

This is NOT a classifier and reports NO classification performance. It describes how exploratory measurements
(scripts/airway_measure.py) differ between two labelled groups on the development splits. No threshold is chosen.
If a classifier is added later it belongs in its own stage, with its own metrics, labelled exploratory until experts
have validated the labels and the measurements.

Leakage rules (lung_ct.airway.cohort)
  * splits are made per PATIENT (all scans / reconstructions of a patient stay together) and written once to --splits;
    an existing file is read and never changed
  * default: only the train split is analysed. --use train validation adds validation
  * the test split is refused unless --frozen-plan <file> is given: a written analysis plan (features, direction,
    any threshold chosen on train / validation). Its SHA-256 and time stamp are recorded with the result
  * scans of one patient are combined (median) before comparison, so a patient counts once

Labels file (CSV): case_id,label[,patient_id][,label_source]   label 1 = bronchiectasis, 0 = control.
  State the label source honestly. CT-RATE labels are derived from report text (not an expert reading of the images);
  --labels-from-ctrate writes such a file from data/raw/ct_rate/cases.json and marks the source.

Usage:
    python scripts/airway_group_compare.py --labels-from-ctrate runs/airway_labels_ctrate.csv            # optional helper
    python scripts/airway_group_compare.py --measure-run runs/P6_airway_measures_<ts> --labels <labels.csv> \
        --splits runs/airway_splits_seed20261123.json --seed 20261123 [--use train validation]
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from lung_ct.airway import cohort  # noqa: E402

CTRATE_SOURCE = "CT-RATE report-derived label (text classifier on the radiology report; not an expert image annotation; 'absent' = not mentioned)"


def labels_from_ctrate(out: Path) -> int:
    cases = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "label", "patient_id", "label_source"])
        for cid in sorted(cases):
            w.writerow([cid, int("Bronchiectasis" in cases[cid]["labels"]), cohort.patient_id_from_case(cid), CTRATE_SOURCE])
    print(f"{len(cases)} cases -> {out}  (label source: report text, not expert image annotation)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels-from-ctrate", type=Path, help="write a labels CSV from the local CT-RATE case file and stop")
    ap.add_argument("--measure-run", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--splits", type=Path, help="JSON with the patient-level splits; created if absent, never changed")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.6, 0.2, 0.2], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--use", nargs="+", choices=cohort.SPLITS, default=["train"])
    ap.add_argument("--frozen-plan", type=Path, help="required to read the test split: the analysis plan written before looking at it")
    args = ap.parse_args()
    if args.labels_from_ctrate:
        return labels_from_ctrate(ROOT / args.labels_from_ctrate)
    if not (args.measure_run and args.labels and args.splits):
        ap.error("--measure-run, --labels and --splits are required")
    if "test" in args.use and not args.frozen_plan:
        ap.error("the test split is locked: pass --frozen-plan <file> (analysis plan fixed on train / validation)")
    plan = None
    if args.frozen_plan:
        pp = ROOT / args.frozen_plan
        plan = {"file": str(args.frozen_plan), "sha256": hashlib.sha256(pp.read_bytes()).hexdigest()}

    with open(ROOT / args.labels, newline="", encoding="utf-8-sig") as f:
        lab_rows = list(csv.DictReader(f))
    for r in lab_rows:
        r["patient_id"] = r.get("patient_id") or cohort.patient_id_from_case(r["case_id"])
    sources = sorted({r.get("label_source") or "not stated" for r in lab_rows})
    pat_label, conflict = cohort.patient_labels(lab_rows)

    sp = ROOT / args.splits
    if sp.exists():
        saved = json.loads(sp.read_text(encoding="utf-8"))
        splits = saved["patient_split"]
        missing = sorted(set(pat_label) - set(splits))
        if missing:
            raise SystemExit(f"{len(missing)} labelled patients are not in the existing split file {args.splits} (first: {missing[:3]}). "
                             "The file is never changed: make a new split file for a new cohort.")
    else:
        splits = cohort.assign_splits(pat_label, args.seed, tuple(args.fractions))
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "fractions": args.fractions,
                                  "unit": "patient", "labels_file": str(args.labels), "label_sources": sources,
                                  "excluded_conflicting_labels": conflict, "patient_split": splits}, indent=1), encoding="utf-8")
    case_patient = {r["case_id"]: r["patient_id"] for r in lab_rows if r["patient_id"] in pat_label}
    cohort.check_no_leakage({c: splits[p] for c, p in case_patient.items()}, case_patient)

    with open(ROOT / args.measure_run / "case_features.csv", newline="", encoding="utf-8") as f:
        feats = list(csv.DictReader(f))
    with open(ROOT / args.measure_run / "summary.csv", newline="", encoding="utf-8") as f:
        quality = {r["case_id"]: r for r in csv.DictReader(f)}
    measured = {r["case_id"] for r in feats}
    use = set(args.use)
    # patient -> feature -> values over that patient's scans (combined by median)
    per = {}
    for r in feats:
        cid = r["case_id"]
        if cid not in case_patient or splits[case_patient[cid]] not in use or r["value"] in ("", None):
            continue
        per.setdefault(case_patient[cid], {}).setdefault((r["feature"], r["group"]), []).append(float(r["value"]))
    pats = {p for p in pat_label if splits.get(p) in use}
    rows = []
    for key in sorted({k for d in per.values() for k in d}):
        a = [float(np.median(per[p][key])) for p in pats if pat_label[p] == 1 and key in per.get(p, {})]
        b = [float(np.median(per[p][key])) for p in pats if pat_label[p] == 0 and key in per.get(p, {})]
        rows.append({"feature": key[0], "group": key[1], **cohort.compare_groups(a, b)})

    run_dir = ROOT / "runs" / f"E20_airway_group_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True)
    with open(run_dir / "group_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["feature"])
        w.writeheader()
        w.writerows(rows)
    n = {s: {"patients": sum(1 for p in pat_label if splits.get(p) == s), "labelled_bronchiectasis": sum(1 for p in pat_label if splits.get(p) == s and pat_label[p] == 1)}
         for s in cohort.SPLITS}
    summ = {"created_utc": datetime.now(timezone.utc).isoformat(), "what": "descriptive comparison of exploratory airway measurements between labelled groups; "
            "NOT classification performance; no threshold chosen", "measure_run": str(args.measure_run), "labels_file": str(args.labels), "label_sources": sources,
            "splits_file": str(args.splits), "splits_used": sorted(use), "test_split_read": "test" in use, "frozen_plan": plan, "patients_per_split": n,
            "patients_excluded_conflicting_labels": conflict, "labelled_cases_without_measurements": sorted(set(case_patient) - measured),
            "cases_with_quality_flags": {c: quality[c]["case_flags"] for c in sorted(quality) if c in case_patient and int(quality[c]["n_case_flags"] or 0) > 0},
            "features_compared": len(rows), "multiple_comparisons": "p values are uncorrected; with this many features some small p values are expected by chance",
            "limits": ["measurements are unvalidated and come from a predicted airway mask", "labels are only as good as their source (see label_sources)",
                       "fewer or shorter detected branches do not indicate airway loss", "group size decides what can be concluded; small groups: pipeline check only"]}
    (run_dir / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("splits_used", "patients_per_split", "features_compared", "label_sources")}, indent=1))
    print("->", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
