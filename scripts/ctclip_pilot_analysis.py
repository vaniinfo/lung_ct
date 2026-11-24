"""Descriptive analysis of the CT-CLIP pilot on local CT-RATE scans (core env). Reads stored scores; runs no model.

Unit: ONE RECONSTRUCTION PER PATIENT (the first case id in sorted order for each patient), unless stated.

Shared score component - exact definition
  X = matrix patients x 18 conditions of one model's scores. Each column has its mean subtracted. s_1 >= s_2 >= ... are the
  singular values of that matrix. "Share of the first component" = s_1^2 / sum_k s_k^2: the fraction of the total variance of
  the 18 scores (summed over conditions) that lies along one direction. It is computed four ways:
    raw            scores as output by the runner (softmax 'present' share, or sigmoid for ClassFine)
    logit          ln(p / (1 - p)); for zero-shot and VocabFine this equals logit_present - logit_not_present, for ClassFine the head's logit
    standardised   each condition's raw scores divided by their standard deviation (z-scores): every condition weighs the same
    std. logit     z-scores of the logits
  A high share means the 18 scores of a patient move together. It is called a SHARED COMPONENT. It is not shown here to
  measure abnormality: the only outside quantity it is compared with is the number of report labels, itself report-derived.

Agreement with report labels: probability that a labelled patient scores above an unlabelled one (rank statistic, ties 0.5).
"Not mentioned" is the comparison group, so this is not accuracy and not a false-positive analysis.
No threshold is chosen anywhere. Usage: python scripts/ctclip_pilot_analysis.py --out runs/E21_ctclip_pilot_20261124
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
RUNS = {"zero-shot": "runs/E21_ctclip_base_85cases_*", "vocabfine": "runs/E21_ctclip_vocabfine_85cases_*", "classfine": "runs/E21_ctclip_classfine_85cases_*"}
COLIPRI_KEY = {"Hiatal hernia": "hiatal_hernia", "Pleural effusion": "pleural_effusion", "Consolidation": "consolidation", "Cardiomegaly": "cardiomegaly",
               "Lung nodule": "lung_nodule", "Emphysema": "emphysema", "Atelectasis": "atelectasis", "Pericardial effusion": "pericardial_effusion",
               "Lymphadenopathy": "lymphadenopathy", "Arterial wall calcification": "arterial_calcification", "Coronary artery wall calcification": "coronary_calcification",
               "Bronchiectasis": "bronchiectasis", "Medical material": "medical_material", "Pulmonary fibrotic sequela": "fibrotic_sequela",
               "Mosaic attenuation pattern": "mosaic_attenuation", "Peribronchial thickening": "peribronchial_thickening"}
READER = {"valid_301_a_2": "present", "valid_541_a_2": "absent", "valid_832_a_2": "present", "valid_988_a_2": "absent"}   # hiatal hernia, the clinician reviewer, open pages


def patient(c: str) -> str:
    return re.match(r"(valid_\d+)_", c).group(1)


def load():
    S, src = {}, {}
    for k, pat in RUNS.items():
        r = Path(sorted(glob.glob(str(ROOT / pat)))[-1])
        m = json.loads((r / "run_manifest.json").read_text(encoding="utf-8"))
        S[k] = {c["case_id"]: c["scores"] for c in m["cases"] if c["status"] == "ok"}
        src[k] = {"run": r.name, "checkpoint_sha256": m["model"]["checkpoint_sha256"], "scans": len(S[k])}
    col = {}
    for f in glob.glob(str(ROOT / "runs/A1_agent_valid_*/evidence.json")):
        e = json.loads(Path(f).read_text(encoding="utf-8"))
        for r in e["rows"]:
            if r.get("group") == "classifier" and r.get("value") is not None and r.get("ckey"):
                col.setdefault(e["case_id"], {})[r["ckey"]] = float(r["value"])
    return S, src, col


def rank_stat(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def pc1_share(X: np.ndarray) -> dict:
    Xc = X - X.mean(0)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    v = s ** 2
    return {"first": round(float(v[0] / v.sum()), 3), "second": round(float(v[1] / v.sum()), 3),
            "loadings_all_same_sign": bool(np.all(vt[0] > 0) or np.all(vt[0] < 0)), "patient_scores": (u[:, 0] * s[0] * (1 if vt[0].sum() > 0 else -1))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    S, src, col = load()
    scans = sorted(S["zero-shot"])
    names = list(S["zero-shot"][scans[0]])
    first = {}
    for c in scans:
        first.setdefault(patient(c), c)
    one = sorted(first.values())
    R = {"unit": "one reconstruction per patient (first case id in sorted order)", "scans_scored": len(scans), "patients": len(one), "sources": src,
         "definition": __doc__.split("Agreement with report labels")[0].strip(), "shared_component": {}, "follows_other_scores": {}, "report_label_rank_statistic": {}}

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    nlab = np.array([len(gt[c]["labels"]) for c in one])
    for k in RUNS:
        X = np.array([[S[k][c][n] for n in names] for c in one])
        L = logit(X)
        forms = {"raw": X, "logit": L, "standardised": (X - X.mean(0)) / X.std(0), "standardised_logit": (L - L.mean(0)) / L.std(0)}
        R["shared_component"][k] = {}
        for fk, M in forms.items():
            r = pc1_share(M)
            ps = r.pop("patient_scores")
            r["spearman_with_number_of_report_labels"] = round(float(spearmanr(ps, nlab)[0]), 3)
            R["shared_component"][k][fk] = r
        Xall = np.array([[S[k][c][n] for n in names] for c in scans])
        R["shared_component"][k]["raw_all_85_scans_for_comparison"] = {kk: v for kk, v in pc1_share(Xall).items() if kk != "patient_scores"}
        rs = {n: float(spearmanr(X[:, i], np.delete(X, i, axis=1).mean(1))[0]) for i, n in enumerate(names)}
        R["follows_other_scores"][k] = {"what": "Spearman of a condition's raw score with the mean of the patient's other 17 raw scores",
                                        "median_over_conditions": round(float(np.median(list(rs.values()))), 3), "min": round(min(rs.values()), 3),
                                        "hiatal_hernia": round(rs["Hiatal hernia"], 3)}
    models = {**{k: {c: S[k][c] for c in one} for k in RUNS}, "COLIPRI": {c: {n: col[c].get(COLIPRI_KEY[n]) for n in names if n in COLIPRI_KEY} for c in one if c in col}}
    for n in names:
        lab = {c: n in gt[c]["labels"] for c in one}
        row = {"patients_labelled": sum(lab.values())}
        for k, sc in models.items():
            cs = [c for c in one if c in sc and sc[c].get(n) is not None]
            p, q = [sc[c][n] for c in cs if lab[c]], [sc[c][n] for c in cs if not lab[c]]
            row[k] = round(rank_stat(p, q), 3) if len(p) >= 5 and len(q) >= 5 else None
        R["report_label_rank_statistic"][n] = row
    R["report_label_rank_statistic_note"] = "None = fewer than 5 labelled or unlabelled patients (hiatal hernia: 3 labelled patients) or no COLIPRI score for the condition"

    hh = []
    for c in one:
        r = {"case_id": c, "patient": patient(c), "report_label_hiatal_hernia_mentioned": int("Hiatal hernia" in gt[c]["labels"]), "reader_grade_open_page": READER.get(c, ""),
             "n_report_labels": len(gt[c]["labels"])}
        for k in RUNS:
            v = np.array([S[k][x]["Hiatal hernia"] for x in one])
            r[f"{k}_score"] = round(S[k][c]["Hiatal hernia"], 4)
            r[f"{k}_percentile"] = int(round(100 * float((v <= S[k][c]["Hiatal hernia"]).mean())))
        cv = (col.get(c) or {}).get("hiatal_hernia")
        r["COLIPRI_score"] = "" if cv is None else round(cv, 4)
        hh.append(r)
    with open(out / "hiatal_hernia_per_patient.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hh[0]))
        w.writeheader()
        w.writerows(hh)
    R["hiatal_hernia"] = {"patients_with_report_label": [r["case_id"] for r in hh if r["report_label_hiatal_hernia_mentioned"]],
                          "patients_with_reader_grade": {r["case_id"]: r["reader_grade_open_page"] for r in hh if r["reader_grade_open_page"]},
                          "rows_of_interest": [r for r in hh if r["report_label_hiatal_hernia_mentioned"] or r["reader_grade_open_page"]],
                          "note": "too few labelled patients for any discrimination or calibration estimate"}
    (out / "pilot_analysis.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(json.dumps({k: R[k] for k in ("scans_scored", "patients", "shared_component", "follows_other_scores", "report_label_rank_statistic")}, indent=1))
    for r in R["hiatal_hernia"]["rows_of_interest"]:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
