"""Compare AI-reader checklist answers with CT-RATE report labels, reader by reader (core env).

For reader runs laid out like scripts/run_medgemma.py / run_hulumed.py / run_nvreason.py with the
checklist question (<run>/<case_id>/vlm_parsed.json, schema checklist_v1).

CT-RATE labels come from the report text, so "not labelled" means "not mentioned in the report",
not "proven absent". Per reader and condition:
  hit      reader present, labelled          miss      reader absent, labelled
  extra    reader present, not labelled      agree_no  reader absent, not labelled
  open     reader said cannot tell           no_output no valid answer for that case
Only cases present in EVERY run are scored; a case without a valid answer counts as no_output.
This is a description of development scans. It sets no threshold and changes no checklist rule.

Usage:
  python scripts/score_reader_checklists.py --run medgemma=runs/<P3_vlm...> --run hulumed=runs/<P3_hulumed...> \
        --run nvreason=runs/<P3_nvreason...> [--label test20]
Writes runs/E23_reader_label_scores_<time>/{scores.json,scores.md,per_case.csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.vlm.checklist_schema import READER_CONDITIONS  # noqa: E402

LABEL = {  # reader condition -> CT-RATE label (ground_glass has no label of its own)
    "consolidation": "Consolidation", "atelectasis": "Atelectasis", "lung_nodule": "Lung nodule",
    "emphysema": "Emphysema", "bronchiectasis": "Bronchiectasis",
    "peribronchial_thickening": "Peribronchial thickening",
    "interlobular_septal_thickening": "Interlobular septal thickening",
    "mosaic_attenuation": "Mosaic attenuation pattern", "fibrotic_sequela": "Pulmonary fibrotic sequela",
    "pleural_effusion": "Pleural effusion", "pericardial_effusion": "Pericardial effusion",
    "cardiomegaly": "Cardiomegaly", "arterial_calcification": "Arterial wall calcification",
    "coronary_calcification": "Coronary artery wall calcification", "lymphadenopathy": "Lymphadenopathy",
    "hiatal_hernia": "Hiatal hernia", "medical_material": "Medical material",
}
OUTCOMES = ("hit", "miss", "extra", "agree_no", "open", "no_output")


def outcome(status: str | None, labelled: bool) -> str:
    if status is None:
        return "no_output"
    if status == "cannot_tell":
        return "open"
    if status == "present":
        return "hit" if labelled else "extra"
    return "miss" if labelled else "agree_no"


def score(runs: dict[str, dict[str, dict | None]], labels: dict[str, set]) -> dict:
    """runs: name -> case -> parsed checklist or None; labels: case -> set of CT-RATE labels."""
    cases = sorted(set.intersection(*(set(r) for r in runs.values())) & set(labels))
    out = {"cases": cases, "labelled_scans": {k: sum(lab in labels[c] for c in cases) for k, lab in LABEL.items()},
           "readers": {}, "per_case": []}
    for name, run in runs.items():
        per = {k: Counter() for k in LABEL}
        n_present = 0
        for cid in cases:
            parsed = run[cid]
            for k, lab in LABEL.items():
                st = parsed["conditions"][k]["status"] if parsed else None
                o = outcome(st, lab in labels[cid])
                per[k][o] += 1
                out["per_case"].append({"reader": name, "case_id": cid, "condition": k, "status": st or "",
                                        "labelled": lab in labels[cid], "outcome": o})
            if parsed:
                n_present += sum(v["status"] == "present" for k, v in parsed["conditions"].items() if k in LABEL)
        total = Counter()
        for c in per.values():
            total.update(c)
        lab_n = total["hit"] + total["miss"]
        said = total["hit"] + total["extra"]
        out["readers"][name] = {
            "per_condition": {k: {o: per[k][o] for o in OUTCOMES} for k in LABEL},
            "total": {o: total[o] for o in OUTCOMES},
            "labelled_findings": lab_n,
            "found_of_labelled": f"{total['hit']} of {lab_n}",
            "labelled_of_said_present": f"{total['hit']} of {said}",
            "cases_with_valid_answer": sum(run[c] is not None for c in cases),
            "cases_all_absent": sum(run[c] is not None and all(v["status"] == "absent" for v in run[c]["conditions"].values())
                                    for c in cases),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="name=runs/<reader run>")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    runs, sources = {}, {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        d = ROOT / path
        man = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
        sources[name] = {"run": path, "config": man["provenance"].get("config_file")}
        runs[name] = {}
        for c in man["cases"]:
            f = d / c["case_id"] / "vlm_parsed.json"
            p = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
            runs[name][c["case_id"]] = p if p and set(p.get("conditions", {})) == set(READER_CONDITIONS) else None
    res = score(runs, {c: set(v["labels"]) for c, v in gt.items()})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "runs" / f"E23_reader_label_scores_{args.label + '_' if args.label else ''}{stamp}"
    out.mkdir(parents=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    res["provenance"] = {"created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "sources": sources,
                         "note": "development scans; labels are report mentions; no threshold or rule is derived from this"}
    per_case = res.pop("per_case")
    (out / "scores.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    with open(out / "per_case.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_case[0]))
        w.writeheader()
        w.writerows(per_case)
    names = list(res["readers"])
    md = [f"# AI readers against CT-RATE report labels ({len(res['cases'])} scans, 17 labelled conditions)", "",
          "Labels are report mentions: 'extra' can be a real finding the report did not mention. Development scans.", "",
          "| Reader | Found of labelled | Labelled of said present | Extra | Open | Scans all absent | Valid answers |",
          "|---|---|---|---|---|---|---|"]
    for n in names:
        r = res["readers"][n]
        md.append(f"| {n} | {r['found_of_labelled']} | {r['labelled_of_said_present']} | {r['total']['extra']} | "
                  f"{r['total']['open']} | {r['cases_all_absent']} | {r['cases_with_valid_answer']} of {len(res['cases'])} |")
    md += ["", "## Per condition (hit / miss / extra)", "", "| Condition | Labelled | " + " | ".join(names) + " |",
           "|---|---|" + "---|" * len(names)]
    for k in LABEL:
        cells = [f"{res['readers'][n]['per_condition'][k]['hit']} / {res['readers'][n]['per_condition'][k]['miss']} / "
                 f"{res['readers'][n]['per_condition'][k]['extra']}" for n in names]
        md.append(f"| {k} | {res['labelled_scans'][k]} | " + " | ".join(cells) + " |")
    (out / "scores.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
