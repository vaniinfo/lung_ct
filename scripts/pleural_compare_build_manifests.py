"""Build the fixed case lists for the pleural-effusion model comparison (core env). Reads only.

Sets (fixed in docs/PLEURAL_MODEL_COMPARISON_PLAN.md before any Arm B result):
  plethora_effusion           77 scans, expert outlines (first = official; second / third kept separate)
  plethora_coded_no_effusion  78 scans, no expert outline, NOT confirmed negatives
  ctrate_development          58 development runs, no outline; report label for all, clinician grade for 4
Usage: python scripts/pleural_compare_build_manifests.py --out runs/E17_.../full
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = "data/raw/plethora"
GRADE = {"valid_301_a_2": "present", "valid_541_a_2": "absent", "valid_832_a_2": "present", "valid_988_a_2": "absent"}   # development review, v3_2 / v3_3 pages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    base = []
    for c in json.loads((ROOT / P / "selection.json").read_text(encoding="utf-8"))["cases"]:
        pid = c["patient_id"]
        if pid == "LUNG1-095":      # excluded since E14: non-uniform slice spacing in the DICOM series
            continue
        d = ROOT / P / "effusions" / "Effusions" / pid
        refs = {f.name.split("_effusion_")[1].replace("_reviewer.nii.gz", ""): f.relative_to(ROOT).as_posix() for f in sorted(d.glob("*.nii.gz"))} if d.exists() else {}
        g = ROOT / P / "gtv" / f"{pid}_gtv.nii.gz"
        base.append({"case_id": pid, "set": "plethora_effusion" if refs else "plethora_coded_no_effusion", "ct": f"{P}/volumes/{pid}.nii.gz", "references": refs,
                     "gtv": g.relative_to(ROOT).as_posix() if g.exists() else None, "clinician_grade": None, "report_label_pleural_effusion": None, "plethora": True})
    gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    for line in (ROOT / "runs/test_logs/pages_58runs_with_v3_4.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = ROOT / Path(line.strip().replace("\\", "/")).parent
        ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
        cid = ev["case_id"]
        base.append({"case_id": cid, "set": "ctrate_development", "ct": ev["inputs"]["ct"].replace("\\", "/"), "references": {}, "gtv": None,
                     "clinician_grade": GRADE.get(cid), "report_label_pleural_effusion": "Pleural effusion" in gt[cid]["labels"], "plethora": False, "run": run.name})
    ids = [c["case_id"] for c in base]
    assert len(ids) == len(set(ids)), "duplicate case ids"

    def pred(c, arm):
        if arm == "B":
            return f"data/derived/pleural_armB/{c['case_id']}/pleural_effusion.nii.gz"
        return (f"data/derived/plethora/{c['case_id']}/pleural_pericard_effusion/pleural_effusion.nii.gz" if c["plethora"]
                else f"data/derived/{c['case_id']}/segmentation/pleural_pericard_effusion/pleural_effusion.nii.gz")
    for arm, name in (("A", "A_totalsegmentator_315"), ("B", "B_task307_pleuralfinal")):
        cases = [{**{k: v for k, v in c.items() if k != "plethora"}, "prediction": pred(c, arm)} for c in base]
        (out / f"manifest_arm{arm}.json").write_text(json.dumps({"arm": name, "cases": cases}, indent=1), encoding="utf-8")
    import collections
    print(collections.Counter(c["set"] for c in base), "missing CT:", [c["case_id"] for c in base if not (ROOT / c["ct"]).exists()],
          "missing Arm A mask:", [c["case_id"] for c in base if not (ROOT / pred(c, "A")).exists()])
    return 0


if __name__ == "__main__":
    sys.exit(main())
