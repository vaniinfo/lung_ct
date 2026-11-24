"""Add the pericardial fluid thickness row (E79) to finished agent runs and tabulate it (core env).

E79 = greatest thickness (mm) of the TotalSegmentator pericardial fluid outline (largest sphere that fits inside it).
It was introduced after these runs were made, so it is appended to each run's evidence.json with a provenance
entry under "addenda"; no existing row is changed. Safe to re-run (a run that already has E79 is left alone).
E79 is shown for information and is not used in any checklist decision.

Usage: python scripts/add_pericardial_thickness.py --pages-file runs/test_logs/pages_test20c_v3_1.txt [more files]
           [--config configs/checklist_v3_3.json]
Writes runs/E10_pericardial_thickness_<time>/{thickness.csv,thickness.md}. Re-apply the checklist afterwards with
scripts/rescore_checklist.py so the row appears with the pericardial answer.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.checklist import load_checklist_config  # noqa: E402
from lung_ct.agent.conditions import pericardial_thickness_row, thickness_grade  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pages-file", nargs="+", required=True)
    ap.add_argument("--config", default="configs/checklist_v3_3.json")
    ap.add_argument("--refresh", action="store_true", help="recompute and replace an E79 row that this script added earlier")
    args = ap.parse_args()
    ck = load_checklist_config(ROOT / args.config)
    gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    table, failed = [], []
    for pf in args.pages_file:
        for line in (ROOT / pf).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            run = ROOT / Path(line.strip().replace("\\", "/")).parent
            evp = run / "evidence.json"
            ev = json.loads(evp.read_text(encoding="utf-8"))
            cid = ev["case_id"]
            try:
                rows = {r["id"]: r for r in ev["rows"]}
                added = False
                mine = any(a.get("row") == "E79" for a in ev.get("addenda", []))
                if "E79" in rows and args.refresh and mine:
                    ev["rows"] = [x for x in ev["rows"] if x["id"] != "E79"]
                    del rows["E79"]
                if "E79" not in rows:
                    ct = nib.load(str(ROOT / "data/raw/ct_rate/volumes" / f"{cid}.nii.gz"))
                    mp = ROOT / "data" / "derived" / cid / "segmentation" / "pleural_pericard_effusion" / "pericardial_effusion.nii.gz"
                    pc = None
                    if mp.exists():
                        img = nib.load(str(mp))
                        if img.shape != ct.shape or not np.allclose(img.affine, ct.affine, atol=1e-3):
                            raise ValueError("pericardial mask is not on the CT grid")
                        pc = np.asanyarray(img.dataobj) > 0
                    r = pericardial_thickness_row(pc, ct.affine, ck)
                    at = next((i for i, x in enumerate(ev["rows"]) if x["id"] == "E73"), len(ev["rows"]) - 1) + 1
                    ev["rows"].insert(at, r)
                    ev.setdefault("addenda", []).append({"row": "E79", "added_utc": datetime.now(timezone.utc).isoformat(),
                                                         "by": "scripts/add_pericardial_thickness.py", "config": ck["config_version"],
                                                         "note": "new measurement appended; existing rows unchanged"})
                    evp.write_text(json.dumps(ev, indent=2), encoding="utf-8")
                    rows["E79"], added = r, True
                mm = rows["E79"]["value"]
                table.append({"set": Path(pf).stem.replace("pages_", ""), "case_id": cid, "volume_ml": rows.get("E73", {}).get("value"),
                              "thickness_mm": mm, "grade": thickness_grade(mm), "report_label": "Pericardial effusion" in gt[cid]["labels"],
                              "added_now": added})
            except Exception as e:  # reported, never dropped
                failed.append({"case_id": cid, "error": f"{type(e).__name__}: {e}"})
    out = ROOT / "runs" / f"E10_pericardial_thickness_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    with open(out / "thickness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    table.sort(key=lambda r: (r["thickness_mm"] is None, r["thickness_mm"] or 0))
    md = [f"# Pericardial fluid: volume and greatest thickness ({len(table)} scans, {len(failed)} failed)", "",
          "Thickness = largest sphere inside the automatic outline (provisional, not validated). Report label = CT-RATE, text-derived.", "",
          "| scan | set | volume mL | thickness mm | grade | effusion in report |", "|---|---|---|---|---|---|"]
    md += [f"| {r['case_id']} | {r['set']} | {r['volume_ml']} | {r['thickness_mm']} | {r['grade']} | {'yes' if r['report_label'] else ''} |"
           for r in table]
    md += [f"- FAILED {f['case_id']}: {f['error']}" for f in failed]
    (out / "thickness.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
