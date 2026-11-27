"""Add the per-lobe distribution rows (E83, E89, E91) to finished agent runs, and prove nothing else moved (core env).

For each run the checklist measurements are recomputed from the CT and the stored segmentation with the same code
(conditions.measure_conditions) and the given config. Every recomputed row that already exists in evidence.json must
have exactly the stored value; if any differs, the run is reported as FAILED and its evidence.json is not touched.
Only then are the three new rows appended, with a provenance entry under "addenda". No existing row is changed.
Safe to re-run (a run that already has the rows is verified again and otherwise left alone; --refresh recomputes them).

Usage: python scripts/add_lobe_distribution.py --pages-file runs/test_logs/pages_test20b_v3.txt [more files]
           [--config configs/checklist_v3_4.json] [--cases valid_301_a_2 ...]
Writes runs/E12_lobe_distribution_<time>/{verification.json,verification.md,distribution.csv}.
Re-apply the checklist afterwards with scripts/rescore_checklist.py --config configs/checklist_v3_4.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.checklist import load_checklist_config  # noqa: E402
from lung_ct.agent.conditions import image_noise, measure_conditions  # noqa: E402
from lung_ct.agent.evidence import LOBES  # noqa: E402

NEW = {"E83": "E82", "E89": "E87", "E91": "E88"}  # new row -> existing row it is inserted after


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pages-file", nargs="+", required=True)
    ap.add_argument("--config", default="configs/checklist_v3_4.json")
    ap.add_argument("--cases", nargs="*", default=None, help="only these case ids")
    ap.add_argument("--refresh", action="store_true", help="recompute rows this script added earlier")
    args = ap.parse_args()
    ck = load_checklist_config(ROOT / args.config)
    if not ck["measurements"].get("lobe_distribution"):
        raise SystemExit(f"{args.config} has no measurements.lobe_distribution")
    out = ROOT / "runs" / f"E12_lobe_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    recs, table = [], []
    for pf in args.pages_file:
        for line in (ROOT / pf).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            run = ROOT / Path(line.strip().replace("\\", "/")).parent
            evp = run / "evidence.json"
            ev = json.loads(evp.read_text(encoding="utf-8"))
            cid = ev["case_id"]
            if args.cases and cid not in args.cases:
                continue
            rec = {"case_id": cid, "run": run.name, "status": "ok", "measured_with": ev.get("checklist_config")}
            t0 = time.time()
            try:
                mine = {a["row"] for a in ev.get("addenda", []) if a.get("by") == "scripts/add_lobe_distribution.py"}
                if args.refresh and mine:
                    ev["rows"] = [r for r in ev["rows"] if r["id"] not in mine]
                    ev["addenda"] = [a for a in ev["addenda"] if a.get("by") != "scripts/add_lobe_distribution.py"]
                stored = {r["id"]: r for r in ev["rows"]}
                old_hash = sha([r for r in ev["rows"] if r["id"] not in NEW])
                qc = json.loads((run / "qc.json").read_text(encoding="utf-8"))
                ct = nib.load(str(ROOT / qc["ct"]["file"].replace("\\", "/")))
                hu = np.asanyarray(ct.dataobj).astype(np.float32)
                seg = ROOT / "data" / "derived" / cid / "segmentation"
                lobes = {}
                for n in LOBES:
                    img = nib.load(str(seg / "total" / f"{n}.nii.gz"))
                    if img.shape != ct.shape or not np.allclose(img.affine, ct.affine, atol=1e-3):
                        raise ValueError(f"{n} is not on the CT grid")
                    lobes[n] = np.asanyarray(img.dataobj) > 0
                lab = np.zeros(hu.shape, np.uint8)
                for n, (_, side_label, _) in LOBES.items():
                    lab[lobes[n]] = side_label
                hu_thr, _, noise = image_noise(hu, ct, seg, ck)
                rows, _ = measure_conditions(hu_thr, ct, lab, lobes, {n: v[2] for n, v in LOBES.items()}, seg, ck, noise)
                new = {r["id"]: r for r in rows}
                # 1. every existing measurement must be reproduced exactly
                common = [i for i in new if i in stored and i not in NEW]
                diff = [{"id": i, "stored": stored[i]["value"], "recomputed": new[i]["value"]} for i in common
                        if stored[i]["value"] != new[i]["value"]]
                rec.update(rows_compared=len(common), rows_different=len(diff), differences=diff)
                if diff:
                    raise ValueError(f"{len(diff)} existing rows are not reproduced: {[d['id'] for d in diff]}")
                # 2. the distribution must add up to the stored region volumes (same mask)
                for nid, vid in (("E83", "E80"), ("E89", "E85"), ("E91", "E88")):
                    v = new.get(nid, {}).get("value")
                    if isinstance(v, dict) and stored.get(vid, {}).get("value") is not None:
                        if abs(v["region_ml"] - stored[vid]["value"]) > 0.51:  # stored value is rounded to whole mL
                            raise ValueError(f"{nid} region {v['region_ml']} mL does not match {vid} {stored[vid]['value']} mL")
                        tot = sum(d["voxels"] for d in v["lobes"].values())
                        if tot > v["region_voxels"]:
                            raise ValueError(f"{nid}: lobes hold more voxels than the region")
                added = []
                for nid, after in NEW.items():
                    if nid in new and nid not in stored:
                        at = next((k for k, r in enumerate(ev["rows"]) if r["id"] == after), len(ev["rows"]) - 1) + 1
                        ev["rows"].insert(at, new[nid])
                        ev.setdefault("addenda", []).append({
                            "row": nid, "added_utc": datetime.now(timezone.utc).isoformat(), "by": "scripts/add_lobe_distribution.py",
                            "config": ck["config_version"], "note": "new measurement appended after verifying that every existing "
                                                                    "row is reproduced exactly; existing rows unchanged"})
                        added.append(nid)
                if sha([r for r in ev["rows"] if r["id"] not in NEW]) != old_hash:
                    raise RuntimeError("existing rows changed while appending")  # cannot happen; guard
                if added:
                    evp.write_text(json.dumps(ev, indent=2), encoding="utf-8")
                rec.update(added=added, existing_rows_sha256=old_hash)
                for nid in NEW:
                    v = {r["id"]: r for r in ev["rows"]}.get(nid, {}).get("value")
                    if isinstance(v, dict):
                        for lobe, d in v["lobes"].items():
                            table.append({"case_id": cid, "row": nid, "region_ml": v["region_ml"], "lobe": lobe, "ml": d["ml"],
                                          "pct_of_lobe": d["pct_of_lobe"], "pct_of_region": d["pct_of_region"],
                                          "near_fissure_pct_of_lobe_part": d.get("near_fissure_pct_of_lobe_part"),
                                          "region_near_fissure_pct": v["near_fissure_pct_of_region"],
                                          "region_outside_lobes_pct": v["outside_lobes_pct_of_region"]})
            except Exception as e:  # reported, never dropped
                rec.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            rec["seconds"] = round(time.time() - t0, 1)
            recs.append(rec)
            print(f"[{cid}] {rec['status']} compared {rec.get('rows_compared')} different {rec.get('rows_different')} "
                  f"added {rec.get('added')} {rec['seconds']}s {rec.get('error', '')}", flush=True)
    ok = [r for r in recs if r["status"] == "ok"]
    summary = {"config": ck["config_version"], "runs": len(recs), "ok": len(ok), "failed": len(recs) - len(ok),
               "existing_rows_compared": sum(r.get("rows_compared", 0) for r in recs),
               "existing_rows_different": sum(r.get("rows_different", 0) for r in recs), "records": recs}
    (out / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if table:
        with open(out / "distribution.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
    md = [f"# Per-lobe distribution rows added ({summary['ok']} of {summary['runs']} runs ok, {summary['failed']} failed)", "",
          f"Existing measurement rows recomputed and compared: {summary['existing_rows_compared']}; different: "
          f"{summary['existing_rows_different']}.", ""]
    md += [f"- FAILED {r['case_id']}: {r['error']}" for r in recs if r["status"] != "ok"]
    (out / "verification.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 0 if len(ok) == len(recs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
