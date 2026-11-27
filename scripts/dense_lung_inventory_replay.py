"""Replay dense_lung_inventory_v2 / checklist_v3_6 on finished agent runs (core env). Development cohort only.

For every run: rebuild the consolidation-range region stage by stage from the stored CT and segmentations, check that
the historical baseline values (E85, E88) are reproduced exactly, then compute the retained inventory rows E92-E94
with the SAME function new runs use (lung_ct.agent.conditions.dense_lung_inventory_rows).

Nothing historical is rewritten: evidence.json and earlier checklist files stay as they are. Written per run:
  <replay>/evidence_rows/<case>.json            the new rows E92-E94 (+ E88 with its neutral v3_6 name)
  <run>/checklist_checklist_v3_6.json           additive file, built from the stored evidence + those rows
HoloCT statements are not read here: model interpretations stay separate from measurements.

Usage: python scripts/dense_lung_inventory_replay.py --pages-file runs/test_logs/pages_58runs_with_v3_4.txt
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dev_audit_density import Case  # noqa: E402
from lung_ct.agent.checklist import build_checklist, load_checklist_config  # noqa: E402
from lung_ct.agent.conditions import (DENSE_TAGS, E88_NAME_NEUTRAL, E88_NOTE_NEUTRAL, blocked, dense_lung_inventory_rows,  # noqa: E402
                                      fissure_distance_mm)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pages-file", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--part", default=None, metavar="I/N", help="work on every N-th run starting at I (parallel workers, same --out); the summary is "
                                                                "written by a final call without --part once every run has a record")
    args = ap.parse_args()
    ck5 = load_checklist_config(ROOT / "configs" / "checklist_v3_5.json")
    ck6 = load_checklist_config(ROOT / "configs" / "checklist_v3_6.json")
    m = ck6["measurements"]
    assert {k: v for k, v in m.items() if k != "dense_lung_inventory"} == ck5["measurements"], "v3_6 must not change any measurement setting"
    out = ROOT / (args.out or f"runs/E15_dense_lung_inventory_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    (out / "evidence_rows").mkdir(parents=True, exist_ok=True)
    (out / "records").mkdir(exist_ok=True)
    runs = [ROOT / Path(x.strip().replace("\\", "/")).parent for x in (ROOT / args.pages_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    table, trans, changed = [], collections.Counter(), []
    part = [int(x) for x in args.part.split("/")] if args.part else None
    for k, run in enumerate(runs, 1):
        if part and (k - 1) % part[1] != part[0]:
            continue
        ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
        cid = ev["case_id"]
        rp = out / "records" / f"{run.name}.json"
        if rp.exists():
            table.append(json.loads(rp.read_text(encoding="utf-8")))
            continue
        rec = {"case_id": cid, "run": run.name, "status": "ok", "issues": ""}
        try:
            c = Case(cid, run, ck6)
            stored85, stored88 = c.rows.get("E85", {}).get("value"), c.rows.get("E88", {}).get("value")
            rec.update(baseline_E85_ml=stored85, fluid_adjacent_E88_ml=stored88, posterior_band_E90_ml=c.rows.get("E90", {}).get("value"),
                       pleural_fluid_estimate_ml=round((c.rows.get("E71", {}).get("value") or 0) + (c.rows.get("E72", {}).get("value") or 0), 1))
            if blocked(c.noise, "consolidation") or stored85 is None:
                raise RuntimeError("consolidation range not measured in the baseline (image too grainy or value missing); inventory not computed")
            st = c.stages("consolidation")
            d = st[3]
            near = c.near_eff if c.near_eff is not None else np.zeros(d.shape, bool)
            if round(float(st[5].sum()) * c.vml) != stored85:
                raise RuntimeError(f"baseline E85 not reproduced ({round(float(st[5].sum()) * c.vml)} vs stored {stored85} mL)")
            if stored88 is not None and round(float((d & near).sum()) * c.vml) != stored88:
                raise RuntimeError(f"baseline E88 not reproduced ({round(float((d & near).sum()) * c.vml)} vs stored {stored88} mL)")
            ld = m.get("lobe_distribution")
            fis = fissure_distance_mm(c.lobe_idx, c.short, c.zr) if ld else None
            new = dense_lung_inventory_rows(d, c.near_eff, c.dep, c.lobe_idx, c.short, c.vml, fis, m["dense_lung_inventory"], m["consolidation_hu"],
                                            m.get("effusion_adjacent_mm"), m["dependent_band_fraction"], ld)
            val = new[2]["value"]
            assert np.array_equal(d & ~near & ~c.dep, st[5]), "untagged part must equal the baseline region"
            ev6 = json.loads(json.dumps(ev))
            for r in ev6["rows"]:
                if r["id"] == "E88":
                    r["name"], r["note"] = E88_NAME_NEUTRAL, E88_NOTE_NEUTRAL
            e88 = [r for r in ev6["rows"] if r["id"] == "E88"]
            ev6["rows"] += new
            (out / "evidence_rows" / f"{cid}.json").write_text(json.dumps({"case_id": cid, "run": run.name, "evidence_sha256": hashlib.sha256((run / "evidence.json").read_bytes()).hexdigest(),
                                                                           "rows": e88 + new}, indent=1), encoding="utf-8")
            rd = json.loads((run / "readers.json").read_text(encoding="utf-8")).get("parsed", {})
            cl = ev.get("classifier") or {}
            a = build_checklist(ev, rd, ck5, cl.get("high_threshold", 0.7), cl.get("low_threshold", 0.3))
            b = build_checklist(ev6, rd, ck6, cl.get("high_threshold", 0.7), cl.get("low_threshold", 0.3))
            b["rescored_from"] = f"evidence.json (unchanged) + rows E92-E94 from {out.relative_to(ROOT).as_posix()}/evidence_rows/{cid}.json"
            (run / "checklist_checklist_v3_6.json").write_text(json.dumps(b, indent=2), encoding="utf-8")
            ia, ib = {i["key"]: i for i in a["items"]}, {i["key"]: i for i in b["items"]}
            rec["_pairs"] = {key: [ia[key]["status"], ib[key]["status"], ia[key]["basis"]] for key in ia}
            t = val["tags"]
            rec.update(inventory_total_E92_ml=val["total_ml"], **{f"tag_{x}_ml": t[x]["ml"] for x in DENSE_TAGS}, **{f"tag_{x}_pieces": t[x]["pieces"] for x in DENSE_TAGS},
                       outside_baseline_ml=round(sum(t[x]["ml"] for x in DENSE_TAGS[1:]), 1),
                       **{f"lobe_{n}_ml": round(sum(t[x]["per_lobe_ml"][n] for x in DENSE_TAGS), 1) for n in c.short},
                       consolidation_v3_5=ia["consolidation"]["status"], consolidation_v3_6=ib["consolidation"]["status"],
                       atelectasis_v3_5=ia["atelectasis"]["status"], atelectasis_v3_6=ib["atelectasis"]["status"],
                       pleural_effusion_status=ib["pleural_effusion"]["status"], pleural_effusion_mask_state=ib["pleural_effusion"].get("mask_state"),
                       pleural_effusion_display_v3_5=ia["pleural_effusion"].get("display"), pleural_effusion_display_v3_6=ib["pleural_effusion"].get("display"))
            del c, d, near, st, fis
        except Exception as e:  # noqa: BLE001  reported, never dropped
            rec.update(status="not computed", issues=f"{type(e).__name__}: {e}")
        rp.write_text(json.dumps(rec), encoding="utf-8")
        table.append(rec)
        print(f"[{k}/{len(runs)}] {cid} {rec['status']} baseline {rec.get('baseline_E85_ml')} -> retained {rec.get('inventory_total_E92_ml')} mL {rec['issues']}", flush=True)
    if part or len(table) != len(runs):
        print(f"part done: {len(table)} of {len(runs)} runs have a record; run again without --part to write the summary")
        return 0
    for r in table:
        for key, (a_, b_, basis) in (r.pop("_pairs", None) or {}).items():
            trans[(key, a_, b_)] += 1
            if a_ != b_:
                changed.append({"case_id": r["case_id"], "condition": key, "v3_5": a_, "v3_6": b_, "v3_5_basis": basis})
    cols = list(dict.fromkeys(kk for r in table for kk in r))
    with open(out / "replay_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    ok = [r for r in table if r["status"] == "ok"]

    def n_at(lim, rows=ok):
        return sum(r["outside_baseline_ml"] >= lim for r in rows)
    base_abs = [r for r in ok if r["consolidation_v3_5"] == "absent"]
    summ = {"created_utc": datetime.now(timezone.utc).isoformat(), "inventory": "dense_lung_inventory_v2", "config": "checklist_v3_6", "compared_with": "checklist_v3_5",
            "cohort": "existing development runs (open pages, not blinded); descriptive replay, no accuracy claim",
            "runs": len(table), "computed": len(ok), "not_computed": [{"case_id": r["case_id"], "why": r["issues"]} for r in table if r["status"] != "ok"],
            "baseline_reproduced_on_all_computed_runs": True,
            "volumes_ml_sum_over_runs": {"baseline_E85": round(sum(r["baseline_E85_ml"] for r in ok), 1), "retained_inventory_E92": round(sum(r["inventory_total_E92_ml"] for r in ok), 1),
                                         **{f"tag_{x}": round(sum(r[f"tag_{x}_ml"] for r in ok), 1) for x in DENSE_TAGS}},
            "pieces_sum_over_runs": {x: sum(r[f"tag_{x}_pieces"] for r in ok) for x in DENSE_TAGS},
            "runs_with_candidates_outside_baseline": {"more_than_0_mL": sum(r["outside_baseline_ml"] > 0 for r in ok), "at_least_1_mL": n_at(1), "at_least_5_mL": n_at(5),
                                                      "at_least_30_mL": n_at(30), "note": "descriptive cut points only; none is a decision limit"},
            "runs_answered_no_consolidation_by_the_baseline": {"n": len(base_abs), "with_at_least_5_mL_outside_baseline": n_at(5, base_abs),
                                                               "with_at_least_30_mL_outside_baseline": n_at(30, base_abs)},
            "status_changes_v3_5_to_v3_6": [{"condition": k[0], "from": k[1], "to": k[2], "runs": v} for k, v in sorted(trans.items()) if k[1] != k[2]],
            "status_changes_detail": changed,
            "pleural_effusion_mask_state": dict(collections.Counter(str(r["pleural_effusion_mask_state"]) for r in ok)),
            "pleural_effusion_display": dict(collections.Counter(f"{r['pleural_effusion_display_v3_5']} -> {r['pleural_effusion_display_v3_6']}" for r in ok))}
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "status_changes_detail"}, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
