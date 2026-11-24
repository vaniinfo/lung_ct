"""Re-score finished runs with a newer checklist config, reusing the dense-lung inventory rows of an E15 replay (core env).

No image is read and nothing historical is rewritten: stored evidence.json + the rows E92-E94 already computed by
scripts/dense_lung_inventory_replay.py -> an additional <run>/checklist_<config_version>.json per run, and a table of
every answer that differs from the comparison config.

Usage: python scripts/rescore_with_inventory_rows.py --pages-file runs/test_logs/pages_58runs_with_v3_4.txt \
           --rows runs/E15_dense_lung_inventory_replay_20261123 --config configs/checklist_v3_7.json \
           --compare configs/checklist_v3_6.json --out runs/E16_.../rescore_v3_7.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.checklist import build_checklist, load_checklist_config  # noqa: E402
from lung_ct.agent.conditions import E88_NAME_NEUTRAL, E88_NOTE_NEUTRAL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pages-file", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    new, old = load_checklist_config(ROOT / args.config), load_checklist_config(ROOT / args.compare)
    runs = [ROOT / Path(x.strip().replace("\\", "/")).parent for x in (ROOT / args.pages_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    trans, changed, states, no_rows = collections.Counter(), [], collections.Counter(), []
    for run in runs:
        ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
        cid = ev["case_id"]
        rp = ROOT / args.rows / "evidence_rows" / f"{cid}.json"
        if rp.exists():
            add = json.loads(rp.read_text(encoding="utf-8"))["rows"]
            for r in ev["rows"]:
                if r["id"] == "E88":
                    r["name"], r["note"] = E88_NAME_NEUTRAL, E88_NOTE_NEUTRAL
            ev["rows"] += [r for r in add if r["id"] != "E88"]
        else:
            no_rows.append(cid)
        rd = json.loads((run / "readers.json").read_text(encoding="utf-8")).get("parsed", {})
        cl = ev.get("classifier") or {}
        a = build_checklist(ev, rd, old, cl.get("high_threshold", 0.7), cl.get("low_threshold", 0.3))
        b = build_checklist(ev, rd, new, cl.get("high_threshold", 0.7), cl.get("low_threshold", 0.3))
        b["rescored_from"] = f"evidence.json (unchanged)" + (f" + rows E92-E94 from {args.rows}/evidence_rows/{cid}.json" if rp.exists() else "; no inventory rows for this run")
        (run / f"checklist_{new['config_version']}.json").write_text(json.dumps(b, indent=2), encoding="utf-8")
        ia, ib = {i["key"]: i for i in a["items"]}, {i["key"]: i for i in b["items"]}
        for k in ia:
            trans[(k, ia[k]["status"], ib[k]["status"])] += 1
            if ia[k]["status"] != ib[k]["status"]:
                changed.append({"case_id": cid, "condition": k, "from": ia[k]["status"], "to": ib[k]["status"], "measured_value": ib[k].get("measured_value"),
                                "display": ib[k].get("display"), "historical_rule_answer": ib[k].get("historical_rule_answer")})
        pe = ib.get("pleural_effusion", {})
        states[(pe.get("mask_state"), pe.get("status"), pe.get("display"), pe.get("historical_rule_answer"))] += 1
    summ = {"config": new["config_version"], "compared_with": old["config_version"], "runs": len(runs), "runs_without_inventory_rows": no_rows,
            "status_changes": [{"condition": k[0], "from": k[1], "to": k[2], "runs": v} for k, v in sorted(trans.items()) if k[1] != k[2]],
            "pleural_effusion": [{"mask_state": k[0], "status": k[1], "display": k[2], "historical_rule_answer": k[3], "runs": v} for k, v in sorted(states.items(), key=str)],
            "status_changes_detail": changed}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "status_changes_detail"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
