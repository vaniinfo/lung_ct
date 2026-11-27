"""Re-apply a checklist config to finished agent runs without recomputing any measurement (core env).

Only the decision rules change; evidence.json and readers.json are reused. Writes
<run>/checklist_<config_version>.json next to the original checklist.json (never overwrites it).
Use it to see what a rule change does on runs you already have; it is NOT a fresh test when the
rule was designed on those same runs.

Usage: python scripts/rescore_checklist.py --pages-file runs/test_logs/pages_test20b_v3.txt \
           --config configs/checklist_v3_1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.checklist import build_checklist, load_checklist_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pages-file", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    ck = load_checklist_config(ROOT / args.config)
    n = 0
    for line in (ROOT / args.pages_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = ROOT / Path(line.strip().replace("\\", "/")).parent
        ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
        rd = json.loads((run / "readers.json").read_text(encoding="utf-8")).get("parsed", {})
        cl = ev.get("classifier") or {}
        chk = build_checklist(ev, rd, ck, cl.get("high_threshold", 0.7), cl.get("low_threshold", 0.3))
        chk["rescored_from"] = "evidence.json (measurements unchanged)"
        out = run / f"checklist_{ck['config_version']}.json"
        out.write_text(json.dumps(chk, indent=2), encoding="utf-8")
        n += 1
    print(f"rescored {n} runs with {ck['config_version']} -> checklist_{ck['config_version']}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
