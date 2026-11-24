"""Compare checklist outputs with CT-RATE report labels across agent runs (core env).

CT-RATE labels come from the report text (RadBERT), so 'label absent' means 'not mentioned'.
Counts per condition: agree (present+labelled or absent+not labelled), missed (absent but labelled),
extra (present but not labelled), open (cannot tell / cannot assess).

Usage: python scripts/score_checklist.py --runs runs/A1_agent_valid_*_2026111718*  (globs allowed)
Writes runs/E6_checklist_scores_<time>/{scores.json,scores.md}
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.checklist import assessment_category  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--checklist-file", default="checklist.json",
                    help="file name inside each run folder (e.g. checklist_checklist_v3_1.json from rescore_checklist.py)")
    ap.add_argument("--label", default="", help="added to the output folder name, e.g. test20b_v3")
    ap.add_argument("--pages-file", help="runs/test_logs/agent_demo_pages.txt from agent_demo.py")
    args = ap.parse_args()
    paths = sorted({p for g in args.runs for p in glob.glob(str(ROOT / g) if not Path(g).is_absolute() else g)})
    if args.pages_file:
        for line in (ROOT / args.pages_file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                paths.append(str(ROOT / Path(line.strip().replace("\\", "/")).parent))
    gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    per = defaultdict(lambda: defaultdict(int))
    cases, used = [], []
    for p in paths:
        cj = Path(p) / args.checklist_file
        if not cj.exists():
            continue
        cid = json.loads((Path(p) / "run_manifest.json").read_text(encoding="utf-8"))["case_id"]
        if cid not in gt:
            continue
        used.append(Path(p).name)
        labels = set(gt[cid]["labels"])
        chk = json.loads(cj.read_text(encoding="utf-8"))
        row = {"case_id": cid, "labels": sorted(labels), "items": {}}
        for it in chk["items"]:
            lab = it.get("ct_rate_label")
            if not lab:
                continue
            pos = lab in labels
            s = assessment_category(it)      # 'suggested' (predicted mask only, v3_8) is never scored as a confirmed present
            k = (("agree" if pos else "extra") if s == "present" else ("missed" if pos else "agree") if s == "absent"
                 else ("suggested_labelled" if pos else "suggested_not_labelled") if s == "suggested" else "open")
            per[it["label"]][k] += 1
            per[it["label"]]["labelled"] += int(pos)
            row["items"][it["key"]] = {"status": it["status"], "current_assessment": s, "labelled": pos, "result": k}
        cases.append(row)
    tot = defaultdict(int)
    for v in per.values():
        for k, n in v.items():
            tot[k] += n
    tag = f"{args.label}_" if args.label else ""
    out = ROOT / "runs" / f"E6_checklist_scores_{tag}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    md = [f"# Checklist vs CT-RATE labels ({len(cases)} cases){' - ' + args.label if args.label else ''}", "",
          "| condition | labelled in reports | agree | missed | extra | open | suggested, labelled | suggested, not labelled |", "|---|---|---|---|---|---|---|---|"]
    for name, v in per.items():
        md.append(f"| {name} | {v['labelled']} | {v['agree']} | {v['missed']} | {v['extra']} | {v['open']} | {v['suggested_labelled']} | {v['suggested_not_labelled']} |")
    md.append(f"| **all** | {tot['labelled']} | {tot['agree']} | {tot['missed']} | {tot['extra']} | {tot['open']} | {tot['suggested_labelled']} | {tot['suggested_not_labelled']} |")
    md += ["", "agree = matches the report label; missed = agent says absent, report says present; "
           "extra = agent says present, report does not mention it; open = cannot tell / cannot assess; "
           "suggested = positive answer from a predicted mask only (not a confirmed finding), counted apart from agree / extra. "
           "A provisional threshold result is never scored.",
           "Report labels are text-derived and a finding the radiologist did not mention may still be real."]
    (out / "scores.md").write_text("\n".join(md), encoding="utf-8")
    (out / "scores.json").write_text(json.dumps({"runs": used, "per_condition": per, "total": tot, "cases": cases},
                                                indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
