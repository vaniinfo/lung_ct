"""Score HoloCT yes/no answers against CT-RATE report labels, side by side with the checklist (core env).

For every --compare folder (a runs/E6_checklist_scores_* folder) the same cases are scored for both
systems, so the two columns are directly comparable. Per CT-RATE label:
  found  = says present and the report labels it        missed = says absent / no, report labels it
  extra  = says present, report does not mention it     open   = checklist only: cannot tell / cannot assess
HoloCT always answers yes or no, so it has no 'open'. CT-RATE labels come from report text, so an
'extra' can be a real finding the radiologist did not mention.

Usage: python scripts/score_holoct.py --run runs/P3_holoct_all85_<time> --compare runs/E6_checklist_scores_test20c_v3_1_<time> [...]
Writes runs/E8_holoct_vs_checklist_<time>/{scores.md,scores.json}
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("labelled", "found", "missed", "extra", "open", "correct_absent")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--compare", nargs="+", required=True, type=Path)
    ap.add_argument("--checklist-config", default="configs/checklist_v3_2.json", help="only for item key -> CT-RATE label")
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    cfg = json.loads((ROOT / args.checklist_config).read_text(encoding="utf-8"))
    key2lab = {c["key"]: c["ct_rate_label"] for c in cfg["conditions"] if c.get("ct_rate_label")}
    holo = {}
    for f in sorted(run.glob("*/answers.json")):
        a = json.loads(f.read_text(encoding="utf-8"))
        holo[a["case_id"]] = {x["ct_rate_label"]: x["answer"] for x in a["answers"]}
    failed = [p.name.replace("_error.txt", "") for p in run.glob("*_error.txt")]

    out = ROOT / "runs" / f"E8_holoct_vs_checklist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    md = ["# HoloCT-4B vs lung_ct checklist, against CT-RATE report labels", "",
          f"HoloCT run: `{run.name}` ({len(holo)} cases answered, {len(failed)} failed).", ""]
    result = {"holoct_run": run.name, "holoct_failed": failed, "comparisons": []}
    for cmp_dir in args.compare:
        cmp_dir = cmp_dir if cmp_dir.is_absolute() else ROOT / cmp_dir
        sc = json.loads((cmp_dir / "scores.json").read_text(encoding="utf-8"))
        ids = [c["case_id"] for c in sc["cases"]]
        missing = [c for c in ids if c not in holo]
        use = [c for c in sc["cases"] if c["case_id"] in holo]
        H = defaultdict(lambda: defaultdict(int))
        C = defaultdict(lambda: defaultdict(int))
        labs = []
        for c in use:
            labels = set(gt[c["case_id"]]["labels"])
            for key, it in c["items"].items():
                lab = key2lab.get(key)
                if lab is None:
                    continue
                if lab not in labs:
                    labs.append(lab)
                pos = lab in labels
                s = it["status"]
                ck = ("found" if pos else "extra") if s == "present" else ("missed" if pos else "correct_absent") if s == "absent" else "open"
                C[lab][ck] += 1
                C[lab]["labelled"] += int(pos)
                h = holo[c["case_id"]].get(lab, "unparsed")
                hk = ("found" if pos else "extra") if h == "yes" else ("missed" if pos else "correct_absent") if h == "no" else "open"
                H[lab][hk] += 1
                H[lab]["labelled"] += int(pos)
        th = {k: sum(H[l][k] for l in labs) for k in KEYS}
        tc = {k: sum(C[l][k] for l in labs) for k in KEYS}
        n_abs = lambda t: t["extra"] + t["correct_absent"]  # noqa: E731  (decided cases only for the checklist)
        md += [f"## {cmp_dir.name} ({len(use)} cases)", ""]
        if missing:
            md += [f"Not answered by HoloCT and left out for both: {', '.join(missing)}", ""]
        md += ["| condition (CT-RATE label) | labelled | HoloCT found | HoloCT missed | HoloCT extra | checklist found | checklist missed | checklist extra | checklist cannot tell |",
               "|---|---|---|---|---|---|---|---|---|"]
        for lab in labs:
            h, c = H[lab], C[lab]
            md.append(f"| {lab} | {h['labelled']} | {h['found']} | {h['missed']} | {h['extra']} | {c['found']} | {c['missed']} | {c['extra']} | {c['open']} |")
        md.append(f"| **all** | {th['labelled']} | {th['found']} | {th['missed']} | {th['extra']} | {tc['found']} | {tc['missed']} | {tc['extra']} | {tc['open']} |")
        md += ["",
               f"- HoloCT: found {th['found']} of {th['labelled']} labelled findings "
               f"({100 * th['found'] / max(th['labelled'], 1):.0f}%); {th['extra']} extras on {n_abs(th)} unlabelled questions "
               f"({100 * th['extra'] / max(n_abs(th), 1):.1f}%); {th['open']} unparsed.",
               f"- Checklist: found {tc['found']} of {tc['labelled']} ({100 * tc['found'] / max(tc['labelled'], 1):.0f}%); "
               f"{tc['extra']} extras; {tc['open']} of {sum(tc[k] for k in ('found', 'missed', 'extra', 'open', 'correct_absent'))} answers are 'cannot tell'.",
               ""]
        result["comparisons"].append({"checklist_scores": cmp_dir.name, "cases": [c["case_id"] for c in use],
                                      "left_out": missing, "holoct": H, "checklist": C,
                                      "holoct_total": th, "checklist_total": tc})
    md += ["Notes: small numbers, read as counts not rates. CT-RATE labels are derived from report text; 'extra' may be a real "
           "unreported finding. HoloCT was trained on the CT-RATE train split (same hospital and report style); our cases are "
           "in its published test split, not its train split. The checklist is built to say 'cannot tell' where it has no "
           "measuring tool. Research use only; not a diagnosis."]
    (out / "scores.md").write_text("\n".join(md), encoding="utf-8")
    (out / "scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
