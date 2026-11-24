"""Build a policy-labelled viewer pack for the four development cases (core env). New folder only; nothing older is touched.

One open 2D-canvas page (viewer v4) per case, showing the answers of ONE checklist version, named on every page and in
every grade file saved from it. Earlier packs (M1-M4), the adjudication package, evidence.json, earlier checklist files and
the grade files in runs/grades_received/ are read, never written. Earlier grades are shown read-only with the policy they
were made on and are not filled into the form.

Usage: python scripts/build_policy_pack.py --pack M5 --folder runs/M5_policy_v3_8_20261123 --checklist checklist_v3_8 \
           --rows runs/E15_dense_lung_inventory_replay_20261123
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E = html.escape
CASES = [  # case, run, earlier authenticated grade file
    ("valid_301_a_2", "A1_agent_valid_301_a_2_20261118_152058", "grades_valid_301_a_2_*_2026-11-20.csv"),
    ("valid_541_a_2", "A1_agent_valid_541_a_2_20261118_151640", "grades_valid_541_a_2_*_2026-11-21.csv"),
    ("valid_832_a_2", "A1_agent_valid_832_a_2_20261118_094216", "grades_valid_832_a_2_*_2026-11-22.csv"),
    ("valid_988_a_2", "A1_agent_valid_988_a_2_20261118_153023", "grades_valid_988_a_2_*_2026-11-22.csv"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--checklist", required=True, help="config version, e.g. checklist_v3_8")
    ap.add_argument("--rows", required=True, help="replay folder holding evidence_rows/<case>.json (E92-E94)")
    args = ap.parse_args()
    out = ROOT / args.folder
    if out.exists() and any(out.iterdir()) and not (out / "PACK_INFO.json").exists():
        raise SystemExit(f"{out} exists and is not a policy pack; refusing to write into it")
    if out.exists() and (out / "PACK_INFO.json").exists() and json.loads((out / "PACK_INFO.json").read_text(encoding="utf-8"))["pack"] != args.pack:
        raise SystemExit(f"{out} belongs to another pack")
    out.mkdir(parents=True, exist_ok=True)
    auth = {f["file"]: f["sha256"] for f in json.loads((ROOT / "runs/grades_received/AUTHENTICATED_EXPORTS.json").read_text(encoding="utf-8"))["files"]}
    pages = []
    for case, run, grades in CASES:
        gp = ROOT / "runs" / "grades_received" / grades
        if auth.get(grades) != sha(gp):
            raise SystemExit(f"{grades}: not an authenticated export, or changed")
        chk_file = f"checklist_{args.checklist}.json"
        chk = json.loads((ROOT / "runs" / run / chk_file).read_text(encoding="utf-8"))
        if chk["config_version"] != args.checklist:
            raise SystemExit(f"{run}/{chk_file}: holds {chk['config_version']}")
        name = f"viewer_{case[:-4]}_{args.pack}_{args.checklist.replace('checklist_', '')}.html"
        cmd = [sys.executable, str(ROOT / "scripts" / "build_viewer_page.py"), "--run", f"runs/{run}", "--checklist-file", chk_file, "--rule-based",
               "--viewer", "v4", "--corrected-wording", "--policy-pack", args.pack, "--extra-evidence-rows", f"{args.rows}/evidence_rows/{case}.json",
               "--earlier-grades-csv", f"runs/grades_received/{grades}", "--out", f"{args.folder}/{name}"]
        print(" ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        pe = next(i for i in chk["items"] if i["key"] == "pleural_effusion")
        with open(gp, encoding="utf-8-sig") as f:
            first = f.readline(), f.readline()
        cols = first[0].strip().split(",")
        made_on = first[1].split('","')[cols.index("checklist_config")].strip('"\r\n')
        pages.append({"case_id": case, "run": run, "file": name, "sha256": sha(out / name), "bytes": (out / name).stat().st_size,
                      "checklist_file": f"runs/{run}/{chk_file}", "checklist_file_sha256": sha(ROOT / "runs" / run / chk_file),
                      "evidence_sha256": sha(ROOT / "runs" / run / "evidence.json"), "extra_rows": f"{args.rows}/evidence_rows/{case}.json",
                      "pleural_effusion_shown": {"display": pe.get("display"), "headline": pe.get("headline"), "mask_state": pe.get("mask_state"),
                                                 "provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT": (pe.get("provisional_threshold_result") or {}).get("answer")},
                      "earlier_review": {"file": f"runs/grades_received/{grades}", "sha256": auth[grades], "made_on_policy": made_on,
                                         "shown": "read-only, collapsed, labelled with that policy; not filled into the form"}})
    li = "".join(f'<li><a href="{E(p["file"])}"><b>{E(p["case_id"][:-4])}</b></a> <span class="muted">earlier review was made on {E(p["earlier_review"]["made_on_policy"])}</span></li>'
                 for p in pages)
    (out / "index.html").write_text(f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Lung CT agent - pack {E(args.pack)} ({E(args.checklist)})</title>
<style>
body{{margin:0;background:#f4f5f7;font:19px/1.55 'Segoe UI',Arial,sans-serif;color:#1f2328}}
.page{{max-width:900px;margin:0 auto;padding:26px}}
h2{{font-size:24px;margin:0 0 6px}}.muted{{color:#4f5a66;font-size:17px}}
.card{{background:#fff;border:1px solid #d5d9de;border-radius:12px;padding:20px 24px;margin-top:14px}}
ul{{margin:14px 0 12px 24px;padding:0}}li{{margin:14px 0;font-size:24px}}a{{color:#0969da}}
.how{{margin-top:12px;background:#eef4ff;border:1px solid #b9cdf5;border-radius:10px;padding:12px 16px;font-size:18px}}
.pol{{background:#e7f0ff;border:1px solid #9db9ee;border-radius:10px;padding:12px 16px;font-size:18px}}
</style></head><body><div class="page">
<div class="card"><h2>Pack {E(args.pack)}: answer policy {E(args.checklist)}</h2>
<div class="pol"><b>These pages show the current answer policy, {E(args.checklist)}.</b> Development pages, open: the tool&rsquo;s answers are visible
from the start. Not blinded and not an evaluation. Earlier packs (M1 to M4) show earlier policies and are kept unchanged.</div>
<ul>{li}</ul>
<div class="muted">What is different in this policy: a classifier score is supporting evidence only. An empty or small predicted outline never means
&ldquo;absent&rdquo;. A positive answer that rests only on a predicted outline reads &ldquo;suggested by segmentation&rdquo;. Dense lung is listed as tagged
candidates, not as findings. The limits are provisional reporting limits, not validated diagnostic boundaries.</div>
<div class="muted" style="margin-top:8px">Grades saved from these pages name {E(args.checklist)} and pack {E(args.pack)} in every row. Your earlier grades
are shown on each page for reference, with the policy they were made on, and are not copied into the new form.</div>
<div class="how"><b>Using the scan:</b> wheel or arrow keys = next / previous image &middot; click = move the cross-hair &middot; drag = move the image &middot;
right-drag or Shift+drag = brightness / contrast &middot; Ctrl+wheel or the Zoom buttons = zoom &middot; Axial / Coronal / Sagittal buttons change the view.
R, L, A, P, S, I at the image edges mark the patient&rsquo;s right, left, front, back, head and feet.</div>
<div class="muted" style="margin-top:8px">Contains CT-RATE images and report text (CC BY-NC-SA, research use): keep within the research group.</div>
</div></div></body></html>
""", encoding="utf-8")
    info = {"pack": args.pack, "folder": args.folder, "built_utc": datetime.now(timezone.utc).isoformat(), "policy": args.checklist,
            "page_kind": "development, open (tool answers visible); not blinded, not evaluation data",
            "viewer": "checklist_viewer_v4_canvas2d, form grades_v2, corrected wording copy",
            "grades_from_these_pages": "CSV rows carry checklist_config / checklist_version = the policy, and pack; browser storage key is separate per policy and pack",
            "historical_outputs": "not written: M1-M4, adjudication package, evidence.json, earlier checklist files, runs/grades_received/",
            "index": {"file": "index.html", "sha256": sha(out / "index.html")}, "pages": pages}
    (out / "PACK_INFO.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
