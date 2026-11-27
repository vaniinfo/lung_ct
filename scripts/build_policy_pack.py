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


NOT_COPIED_NOTE = "Your earlier grades are shown on each page for reference, with the policy they were made on, and are not copied into the new form."
MIP_HOW = ("""<div class="how" style="background:#eaf7ee;border-color:#a9d8b8"><b>New on these pages: MIP (maximum intensity projection).</b> The page opens as before, on a single axial image. In the bar above the scan choose <b>MIP</b> to see the brightest value through a slab of neighbouring axial images, and <b>5 mm</b>, <b>8 mm</b> or <b>10 mm</b> for the slab. The wheel moves the slab one image at a time. The bar shows the real slab thickness and which images are in it. <b>Single slice</b>, or the green button, returns to the image at the centre of the slab. Outlines are hidden while MIP is on, because an outline belongs to one image; they come back with Single slice. MIP is a way of displaying the scan only: no answer or measurement on the page uses it.</div>""")
V6_HOW = ("""<div class="how" style="background:#eaf7ee;border-color:#a9d8b8"><b>New on these pages: nodule-review display.</b> The scan opens on <b>Axial 2.5 mm</b>, in the lung window, fitted to the window, without outlines. <b>MIP 10 mm</b> shows the brightest value through about 10 mm. The wheel moves one thin slice at a time in both, so nothing is skipped. The 2.5 mm view is calculated on the page by averaging the thin slices; it is not a reconstruction from the scanner. <b>Details</b> gives the exact slab thickness and the <b>native thin slices</b>. Outlines are shown on native thin slices only; &ldquo;Show in scan&rdquo; switches to them by itself. These are ways of displaying the scan: no answer or measurement on the page uses them.</div>""")
READER_HOW = ("""<div class="how" style="background:#fff8e6;border-color:#e6cf8a"><b>New on these pages: one AI reader, __NAME__.</b> Under each answer of the tool a small line gives what this AI model answered for the same condition, and a closed card near the bottom holds the report it wrote. It is shown for information only: it does not change any answer of the tool. It is a research model trained on reports from the same hospital as these scans.</div>""")
READER_USED_HOW = ("""<div class="how" style="background:#fff8e6;border-color:#e6cf8a"><b>New on these pages: AI reader __NAME__, used for three answers.</b> For <b>__FOR__</b> the tool now answers &ldquo;present&rdquo; when this AI model says present; before, the tool answered &ldquo;cannot tell&rdquo;. When the model says absent the tool still answers &ldquo;cannot tell&rdquo;. Hold the pointer over the answer to see the model&rsquo;s score (0 to 1). For every other condition the model&rsquo;s answer is shown under the tool&rsquo;s answer for information only. This is a project decision on development scans, not a validated rule; the model was trained on reports from the same hospital as these scans.</div>""")
POLICY_OLD = ("What is different in this policy: a classifier score is supporting evidence only. An empty or small predicted outline never means "
              "&ldquo;absent&rdquo;. A positive answer that rests only on a predicted outline reads &ldquo;suggested by segmentation&rdquo;. Dense lung is listed as tagged "
              "candidates, not as findings. The limits are provisional reporting limits, not validated diagnostic boundaries.")
POLICY_V312 = ("What this policy does: an answer is &ldquo;present&rdquo; when a measurement shows it, or, for pleural effusion, aortic and coronary calcification, when two "
               "sources agree. A predicted outline on its own gives &ldquo;cannot tell&rdquo;, with what it outlined listed; there is no &ldquo;suggested by segmentation&rdquo; "
               "answer any more. An empty or small outline never means &ldquo;absent&rdquo;. Limits and cut-offs are provisional, fitted on development scans, not validated.")
READER_TWO_SOURCE_HOW = ("""<div class="how" style="background:#fff8e6;border-color:#e6cf8a"><b>AI reader __NAME__ and the two-source rule.</b> For <b>__FOR__</b> the tool answers &ldquo;present&rdquo; when this AI model says present <b>and</b> a second source agrees: the classifier score is above that condition&rsquo;s cut-off, or (pleural effusion) the segmentation model outlined at least 20 mL of fluid. The answer then says which sources agreed. One source alone gives &ldquo;cannot tell&rdquo;. When the model says absent the tool never answers &ldquo;absent&rdquo; from it. For every other condition the model&rsquo;s answer is shown under the tool&rsquo;s answer for information only. Cut-offs were fitted on 198 development scans and checked on 197 others against report labels; they are not validated.</div>""")
PREFILL_NOTE = ("Your earlier grades are filled into the form as a starting point. They were made on pages showing an earlier policy, so the answers "
                "you graded may read differently here: please check each grade and change it where needed.")

SEG_HOW = ("""<div class="how" style="background:#f3eefc;border-color:#cdbdf0"><b>New on these pages: outlines from two segmentation models.</b> Under the outline bar there is a second bar, <b>Segmentation models</b>. It shows what <b>TotalSegmentator</b> (the model the tool&rsquo;s measurements use) and <b>NV-Segment-CT</b> (a second model, added for comparison) each outlined: lung lobes, trachea, airways, heart, and for the second model also bones, other organs and its tumour classes. One button at a time; the name of the structure under the cross-hair is written under the scan. The same structure has the same colour in both models. These outlines are predictions and have not been checked by a person. No answer or measurement on the page uses the second model. Its class &lsquo;lung tumor&rsquo; is the model&rsquo;s own class name, not a finding. Outlines are shown on native thin slices: open <b>Details</b> and choose them first.</div>""")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--checklist", required=True, help="config version, e.g. checklist_v3_8")
    ap.add_argument("--rows", required=True, help="replay folder holding evidence_rows/<case>.json (E92-E94)")
    ap.add_argument("--prefill-earlier-grades", action="store_true", help="fill the reviewer's earlier grades into the form (origin recorded in every exported row)")
    ap.add_argument("--reference-columns", action="store_true", help="show the CT-RATE report label as a visible column")
    ap.add_argument("--viewer", choices=["v4", "v5", "v6", "v7"], default="v4",
                    help="v5 = v4 plus the optional axial MIP mode; v6 = opens on Axial 2.5 mm, with MIP 10 mm; v7 = v6 plus the segmentation-model bar")
    ap.add_argument("--second-seg-run", action="append", default=[], metavar="RUNDIR",
                    help="with --viewer v7: scripts/run_nvsegment.py run(s) holding the cases; shown for comparison, never used. Repeat for several run folders")
    ap.add_argument("--reader-run", action="append", default=[], metavar="RUNDIR",
                    help="AI reader run(s) holding the cases (scripts/run_nvreason.py layout); shown on the pages, not counted. Repeat for several run folders")
    ap.add_argument("--reader-name", default="NV-Reason-CT")
    ap.add_argument("--cases-file", default=None, metavar="JSON",
                    help='[{"case": ..., "run": ...}, ...]: build these cases instead of the four graded ones; no earlier grades are shown')
    args = ap.parse_args()
    if args.second_seg_run and args.viewer != "v7":
        ap.error("--second-seg-run needs --viewer v7")
    out = ROOT / args.folder
    if out.exists() and any(out.iterdir()) and not (out / "PACK_INFO.json").exists():
        raise SystemExit(f"{out} exists and is not a policy pack; refusing to write into it")
    if out.exists() and (out / "PACK_INFO.json").exists() and json.loads((out / "PACK_INFO.json").read_text(encoding="utf-8"))["pack"] != args.pack:
        raise SystemExit(f"{out} belongs to another pack")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "configs" / f"{args.checklist}.json").read_text(encoding="utf-8"))
    trusted = [c["label"] for c in cfg["conditions"] if c.get("reader_decides_present") == args.reader_name] if args.reader_run else []
    two_src = [c["label"] for c in cfg["conditions"] if (c.get("two_source_present") or {}).get("reader") == args.reader_name] if args.reader_run else []
    if any(c.get("two_source_present") for c in cfg["conditions"]) and not two_src:
        raise SystemExit(f"{args.checklist} uses an AI reader in its two-source rule: give --reader-run and the matching --reader-name")
    if any(c.get("reader_decides_present") for c in cfg["conditions"]) and not trusted:
        raise SystemExit(f"{args.checklist} uses an AI reader for some answers: give --reader-run and the matching --reader-name")
    auth = {f["file"]: f["sha256"] for f in json.loads((ROOT / "runs/grades_received/AUTHENTICATED_EXPORTS.json").read_text(encoding="utf-8"))["files"]}
    pages = []
    cases = ([(c["case"], c["run"], None) for c in json.loads((ROOT / args.cases_file).read_text(encoding="utf-8"))] if args.cases_file else CASES)
    if args.cases_file and args.prefill_earlier_grades:
        ap.error("--prefill-earlier-grades needs the graded cases; not with --cases-file")
    for case, run, grades in cases:
        gp = None
        if grades is not None:
            found = sorted((ROOT / "runs" / "grades_received").glob(grades))   # '*' stands for the reader's name in the export's file name
            if len(found) != 1:
                raise SystemExit(f"{grades}: {len(found)} files match in runs/grades_received, need exactly one")
            grades = found[0].name
            gp = ROOT / "runs" / "grades_received" / grades
            if auth.get(grades) != sha(gp):
                raise SystemExit(f"{grades}: not an authenticated export, or changed")
        chk_file = f"checklist_{args.checklist}.json"
        chk = json.loads((ROOT / "runs" / run / chk_file).read_text(encoding="utf-8"))
        if chk["config_version"] != args.checklist:
            raise SystemExit(f"{run}/{chk_file}: holds {chk['config_version']}")
        name = f"viewer_{case[:-4]}_{args.pack}_{args.checklist.replace('checklist_', '')}.html"
        cmd = [sys.executable, str(ROOT / "scripts" / "build_viewer_page.py"), "--run", f"runs/{run}", "--checklist-file", chk_file, "--rule-based",
               "--viewer", args.viewer, "--corrected-wording", "--policy-pack", args.pack, "--extra-evidence-rows", f"{args.rows}/evidence_rows/{case}.json",
               "--out", f"{args.folder}/{name}"] + (["--earlier-grades-csv", f"runs/grades_received/{grades}"] if gp else [])
        cmd += (["--prefill-earlier-grades"] if args.prefill_earlier_grades else []) + (["--reference-columns"] if args.reference_columns else [])
        rrun = [r for r in args.reader_run if (ROOT / r / case / "vlm_parsed.json").exists()]
        if args.reader_run and len(rrun) != 1:
            raise SystemExit(f"{case}: {len(rrun)} of the reader runs hold a valid answer, need exactly one")
        if rrun:
            cmd += ["--reader-run", f"{args.reader_name}={rrun[0]}"]
        srun = [r for r in args.second_seg_run if (ROOT / r / case / "nvseg_labels.nii.gz").exists()]
        if args.second_seg_run and len(srun) != 1:
            raise SystemExit(f"{case}: {len(srun)} of the second-segmentation runs hold this case, need exactly one")
        if srun:
            cmd += ["--second-seg", srun[0]]
        print(" ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        pe = next(i for i in chk["items"] if i["key"] == "pleural_effusion")
        made_on = None
        if gp:
            with open(gp, encoding="utf-8-sig") as f:
                first = f.readline(), f.readline()
            cols = first[0].strip().split(",")
            made_on = first[1].split('","')[cols.index("checklist_config")].strip('"\r\n')
        pages.append({"case_id": case, "run": run, "file": name, "sha256": sha(out / name), "bytes": (out / name).stat().st_size,
                      "ai_reader": ({"name": args.reader_name, "run": rrun[0], "answer_sha256": sha(ROOT / rrun[0] / case / "vlm_parsed.json"),
                                     "shown": "under each tool answer and as a closed report card",
                                     "used_as_answer_for": trusted,
                                     "answers_taken_from_reader": [i["key"] for i in chk["items"] if i.get("reader_decided")]} if rrun else None),
                      "second_segmentation": ({"model": "NV-Segment-CT", "run": srun[0], "label_map_sha256": sha(ROOT / srun[0] / case / "nvseg_labels.nii.gz"),
                                               "shown": "in the segmentation bar, next to TotalSegmentator, native slices only",
                                               "used_by_answers_or_measurements": False} if srun else None),
                      "checklist_file": f"runs/{run}/{chk_file}", "checklist_file_sha256": sha(ROOT / "runs" / run / chk_file),
                      "evidence_sha256": sha(ROOT / "runs" / run / "evidence.json"), "extra_rows": f"{args.rows}/evidence_rows/{case}.json",
                      "pleural_effusion_shown": {"display": pe.get("display"), "headline": pe.get("headline"), "mask_state": pe.get("mask_state"),
                                                 "provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT": (pe.get("provisional_threshold_result") or {}).get("answer")},
                      "earlier_review": None if not gp else {"file": f"runs/grades_received/{grades}", "sha256": auth[grades], "made_on_policy": made_on,
                                         "shown": ("read-only card AND filled into the form as a starting point; each exported row records answer_origin "
                                                   "(carried over unchanged / changed / entered on this page) and the policy and date of the earlier grade"
                                                   if args.prefill_earlier_grades else "read-only, collapsed, labelled with that policy; not filled into the form"),
                                         "reference_columns": bool(args.reference_columns)}})
    li = "".join(f'<li><a href="{E(p["file"])}"><b>{E(p["case_id"][:-4])}</b></a>' + (f' <span class="muted">earlier review was made on {E(p["earlier_review"]["made_on_policy"])}</span>' if p["earlier_review"] else "") + '</li>'
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
<div class="muted">{POLICY_V312 if two_src else POLICY_OLD}</div>
<div class="muted" style="margin-top:8px">Grades saved from these pages name {E(args.checklist)} and pack {E(args.pack)} in every row. {PREFILL_NOTE if args.prefill_earlier_grades else NOT_COPIED_NOTE if not args.cases_file else ""}</div>
<div class="how"><b>Using the scan:</b> wheel or arrow keys = next / previous image &middot; click = move the cross-hair &middot; drag = move the image &middot;
right-drag or Shift+drag = brightness / contrast &middot; Ctrl+wheel or the Zoom buttons = zoom &middot; Axial / Coronal / Sagittal buttons change the view.
R, L, A, P, S, I at the image edges mark the patient&rsquo;s right, left, front, back, head and feet.</div>{MIP_HOW if args.viewer == 'v5' else V6_HOW if args.viewer in ('v6', 'v7') else ''}{SEG_HOW if args.viewer == 'v7' else ''}{(READER_TWO_SOURCE_HOW.replace('__FOR__', E(', '.join(two_src))) if two_src else READER_USED_HOW.replace('__FOR__', E(', '.join(trusted))) if trusted else READER_HOW).replace('__NAME__', E(args.reader_name)) if args.reader_run else ''}
<div class="muted" style="margin-top:8px">Contains CT-RATE images and report text (CC BY-NC-SA, research use): keep within the research group.</div>
</div></div></body></html>
""", encoding="utf-8")
    info = {"pack": args.pack, "folder": args.folder, "built_utc": datetime.now(timezone.utc).isoformat(), "policy": args.checklist,
            "page_kind": "development, open (tool answers visible); not blinded, not evaluation data",
            "viewer": {"v5": "checklist_viewer_v5_2_canvas2d_mip", "v6": "checklist_viewer_v6_canvas2d_axial25_mip10", "v7": "checklist_viewer_v7_canvas2d_axial25_mip10_two_segmentations"}.get(args.viewer, "checklist_viewer_v4_canvas2d") + ", form grades_v2, corrected wording copy",
            "grades_from_these_pages": "CSV rows carry checklist_config / checklist_version = the policy, and pack; browser storage key is separate per policy and pack",
            "historical_outputs": "not written: M1-M4, adjudication package, evidence.json, earlier checklist files, runs/grades_received/",
            "index": {"file": "index.html", "sha256": sha(out / "index.html")}, "pages": pages}
    (out / "PACK_INFO.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
