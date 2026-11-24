"""Grounded agent demo, steps 0, 1, 3 (read existing runs) and 4, for one case (core env).

Writes runs/A1_agent_<case>_<timestamp>/:
  qc.json            step 0: the case's Phase 1 QC record (HU validity, geometry)
  evidence.json      step 1: evidence table (predicted masks only; never the expert masks)
  evidence_view.png  coronal slices: predicted lobes + dense-lung proxy
  readers.json       step 3: the AI readers' parsed answers, from existing runs
  crosscheck.json    step 4: agreements, conflicts and rule-based decisions
  report_input.json  prompt + material for step 5 (scripts/agent_write_report.py, vlm env)
  report_rule_based.json  deterministic fallback report
  run_manifest.json  provenance

Usage:
  python scripts/agent_run_case.py --case coronacases_001 \
      --reader medgemma=runs/P3_vlm_10cases_20261108_083032 \
      --reader hulumed=runs/P3_hulumed_10cases_20261115_065454
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from lung_ct.agent.crosscheck import crosscheck  # noqa: E402
from lung_ct.agent.evidence import build_evidence, load_config  # noqa: E402
from lung_ct.agent.checklist import (build_checklist, build_checklist_prompt,  # noqa: E402
                                     load_checklist_config, rule_based_checklist_report)
from lung_ct.agent.report import build_prompt, rule_based_report  # noqa: E402


def git_commit() -> str:
    try:
        c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        return c + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def save_figure(ct: nib.Nifti1Image, arrays: dict, path: Path) -> None:
    """Three coronal slices, radiological display (patient right on image left), physical aspect."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    can = nib.as_closest_canonical(nib.Nifti1Image(arrays["hu"], ct.affine))
    to_can = lambda a: np.asanyarray(nib.as_closest_canonical(  # noqa: E731
        nib.Nifti1Image(a.astype(np.uint8), ct.affine)).dataobj)
    hu, lab, prox = np.asanyarray(can.dataobj), to_can(arrays["lab"]), to_can(arrays["proxy"]).astype(bool)
    sx, sy, sz = can.header.get_zooms()[:3]
    ys = np.where(lab.any(axis=(0, 2)))[0]
    picks = [int(np.percentile(ys, q)) for q in (30, 50, 70)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), facecolor="black")
    for ax, y, name in zip(axes, picks, ("posterior", "middle", "anterior")):  # canonical RAS: +y = anterior
        sl = lambda a: a[::-1, y, ::-1].T  # noqa: E731  cols: patient right->left, rows: sup->inf
        ax.imshow(np.clip(sl(hu), -1350, 150), cmap="gray", aspect=sz / sx)
        side = sl(lab).astype(float)
        side[side == 0] = np.nan
        ax.imshow(side, cmap=ListedColormap(["#4da3ff", "#ffd24d"]), alpha=0.18, aspect=sz / sx,
                  vmin=1, vmax=2)
        p = np.where(sl(prox), 1.0, np.nan)
        ax.imshow(p, cmap=ListedColormap(["#ff3b30"]), alpha=0.75, aspect=sz / sx)
        ax.set_title(f"coronal, {name}", color="white", fontsize=10)
        ax.text(4, 14, "R", color="white", fontsize=12, weight="bold")
        ax.text(sl(hu).shape[1] - 16, 14, "L", color="white", fontsize=12, weight="bold")
        ax.axis("off")
    fig.suptitle("Red: dense-lung proxy (evidence).  Tint: predicted right (yellow) / left (blue) lung.  "
                 "Radiological view: patient right on image left.", color="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)


def save_checklist_figure(ct: nib.Nifti1Image, arrays: dict, path: Path) -> None:
    """Three axial slices (radiological view, front at top): ground-glass range yellow, consolidation
    range red (non-dependent lung only), dependent band hatched blue."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    def can(a, dt=np.uint8):
        return np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(a.astype(dt), ct.affine)).dataobj)
    hu = np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(arrays["hu"], ct.affine)).dataobj)
    lab, gg, co, dep = can(arrays["lab"]), can(arrays["ground_glass"]), can(arrays["consolidation"]), can(arrays["dependent"])
    zs = np.where(lab.any(axis=(0, 1)))[0]
    picks = [int(np.percentile(zs, q)) for q in (75, 50, 25)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), facecolor="black")
    for ax, z, name in zip(axes, picks, ("upper", "middle", "lower")):
        sl = lambda a: a[::-1, ::-1, z].T  # noqa: E731  RAS -> rows front->back, cols patient right->left
        ax.imshow(np.clip(sl(hu), -1350, 150), cmap="gray")
        ax.imshow(np.where(sl(dep) > 0, 1.0, np.nan), cmap=ListedColormap(["#4da3ff"]), alpha=0.25)
        ax.imshow(np.where(sl(gg) > 0, 1.0, np.nan), cmap=ListedColormap(["#ffd60a"]), alpha=0.8)
        ax.imshow(np.where(sl(co) > 0, 1.0, np.nan), cmap=ListedColormap(["#ff3b30"]), alpha=0.85)
        ax.set_title(f"axial, {name} lung", color="white", fontsize=10)
        ax.text(4, 14, "R", color="white", fontsize=12, weight="bold")
        ax.text(sl(hu).shape[1] - 16, 14, "L", color="white", fontsize=12, weight="bold")
        ax.axis("off")
    fig.suptitle("Yellow: ground-glass range.  Red: consolidation range (front/middle lung only).  "
                 "Blue: back-of-lung band, not counted.  Patient right on image left, front at top.",
                 color="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--reader", action="append", default=[], metavar="NAME=RUNDIR")
    ap.add_argument("--qc-run", default="runs/P1_qc_20cases_20261107_223515")
    ap.add_argument("--airway-run", default="runs/P5_airway_10cases_20261108_092858")
    ap.add_argument("--config", default=None)
    ap.add_argument("--classifier", default=None, metavar="RUNDIR",
                    help="COLIPRI run (scripts/agent_classify.py); step 2 is marked not run without it")
    ap.add_argument("--checklist-config", default="configs/checklist_v3_9.json",
                    help="default since 2026-11-23: v3_8 (classifier cannot establish absence; dense-lung candidates kept and tagged; a predicted "
                         "pleural-fluid mask below the reporting limit is 'cannot tell'; a positive mask-based answer is headlined 'suggested by "
                         "segmentation'). checklist_v3_7 / v3_6 / v3_5 / v3_4 are kept for comparison")
    ap.add_argument("--no-checklist", action="store_true", help="old opacity-only output (v1)")
    args = ap.parse_args()

    cid = args.case
    ck = None if args.no_checklist else load_checklist_config(ROOT / args.checklist_config)
    cfg = load_config(ROOT, args.config)
    out = ROOT / "runs" / f"A1_agent_{cid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)

    # step 0
    qc = json.loads((ROOT / args.qc_run / "run_manifest.json").read_text(encoding="utf-8"))
    qrec = next((c for c in qc["cases"] if c["case_id"] == cid), None)
    if qrec is None or qrec.get("status") == "failed" or not qrec["ct"].get("hu_valid"):
        print(f"step 0 FAILED: {cid} absent, failed or not HU in {args.qc_run}")
        return 2
    (out / "qc.json").write_text(json.dumps(qrec, indent=2), encoding="utf-8")
    print(f"step 0 ok: {qrec['ct'].get('hu_reason')}")

    # step 2 (existing classifier run, GPU env)
    cl = None
    if args.classifier:
        cp = ROOT / args.classifier / cid / "classifier.json"
        cl = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else None
    print(f"step 2 {'ok' if cl else 'not run'}{': ' + cl['model'] if cl else ''}")

    # step 1
    ev, ct, arrays = build_evidence(ROOT, cid, ROOT / qrec["ct"]["file"].replace("\\", "/"),
                                    ROOT / "data" / "derived" / cid / "segmentation",
                                    ROOT / args.airway_run / cid / "airway_summary.json", cfg, cl,
                                    checklist_cfg=ck)
    (out / "evidence.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    save_figure(ct, arrays, out / "evidence_view.png")
    if ck is not None:
        save_checklist_figure(ct, arrays, out / "checklist_view.png")
    print(f"step 1 ok: {len(ev['rows'])} evidence rows; derived {ev['derived_answers']['distribution']}")

    # step 3 (existing runs)
    readers, reader_src = {}, {}
    for spec in args.reader:
        name, _, rdir = spec.partition("=")
        p = ROOT / rdir / cid / "vlm_parsed.json"
        readers[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        reader_src[name] = rdir
    (out / "readers.json").write_text(json.dumps({"sources": reader_src, "parsed": readers}, indent=2),
                                      encoding="utf-8")
    print(f"step 3 ok: readers {list(readers)}")

    # step 4
    v1 = {n: p for n, p in readers.items() if p is None or "opacity_presence" in p}
    if v1:
        cc = crosscheck(ev, v1, cfg)
    else:
        cc = {"n_conflicts": 0, "items": [], "decisions": {}, "rule": "not run: no opacity-schema (v1) readers"}
    (out / "crosscheck.json").write_text(json.dumps(cc, indent=2), encoding="utf-8")
    print(f"step 4 ok: {cc['n_conflicts']} conflicts flagged")

    # step 4b + 5 inputs + fallback
    if ck is not None:
        hi = (cl or {}).get("high_threshold", 0.7)
        lo = (cl or {}).get("low_threshold", 0.3)
        chk = build_checklist(ev, readers, ck, hi, lo)
        (out / "checklist.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
        print(f"step 4b ok: checklist {chk['counts']} (scan type {chk['scan_type']})")
        (out / "report_input.json").write_text(json.dumps(
            {"case_id": cid, "kind": "checklist", "checklist_config": args.checklist_config,
             "prompt": build_checklist_prompt(chk, ev, ck)}, indent=2), encoding="utf-8")
        (out / "report_rule_based.json").write_text(json.dumps(rule_based_checklist_report(chk), indent=2),
                                                    encoding="utf-8")
    else:
        (out / "report_input.json").write_text(json.dumps(
            {"case_id": cid, "kind": "opacity_v1", "prompt": build_prompt(ev, cc, v1)}, indent=2), encoding="utf-8")
        (out / "report_rule_based.json").write_text(json.dumps(rule_based_report(ev, cc, v1), indent=2),
                                                    encoding="utf-8")
    (out / "run_manifest.json").write_text(json.dumps({
        "run_id": out.name, "case_id": cid,
        "provenance": {"finished_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
                       "config": cfg["config_version"], "qc_run": args.qc_run, "airway_run": args.airway_run,
                       "readers": reader_src, "classifier": args.classifier,
                       "checklist_config": None if ck is None else args.checklist_config},
        "steps": {"0": "ok", "1": "ok", "2": ("ok: " + args.classifier) if cl else "not run", "3": "ok (existing runs)",
                  "4": "ok", "5": "pending: scripts/agent_write_report.py"},
        "reference_masks_used": False}, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
