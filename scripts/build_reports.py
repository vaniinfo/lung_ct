"""Phase 4 (core env): assemble reviewable per-case outputs and the feasibility summary.

Reads finished runs only (no inference):
  --qc-run    Phase 1 batch QC (all cases; reference volumes, HU status, QC overlays)
  --seg-run   Phase 2 TotalSegmentator run (timings, GPU memory, weight hashes)
  --eval-run  Phase 2 evaluation (predicted volumes, Dice vs reference, overlays)
  --vlm-run   Phase 3 MedGemma run (raw/parsed outputs, model input)

Writes runs/P4_report_<n>cases_<ts>/:
  cases/<id>/{case_report.html, qc.json, metrics.json, vlm_raw.txt, vlm_parsed.json,
              vlm_result.json, model_input.json, model_input_sheet.png, overlays/*.png}
  run_manifest.json, batch_results.csv, failures.csv, reviewer_template.csv,
  feasibility_summary.md

Every case in the QC run gets a report; cases excluded or failed at any stage are listed
with the reason, never dropped.

Usage:
    python scripts/build_reports.py --qc-run runs/P1_qc_20cases_<ts> \
        --seg-run runs/P2_seg_10cases_<ts> --eval-run runs/P2_eval_10cases_<ts> \
        --vlm-run runs/P3_vlm_10cases_<ts>
"""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import nibabel as nib  # noqa: E402

from lung_ct.metrics.volumes import infection_by_side  # noqa: E402
from lung_ct.reporting.case_report import (  # noqa: E402
    REVIEW_COLUMNS, contact_sheet, img_tag, render_case_html, reviewer_rows)
from lung_ct.reporting.review_flags import (  # noqa: E402
    load_rules, stage_category, vlm_review_flags)
from lung_ct.airway.sensitivity import (  # noqa: E402
    PARAMETER_VS_ANATOMY, TORTUOSITY_STATEMENT, aggregate_ratings, central_summary,
    tortuosity_by_smoothing)

MASK_DIRS = {"lung": ROOT / "data/raw/covid_ct/Lung_Mask",
             "infection": ROOT / "data/raw/covid_ct/Infection_Mask"}
E1_E2_N = 5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
        return c + ("+dirty" if d else "")
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def load_run(path: Path) -> tuple[dict, str]:
    mp = ROOT / path / "run_manifest.json"
    return json.loads(mp.read_text()), sha256_file(mp)


def stage_from(rec: dict | None, skipped: dict | None, ok_values=("ok",)) -> dict:
    if rec is not None:
        st = rec["status"]
        return {"status": "ok" if st in ok_values else st,
                "detail": "; ".join(rec.get("issues", []))}
    if skipped is not None:
        return {"status": "excluded", "detail": skipped["reason"]}
    return {"status": "not_run", "detail": "not included in the supplied run"}


def summarize(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    return f"{min(vals):.1f} / {statistics.median(vals):.1f} / {max(vals):.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("qc", "seg", "eval", "vlm"):
        ap.add_argument(f"--{a}-run", required=True, type=Path)
    ap.add_argument("--airway-run", type=Path, help="Phase 5 airway run (optional)")
    ap.add_argument("--sensitivity-run", type=Path, help="Phase 5 sensitivity run (optional)")
    ap.add_argument("--vlm-compare-run", type=Path,
                    help="VLM run with an alternative image processor (optional; flags "
                         "outputs that change)")
    args = ap.parse_args()

    air_cases, stab = {}, {}
    if args.airway_run:
        air, air_h = load_run(args.airway_run)
        air_cases = {c["case_id"]: c for c in air["cases"]}
    if args.sensitivity_run:
        with open(ROOT / args.sensitivity_run / "stability_by_case.csv", newline="") as f:
            for r in csv.DictReader(f):
                stab.setdefault(r["case_id"], []).append(r)
    long_rows = []
    if args.sensitivity_run:
        with open(ROOT / args.sensitivity_run / "sensitivity_long.csv", newline="") as f:
            long_rows = list(csv.DictReader(f))
    rules = load_rules()

    qc, qc_h = load_run(args.qc_run)
    seg, seg_h = load_run(args.seg_run)
    ev, ev_h = load_run(args.eval_run)
    vlm, vlm_h = load_run(args.vlm_run)
    seg_cases = {c["case_id"]: c for c in seg["cases"]}
    seg_skip = {s["case_id"]: s for s in seg["skipped"]}
    ev_cases = {c["case_id"]: c for c in ev["cases"]}
    vlm_cases = {c["case_id"]: c for c in vlm["cases"]}
    vlm_skip = {s["case_id"]: s for s in vlm["skipped"]}
    vcfg = vlm["config"]

    case_ids = [c["case_id"] for c in qc["cases"]]
    run_id = f"P4_report_{len(case_ids)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    (run_dir / "cases").mkdir(parents=True)
    provenance = {
        "report_run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "inputs": {
            "qc_run": {"path": str(args.qc_run), "manifest_sha256": qc_h,
                       "git_commit": qc["provenance"].get("git_commit")},
            "seg_run": {"path": str(args.seg_run), "manifest_sha256": seg_h},
            "eval_run": {"path": str(args.eval_run), "manifest_sha256": ev_h},
            "vlm_run": {"path": str(args.vlm_run), "manifest_sha256": vlm_h,
                        "git_commit": vlm["provenance"].get("git_commit")},
            "airway_run": {"path": str(args.airway_run),
                           "manifest_sha256": air_h} if args.airway_run else None,
            "sensitivity_run": str(args.sensitivity_run) if args.sensitivity_run else None,
            "review_flag_rules": {"version": rules["rules_version"],
                                  "sha256": rules["_sha256"], "status": rules["status"]},
            "vlm_compare_run": str(args.vlm_compare_run) if args.vlm_compare_run else None,
        },
        "dataset": qc["provenance"].get("dataset"),
        "segmentation": {"packages": seg["provenance"]["packages"],
                         "weights_sha256": seg["provenance"]["weights_sha256"],
                         "gpu": seg["provenance"].get("gpu")},
        "vlm": {"model_id": vcfg["model"]["model_id"], "revision": vcfg["model"]["revision"],
                "weights_sha256": vlm.get("model_info", {}).get("weights_sha256"),
                "image_processor_class": vlm.get("model_info", {}).get("image_processor_class"),
                "config_sha256": vlm["provenance"]["config_sha256"],
                "prompt_version": vcfg["prompt"]["version"],
                "prompt_sha256": vlm["provenance"]["prompt_sha256"],
                "representation": vcfg["representation"],
                "generation": vcfg["generation"],
                "packages": vlm["provenance"]["packages"],
                "gpu": vlm["provenance"].get("gpu")},
    }

    cases, batch, failures, review, targets = [], [], [], [], []
    for qcc in qc["cases"]:
        cid = qcc["case_id"]
        print(f"[{cid}]", end=" ", flush=True)
        cdir = run_dir / "cases" / cid
        (cdir / "overlays").mkdir(parents=True)
        sc, ec, vc = seg_cases.get(cid), ev_cases.get(cid), vlm_cases.get(cid)

        stages = {
            "qc": {"status": qcc["status"], "detail": "; ".join(qcc.get("issues", []))},
            "segmentation": stage_from(sc, seg_skip.get(cid)),
            "segmentation_eval": stage_from(ec, None) if sc else
            {"status": "not_run", "detail": "no segmentation"},
            "vlm": stage_from(vc, vlm_skip.get(cid)),
        }
        ac = air_cases.get(cid)
        if args.airway_run:
            stages["airway"] = stage_from(ac, None) if sc else \
                {"status": "not_run", "detail": "no segmentation"}
        seg_excluded = stages["segmentation"]["status"] == "excluded"
        for st, rec in stages.items():
            rec["category"] = stage_category(st, rec["status"],
                                             seg_excluded and st in ("segmentation_eval", "airway"))

        reference = {"lung_volumes": qcc.get("reference_lung_volumes"),
                     "infection_burden": qcc.get("reference_infection_burden"),
                     "infection_by_side": None}
        geom_ok = all(g.get("match") for g in (qcc.get("geometry") or {}).values()) \
            and bool(qcc.get("geometry"))
        if geom_ok and (qcc.get("laterality") or {}).get("verified"):
            # labels 1/2 = patient left/right, verified from world coordinates in Phase 1
            reference["infection_by_side"] = infection_by_side(
                nib.load(str(MASK_DIRS["infection"] / f"{cid}.nii.gz")),
                nib.load(str(MASK_DIRS["lung"] / f"{cid}.nii.gz")), left_label=1, right_label=2)

        predicted = comparison = None
        if ec and ec.get("status") != "failed":
            predicted = {"lobes": ec.get("pred_lobes", {}),
                         "airways_vessels": ec.get("pred_airways_vessels", {}),
                         "laterality": ec.get("pred_laterality")}
            comparison = {k: ec.get(k) for k in ("lung_left", "lung_right", "lung_total")}

        vlm_block = None
        if vc is not None:
            vdir = ROOT / args.vlm_run / cid
            raw_p, parsed_p, mi_p = vdir / "vlm_raw.txt", vdir / "vlm_parsed.json", \
                vdir / "model_input.json"
            vlm_block = {
                "model_id": vcfg["model"]["model_id"], "model_revision": vcfg["model"]["revision"],
                "result": vc,
                "raw_text": raw_p.read_text(encoding="utf-8") if raw_p.exists() else None,
                "parsed": json.loads(parsed_p.read_text()) if parsed_p.exists() else None,
                "model_input": json.loads(mi_p.read_text()) if mi_p.exists() else {},
            }
            cmp_raw = cmp_parsed = None
            if args.vlm_compare_run:
                cdir_c = ROOT / args.vlm_compare_run / cid
                if (cdir_c / "vlm_raw.txt").exists():
                    cmp_raw = (cdir_c / "vlm_raw.txt").read_text(encoding="utf-8")
                if (cdir_c / "vlm_parsed.json").exists():
                    cmp_parsed = json.loads((cdir_c / "vlm_parsed.json").read_text())
            vlm_block["review_flags"] = vlm_review_flags(
                vlm_block["parsed"], vlm_block["raw_text"],
                vlm_block["model_input"].get("representation", {}).get("n_images"),
                reference["infection_by_side"], cmp_raw, cmp_parsed)
            for fl in vlm_block["review_flags"]:
                targets.append({"case_id": cid, **fl})
            for p in (raw_p, parsed_p, mi_p, vdir / "vlm_result.json"):
                if p.exists():
                    shutil.copy2(p, cdir / p.name)

        images = {}
        qc_imgs = [ROOT / p for p in qcc.get("qc_images", [])]
        ev_imgs = [ROOT / p for p in (ec or {}).get("qc_images", [])]
        for p in qc_imgs + ev_imgs:
            shutil.copy2(p, cdir / "overlays" / p.name)
        if qc_imgs:
            images["ref_overlays"] = "".join(img_tag(p.read_bytes(), p.stem) for p in qc_imgs)
        if ev_imgs:
            images["pred_overlays"] = "".join(img_tag(p.read_bytes(), p.stem) for p in ev_imgs)
        jpgs = sorted((ROOT / args.vlm_run / cid / "model_input").glob("slice_*.jpg"))
        if vc is not None and jpgs:
            sheet = contact_sheet(jpgs)
            (cdir / "model_input_sheet.png").write_bytes(sheet)
            images["model_input_sheet"] = img_tag(sheet, "model input")

        airway_block = None
        if ac is not None and ac.get("status") == "ok":
            adir = ROOT / args.airway_run / cid
            for name in ("airway_tree_qc.png", "airway_branches.csv", "airway_summary.json"):
                if (adir / name).exists():
                    shutil.copy2(adir / name, cdir / name)
            if (adir / "airway_tree_qc.png").exists():
                images["airway_qc"] = img_tag((adir / "airway_tree_qc.png").read_bytes(),
                                              "airway centreline")
            airway_block = {"summary": ac.get("summary"), "stability": stab.get(cid, [])}

        case = {"case_id": cid, "report_run_id": run_id, "stages": stages, "qc": qcc,
                "reference": reference, "predicted": predicted, "comparison": comparison,
                "vlm": vlm_block, "airway": airway_block,
                "provenance": {"ct_sha256": qcc["ct"]["sha256"],
                               "qc_run": str(args.qc_run), "seg_run": str(args.seg_run),
                               "eval_run": str(args.eval_run), "vlm_run": str(args.vlm_run),
                               "airway_run": str(args.airway_run) if args.airway_run else None,
                               "sensitivity_run": str(args.sensitivity_run)
                               if args.sensitivity_run else None,
                               "medgemma": f"{vcfg['model']['model_id']}@{vcfg['model']['revision']}",
                               "prompt_sha256": vlm["provenance"]["prompt_sha256"],
                               "report_git_commit": provenance["git_commit"]}}
        (cdir / "qc.json").write_text(json.dumps(qcc, indent=2))
        (cdir / "metrics.json").write_text(json.dumps(
            {"case_id": cid,
             "reference_from_expert_masks": reference,
             "predicted_by_totalsegmentator": predicted,
             "predicted_vs_reference": comparison}, indent=2))
        (cdir / "case_report.html").write_text(render_case_html(case, images), encoding="utf-8")

        for st, rec in stages.items():
            if rec["status"] not in ("ok",):
                failures.append({"case_id": cid, "stage": st, "status": rec["status"],
                                 "category": rec["category"], "reason": rec["detail"]})
        vres = (vlm_block or {}).get("result", {})
        parsed = (vlm_block or {}).get("parsed") or {}
        rb = reference["infection_burden"] or {}
        batch.append({
            "case_id": cid, "hu_valid": qcc["ct"]["hu_valid"],
            **{f"{k}_status": v["status"] for k, v in stages.items()},
            "parse_status": vres.get("parse_status", ""),
            "ref_lung_ml": (reference["lung_volumes"] or {}).get("total_lung_volume_ml"),
            "ref_infection_pct": rb.get("infection_burden_pct"),
            "dice_left": (comparison or {}).get("lung_left", {}).get("dice"),
            "dice_right": (comparison or {}).get("lung_right", {}).get("dice"),
            "vol_err_left_pct": (comparison or {}).get("lung_left", {}).get("volume_error_pct"),
            "vol_err_right_pct": (comparison or {}).get("lung_right", {}).get("volume_error_pct"),
            "seg_seconds": sum(t["elapsed_seconds"] for t in sc["tasks"].values()) if sc else None,
            "seg_peak_gpu_mb": max(t["peak_gpu_memory_mb"] for t in sc["tasks"].values()) if sc else None,
            "vlm_seconds": vres.get("generate_seconds"),
            "vlm_peak_gpu_mb": vres.get("peak_gpu_memory_mb"),
            "vlm_output_tokens": vres.get("output_tokens"),
            "model_assessability": parsed.get("assessability", ""),
            "model_opacity_presence": parsed.get("opacity_presence", ""),
            "model_laterality": (parsed.get("distribution") or {}).get("laterality", ""),
            "vlm_review_flags": ";".join(f["flag"] for f in (vlm_block or {}).get("review_flags", [])),
            "airway_tree_branches": ((ac or {}).get("summary") or {}).get("n_tree_branches"),
            "airway_unstable_metrics": ";".join(r["metric"] for r in stab.get(cid, [])
                                               if r["rating"] == "unstable"),
        })
        review.extend(reviewer_rows(case))
        cases.append({"case_id": cid, "stages": stages})
        print("ok")

    def write_csv(name, rows, cols):
        with open(run_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    write_csv("batch_results.csv", batch, list(batch[0]))
    write_csv("failures.csv", failures, ["case_id", "stage", "status", "category", "reason"])
    write_csv("review_targets.csv", targets,
              ["case_id", "flag", "kind", "rules_version", "detail"])
    write_csv("reviewer_template.csv", review, REVIEW_COLUMNS)
    (run_dir / "feasibility_summary.md").write_text(
        feasibility_summary(run_id, batch, failures, provenance, targets, stab, long_rows,
                            rules),
        encoding="utf-8")
    provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_manifest.json").write_text(json.dumps(
        {"run_id": run_id, "provenance": provenance, "cases": cases}, indent=2))
    print(f"done -> {run_dir}")
    return 0


def feasibility_summary(run_id: str, batch: list[dict], failures: list[dict],
                        prov: dict, targets: list[dict], stab: dict,
                        long_rows: list[dict] | None = None, rules: dict | None = None) -> str:
    hu = [b for b in batch if b["hu_valid"]]
    first = hu[:E1_E2_N]
    seg_ok = [b for b in hu if b["segmentation_eval_status"] == "ok"]
    vlm_run = [b for b in batch if b["vlm_status"] not in ("excluded", "not_run")]
    vlm_valid = [b for b in vlm_run if b["parse_status"] == "valid"]

    def rng(rows, key, nd=3):
        v = [r[key] for r in rows if r[key] is not None]
        return f"{min(v):.{nd}f} to {max(v):.{nd}f}" if v else "n/a"

    combos = {(b["model_assessability"], b["model_opacity_presence"], b["model_laterality"])
              for b in vlm_valid}
    e0 = batch[0]
    by_cat = {}
    for f in failures:
        k = f"{f['stage']}: {f['status']}"
        by_cat.setdefault(f["category"], {}).setdefault(k, set()).add(f["case_id"])

    def cat_lines(cat):
        d = by_cat.get(cat, {})
        return [f"- {k}: {len(v)} case(s)" for k, v in sorted(d.items())] or ["- none"]

    flag_counts = {}
    for t in targets:
        flag_counts.setdefault((t["kind"], t["flag"]), []).append(
            t["case_id"].replace("coronacases_", ""))

    def flag_lines(kind):
        return [f"- {f}: cases {', '.join(sorted(v))}"
                for (k, f), v in sorted(flag_counts.items()) if k == kind] or ["- none"]

    stab_rows = [r for rows in stab.values() for r in rows]
    agg = aggregate_ratings(stab_rows)
    tort = tortuosity_by_smoothing(long_rows or [])
    air_rows = [b for b in batch if b.get("airway_tree_branches") is not None]

    v = prov["vlm"]
    lines = [
        "# Feasibility summary - lung CT pipeline",
        "",
        f"Report run `{run_id}`; code `{prov['git_commit']}`. Measured results only; no "
        "accuracy, efficacy or diagnostic claim. Dataset: COVID-19-CT-Seg (Zenodo 3757476), "
        f"{len(batch)} CTs, all COVID-19 positive -> disease specificity cannot be estimated.",
        "",
        "## Setup",
        f"- Segmentation: TotalSegmentator {prov['segmentation']['packages'].get('TotalSegmentator')} "
        f"on {prov['segmentation'].get('gpu')}",
        f"- LVLM: {v['model_id']} @ {v['revision'][:8]}, bf16, no quantization, "
        f"{v['representation']['name']} ({v['representation']['max_slices']} axial slices, "
        f"3 HU windows as RGB), prompt {v['prompt_version']} (sha256 {v['prompt_sha256'][:12]}), "
        f"greedy, max {v['generation']['max_new_tokens']} new tokens",
        "",
        "## Results",
        "| Exp | Sample | Question | Measured |",
        "|---|---|---|---|",
        f"| E0 | {e0['case_id']} | Pipeline executes with correct geometry? | QC {e0['qc_status']}, "
        f"segmentation {e0['segmentation_status']}, eval {e0['segmentation_eval_status']} "
        f"(Dice L/R {e0['dice_left'] or float('nan'):.3f}/{e0['dice_right'] or float('nan'):.3f}), "
        f"VLM {e0['vlm_status']} ({e0['parse_status'] or 'n/a'}) |",
        f"| E1 | first {len(first)} HU CTs | Lung segmentation consistent with supplied masks? | "
        f"Dice left {rng(first, 'dice_left')}, right {rng(first, 'dice_right')}; volume error "
        f"left {rng(first, 'vol_err_left_pct', 2)} %, right {rng(first, 'vol_err_right_pct', 2)} % |",
        f"| E2 | same {len(first)} | LVLM produces reviewable observations? | "
        f"{sum(b['parse_status'] == 'valid' for b in first)}/{len(first)} schema-valid; raw "
        f"outputs preserved; reviewer sheet rows: see reviewer_template.csv |",
        f"| E3 | all {len(batch)} CTs | What fails across the batch? | "
        f"{len(hu)} HU-valid / {len(batch) - len(hu)} display-scaled (excluded from HU stages); "
        f"segmentation+eval ok {len(seg_ok)}/{len(hu)}; VLM run {len(vlm_run)}, schema-valid "
        f"{len(vlm_valid)}/{len(vlm_run)} |",
        "| E4 | - | Comparison with paired reports (CT-RATE) | not run |",
        f"| P5 | {len(air_rows)} HU CTs | Python airway centreline metrics (not AirQuant) | "
        f"{len(air_rows)} processed; processing stability per metric below; unvalidated |",
        "",
        f"E3 all-HU segmentation: Dice left {rng(seg_ok, 'dice_left')}, right "
        f"{rng(seg_ok, 'dice_right')}; volume error left {rng(seg_ok, 'vol_err_left_pct', 2)} %, "
        f"right {rng(seg_ok, 'vol_err_right_pct', 2)} %.",
        "",
        "## Runtime and GPU memory (min / median / max per case, RTX 3090 24 GB)",
        f"- Segmentation (2 tasks): {summarize([b['seg_seconds'] for b in batch if b['seg_seconds']])} s; "
        f"peak {summarize([b['seg_peak_gpu_mb'] for b in batch if b['seg_peak_gpu_mb']])} MB",
        f"- MedGemma generate: {summarize([b['vlm_seconds'] for b in vlm_run if b['vlm_seconds']])} s; "
        f"peak {summarize([b['vlm_peak_gpu_mb'] for b in vlm_run if b['vlm_peak_gpu_mb']])} MB",
        "",
        "## Execution failures (stage ran and failed)",
        *cat_lines("execution_failure"),
        "",
        "## Planned exclusions (by protocol, not failures)",
        *cat_lines("planned_exclusion"),
        "- reason: display-scaled 8-bit volumes cannot be windowed in HU; excluded before "
        "HU-dependent stages",
        "",
        "## QC warnings",
        *cat_lines("qc_warning"),
        *(["", "## Not included in supplied runs", *cat_lines("not_in_run")]
          if "not_in_run" in by_cat else []),
        "",
        "## Model output variability and automated flags",
        f"- Distinct (assessability, opacity_presence, laterality) combinations across "
        f"{len(vlm_valid)} valid outputs: {len(combos)} -> {sorted(combos)}"
        + (" - identical structured output for every case; cohort and model tendency "
           "cannot be separated in an all-COVID set" if len(combos) == 1 else ""),
        f"- Flag rules: {rules['rules_version']} (sha256 {rules['_sha256'][:12]}), "
        "configs/review_flag_rules_v1.json. EXPLORATORY: written after inspecting this pilot, "
        "not prospectively defined evaluation criteria. Listed in review_targets.csv.",
        "- Incorrect descriptions of the supplied model input (checkable against the logged "
        "input):",
        *["  " + x for x in flag_lines("input_description_error")],
        "- Review flags (prioritise clinician review; not confirmed errors - the infection "
        "annotation is not an exhaustive opacity annotation):",
        *["  " + x for x in flag_lines("review_flag")],
        "- Processing sensitivity (output changed with the image-processor implementation; "
        "comparison run available for 5 cases only):",
        *["  " + x for x in flag_lines("processing_sensitivity")],
        *(["", "## Airway measurements: parameter stability (not anatomical accuracy)",
           central_summary(stab_rows),
           "",
           PARAMETER_VS_ANATOMY,
           "",
           "| Metric | stable | moderate | unstable | unstable cases |",
           "|---|---|---|---|---|",
           *[f"| {m} | {a['stable']} | {a['moderate']} | {a['unstable']} | "
             f"{', '.join(c.replace('coronacases_', '') for c in a['unstable_cases']) or '-'} |"
             for m, a in agg.items()],
           "",
           "Counts are aggregated from the same per-case ratings shown in each case report "
           "(stability_by_case.csv). Branch counts, generation counts beyond the main bronchi "
           "and cycle counts are unstable in many cases; generation numbers beyond ~3 are graph "
           "depth, not anatomical generation (visual review). No diameter, tapering or wall "
           "thickness."]
          if stab_rows else []),
        *(["", "### Tortuosity across centreline-smoothing settings (median per case; range "
           "across cases)",
           "| Metric | Smoothing | Tortuosity range | Tortuosity - 1 range | n |",
           "|---|---|---|---|---|",
           *[f"| {m} | {lab} | {v['abs'][0]:.3f} - {v['abs'][1]:.3f} | "
             f"{v['excess'][0]:.3f} - {v['excess'][1]:.3f} | {v['n']} |"
             for m, d in tort.items() for lab, v in d.items()],
           "",
           TORTUOSITY_STATEMENT + " The excess over 1, which carries the tortuosity "
           "information, changes several-fold between smoothing settings."]
          if tort and any(tort.values()) else []),
        "",
        "## Limitations",
        "- n is small; all cases are COVID-19 positive from public sources; no healthy controls.",
        "- Infection masks are COVID-19 infection annotations, not opacity-type, GGO or fibrosis labels.",
        "- Lobe, airway and vessel volumes have no reference in this dataset (unvalidated).",
        "- Model observations are unvalidated and await clinician review; model statements are not "
        "calibrated probabilities. Free-text was observed to change under a minor image-processor "
        "change (Phase 3 sensitivity check).",
        "- Display-scaled (8-bit) cases cannot be windowed in HU and were excluded from HU-dependent "
        "stages rather than rescaled.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
