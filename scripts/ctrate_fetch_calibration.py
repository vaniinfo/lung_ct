"""Fetch a random, label-blind set of new CT-RATE validation patients for calibration (vlm env).

Plan: docs/prompt_tests/TR02_calibration_400_PLAN.md. Differences to scripts/ctrate_fetch.py:
  selection   random over patients (sorted, shuffled with the seed, first N), never looking at a label;
              patients already in data/raw/ct_rate/cases.json are excluded;
              the first half of the shuffled order is the "fit" part, the second half the "confirm" part
  storage     everything under --root (default D:/datasets/ct_rate/calib400), outside the repository:
              hf_cache/ (downloads, kept), volumes/ (HU NIfTI), selection.json, fetch_records.jsonl, cases.json, PROVENANCE.md
  resumable   a scan with an "ok" record and its volume on disk is not fetched again
Conversion to HU, spacing repair and padding handling are those of ctrate_fetch.convert, unchanged.
Left / right is NOT trusted until the anatomical check (scripts/anatomy_contrast_check.py).

The Hugging Face token is read from the HF_TOKEN environment variable only; it is never printed or written.

Usage:
  environments/vlm/Scripts/python.exe scripts/ctrate_fetch_calibration.py [--n 400] [--select-only]
Writes also runs/P1_qc_ctrate_calib<n>_<time>/run_manifest.json (QC records in the Phase 1 format, absolute volume paths).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_ROOT = "D:/datasets/ct_rate/calib400"
SEED = 20261126
CSV_DIR = "data/raw/ct_rate/hf_cache/datasets--ibrahimhamamci--CT-RATE/snapshots"


def patient_of(volume_name: str) -> str:
    return "_".join(volume_name.split("_")[:2])


def patient_number(p: str) -> int:
    return int(p.split("_")[1])


def select_patients(volume_names: list[str], used_patients: set[str], n: int, seed: int) -> list[dict]:
    """Label-blind: patients with a scan 'a', sorted by number, shuffled, first n. Order index decides the part."""
    pats = sorted({patient_of(v) for v in volume_names if v.split("_")[2] == "a"} - set(used_patients), key=patient_number)
    random.Random(seed).shuffle(pats)
    chosen = pats[:n]
    half = len(chosen) // 2
    return [{"patient": p, "order": i, "part": "fit" if i < half else "confirm"} for i, p in enumerate(chosen)]


def local_csvs() -> dict:
    """The three CT-RATE validation tables already in the repository's cache (no network)."""
    snaps = sorted((ROOT / CSV_DIR).glob("*/dataset"))
    if not snaps:
        raise SystemExit("CT-RATE tables not found in data/raw/ct_rate/hf_cache; run scripts/ctrate_fetch.py once first")
    d = snaps[-1]
    return {"labels": d / "multi_abnormality_labels/valid_predicted_labels.csv", "reports": d / "radiology_text_reports/validation_reports.csv",
            "metadata": d / "metadata/validation_metadata.csv"}


def build_selection(n: int, seed: int) -> dict:
    import ctrate_fetch as cf
    csvs = local_csvs()
    labels, metadata = cf.read_csv(csvs["labels"]), cf.read_csv(csvs["metadata"])
    met = {r["VolumeName"]: r for r in metadata}
    used = {patient_of(c) for c in json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]}
    names = [r["VolumeName"] for r in labels]
    sel = select_patients(names, used, n, seed)
    have = set(names)
    for s in sel:
        cands = sorted(v for v in have if v.startswith(s["patient"] + "_a_"))
        first = f"{s['patient']}_a_1.nii.gz" if f"{s['patient']}_a_1.nii.gz" in have else cands[0]
        v, note = cf.pick_recon(first, met, "soft")
        s.update(volume=v if v in have else first, reconstruction=note)
    return {"plan": "docs/prompt_tests/TR02_calibration_400_PLAN.md", "seed": seed, "requested": n, "validation_patients": len({patient_of(v) for v in names}),
            "excluded_patients_already_used": len(used), "rule": "patients with a scan 'a', sorted by number, shuffled with the seed, first n; first half = fit, second half = confirm; no label read",
            "created_utc": datetime.now(timezone.utc).isoformat(), "csv_sha256": {k: cf.sha256_file(p) for k, p in csvs.items()}, "selection": sel}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--select-only", action="store_true", help="write selection.json and stop (no network)")
    args = ap.parse_args()
    import ctrate_fetch as cf
    root = Path(args.root)
    if str(root.resolve()).lower().startswith("c:"):
        raise SystemExit("the data root must not be on C:")
    root.mkdir(parents=True, exist_ok=True)
    sel_path = root / "selection.json"
    if sel_path.exists():                                   # the selection is made once and never redrawn
        selection = json.loads(sel_path.read_text(encoding="utf-8"))
        if selection["seed"] != args.seed or selection["requested"] != args.n:
            raise SystemExit(f"{sel_path} was made with seed {selection['seed']} and n {selection['requested']}; use a new --root for another selection")
    else:
        selection = build_selection(args.n, args.seed)
        sel_path.write_text(json.dumps(selection, indent=1), encoding="utf-8")
    sel = selection["selection"]
    print(f"selection: {len(sel)} patients ({sum(s['part'] == 'fit' for s in sel)} fit, {sum(s['part'] == 'confirm' for s in sel)} confirm) -> {sel_path}", flush=True)
    if args.select_only:
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (CT-RATE is gated); start fetch_ctrate_calibration.bat")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    csvs = local_csvs()
    labels, reports, metadata = (cf.read_csv(csvs[k]) for k in ("labels", "reports", "metadata"))
    rep, met, lab_by = ({r["VolumeName"]: r for r in t} for t in (reports, metadata, labels))
    abn = [k for k in labels[0] if k != "VolumeName"]
    cache, rec_path, lock = root / "hf_cache", root / "fetch_records.jsonl", threading.Lock()
    done = {}
    if rec_path.exists():
        for line in rec_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["case_id"]] = r                           # the latest record of a scan counts
    stop = threading.Event()

    def fetch(s: dict) -> dict:
        name = s["volume"]
        cid, pid, scan = name[:-len(".nii.gz")], s["patient"], "_".join(name.split("_")[:3])
        dst = root / "volumes" / name
        old = done.get(cid)
        if old and old["status"] != "failed" and dst.exists():
            return old
        rec = {"case_id": cid, "status": "ok", "issues": [], "dataset": "CT-RATE validation", "set": "calib400", "part": s["part"],
               "reconstruction": s["reconstruction"], "contrast": "non-contrast (all CT-RATE scans)"}
        if stop.is_set():
            rec.update(status="failed", issues=["not fetched: the run stopped after an access error"])
            return rec
        try:
            src = Path(hf_hub_download(cf.REPO, f"dataset/valid_fixed/{pid}/{scan}/{name}", repo_type="dataset", cache_dir=str(cache), token=token))
            info = cf.convert(src, dst, met.get(name))
            rec["ct"] = {"file": str(dst).replace("\\", "/"), "source_folder": "dataset/valid_fixed", "sha256": cf.sha256_file(dst), "source_sha256": cf.sha256_file(src),
                         "source_bytes": src.stat().st_size, "bytes": dst.stat().st_size, **info}
            rec["laterality"] = {"verified": False, "reason": "pending anatomical check (heart left, liver right)"}
            if not info["hu_valid"]:
                rec["status"] = "warn"
                rec["issues"].append(f"not HU after conversion: {info['hu_reason']}")
            print(f"[{cid}] ok {info['shape']} {info['spacing_mm']} {info['hu_reason']}", flush=True)
        except GatedRepoError:
            stop.set()
            rec.update(status="failed", issues=["access refused (401 / 403): accept the CT-RATE terms with this account and use a valid read token"])
            print(f"[{cid}] ACCESS REFUSED: check the token and that the CT-RATE terms are accepted", flush=True)
        except Exception as e:  # recorded, never dropped
            rec.update(status="failed", issues=[f"{type(e).__name__}: {str(e)[:300]}"])
            print(f"[{cid}] FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        with lock, open(rec_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(fetch, sel))

    gt = {}
    for s, rec in zip(sel, records):
        name, cid = s["volume"], rec["case_id"]
        r = rep.get(name, {})
        gt[cid] = {"labels": [a for a in abn if str(lab_by.get(name, {}).get(a, "")).strip() in ("1", "1.0")], "set": "calib400", "part": s["part"], "order": s["order"],
                   "selection_seed": args.seed, "reconstruction": s["reconstruction"], "kernel": cf.kernel_of(met.get(name)),
                   "clinical_information": cf.col(r, "ClinicalInformation_EN"), "technique": cf.col(r, "Technique_EN"),
                   "findings": cf.col(r, "Findings_EN"), "impression": cf.col(r, "Impressions_EN"),
                   "metadata": {k: met.get(name, {}).get(k) for k in ("Manufacturer", "PatientSex", "PatientAge", "ConvolutionKernel", "SeriesDescription")}}
    (root / "cases.json").write_text(json.dumps(
        {"source": f"https://huggingface.co/datasets/{cf.REPO}", "licence": "CC BY-NC-SA 4.0 (research use, no redistribution)", "split": "validation",
         "labels_are": "report-derived (RadBERT), not image annotations", "sets": ["calib400"], "cases": gt}, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = [r for r in records if r["status"] != "failed"]
    size = sum(r["ct"]["bytes"] + r["ct"]["source_bytes"] for r in ok)
    (root / "PROVENANCE.md").write_text(
        f"# CT-RATE calibration set (calib400)\n\n- Source: https://huggingface.co/datasets/{cf.REPO} , folder dataset/valid_fixed (validation split)\n"
        "- Licence: CC BY-NC-SA 4.0. Research use, no redistribution. Gated: terms accepted by the account that downloaded it.\n"
        f"- Selection: {selection['rule']} (seed {selection['seed']}); see selection.json\n"
        f"- Scans: {len(ok)} fetched, {len(records) - len(ok)} failed; {size / 2**30:.1f} GB on disk (downloads in hf_cache/ plus converted volumes/)\n"
        f"- Last fetch finished (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        "- cases.json holds report text and report-derived labels: keep within the research group.\n", encoding="utf-8")
    run = ROOT / "runs" / f"P1_qc_ctrate_calib{len(sel)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps(
        {"run_id": run.name, "data_root": str(root).replace("\\", "/"), "selection": str(sel_path).replace("\\", "/"),
         "provenance": {"finished_utc": datetime.now(timezone.utc).isoformat(), "repo": cf.REPO, "csv_sha256": selection["csv_sha256"],
                        "bytes_on_disk": size, "scans_fetched": len(ok), "scans_failed": len(records) - len(ok)},
         "cases": records}, indent=2), encoding="utf-8")
    (root / "latest_qc_run.txt").write_text(f"runs/{run.name}\n", encoding="utf-8")
    bad = len(records) - len(ok)
    print(f"done: {len(ok)} ok, {bad} failed, {size / 2**30:.1f} GB -> {run}", flush=True)
    return 2 if stop.is_set() else 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
