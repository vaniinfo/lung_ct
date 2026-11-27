"""Fetch a small, mixed CT-RATE validation subset and write a Phase-1-style QC manifest (vlm env).

CT-RATE (ibrahimhamamci/CT-RATE) is gated on Hugging Face: accept its terms with your account and
provide a token in the HF_TOKEN environment variable (never in a file). Licence CC BY-NC-SA 4.0:
research use, no redistribution. Only the VALIDATION split is used, because COLIPRI and CT-CLIP
were trained on the train split. All CT-RATE scans are non-contrast.

Selection (deterministic): one reconstruction (_a_1) per patient; `--normals` scans with no labelled
abnormality; the rest chosen greedily so that each target condition appears at least once,
preferring scans with few labels (clearer cases).

Intensities and geometry: scanner padding (<= -2000) is set to air (-1024); values are converted to HU with the metadata RescaleSlope/Intercept only
if the stored values are not already HU; voxel spacing is taken from the metadata if the file
header disagrees (early CT-RATE files had wrong headers). Left/right is NOT trusted until the
anatomical check after segmentation (heart left, liver right).

Writes:
  data/raw/ct_rate/volumes/<VolumeName>          HU NIfTI used by the pipeline
  data/raw/ct_rate/cases.json                    labels + radiology report text per case (ground truth)
  runs/P1_qc_ctrate_<n>cases_<time>/run_manifest.json   QC records in the Phase 1 format
Usage:
  environments/vlm/Scripts/python.exe scripts/ctrate_fetch.py --n 10 --normals 2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from lung_ct.ingest.nifti_loader import assess_hu  # noqa: E402

REPO = "ibrahimhamamci/CT-RATE"
OUT = ROOT / "data" / "raw" / "ct_rate"
CSV_FILES = {"labels": "dataset/multi_abnormality_labels/valid_predicted_labels.csv",
             "reports": "dataset/radiology_text_reports/validation_reports.csv",
             "metadata": "dataset/metadata/validation_metadata.csv"}
TARGETS = ["Consolidation", "Lung nodule", "Pleural effusion", "Emphysema", "Bronchiectasis",
           "Lymphadenopathy", "Cardiomegaly", "Atelectasis", "Pulmonary fibrotic sequela",
           "Interlobular septal thickening", "Pericardial effusion", "Mosaic attenuation pattern"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def col(row: dict, *names: str):
    low = {k.lower().replace(" ", "").replace("_", ""): k for k in row}
    for n in names:
        k = low.get(n.lower().replace(" ", "").replace("_", ""))
        if k is not None and row[k] not in ("", None):
            return row[k]
    return None


def parse_floats(s) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", str(s))] if s is not None else []


# Smooth / standard reconstruction kernels, most preferred first (density measurements such as
# LAA-950 and calcium need a standard kernel; sharp lung kernels like Philips YA or Siemens Bl56f
# add grain). Checked 2026-11-18: 1447 of 1564 validation scans have one of these.
SOFT_KERNELS = ["B", "A", "Br36d", "Br40f", "Br44f", "SB", "SA"]


def kernel_of(meta_row: dict | None) -> str:
    return re.sub(r"[\[\]' ]", "", str((meta_row or {}).get("ConvolutionKernel", ""))).split(",")[0]


def pick_recon(volume_name: str, met: dict, recon: str) -> tuple[str, str]:
    """Return (volume name to use, note). recon 'first' keeps _a_1; 'soft' picks the most preferred
    standard-kernel reconstruction of the same scan, else falls back to the given volume."""
    if recon == "first":
        return volume_name, f"first reconstruction ({kernel_of(met.get(volume_name))})"
    scan = volume_name.rsplit("_", 1)[0]  # valid_12_a_1.nii.gz -> valid_12_a
    cands = sorted(v for v in met if v.startswith(scan + "_"))
    ranked = [(SOFT_KERNELS.index(kernel_of(met[v])), v) for v in cands if kernel_of(met[v]) in SOFT_KERNELS]
    if ranked:
        v = min(ranked)[1]
        return v, f"soft reconstruction ({kernel_of(met[v])}; others: " + \
            ", ".join(f"{c.rsplit('_', 1)[1][:-7]}={kernel_of(met[c])}" for c in cands if c != v) + ")"
    return volume_name, f"NO soft reconstruction available, kept {kernel_of(met.get(volume_name))}"


def patient_of(volume_name: str) -> str:
    return "_".join(volume_name.split("_")[:2])  # valid_123_a_1.nii.gz -> valid_123


def select(labels: list[dict], n: int, n_normal: int, seed: int, exclude_patients: set | None = None,
           min_per_target: int = 1) -> list[dict]:
    abn = [k for k in labels[0] if k != "VolumeName"]
    rows = [r for r in labels if r["VolumeName"].endswith("_a_1.nii.gz")
            and patient_of(r["VolumeName"]) not in (exclude_patients or set())]
    rng = random.Random(seed)
    rng.shuffle(rows)
    pos = lambda r: [a for a in abn if str(r[a]).strip() in ("1", "1.0")]  # noqa: E731
    normals = [r for r in rows if not pos(r)][:n_normal]
    chosen = list(normals)
    covered = {t: 0 for t in TARGETS}
    pool = [r for r in rows if pos(r)]
    while len(chosen) < n and pool:
        def score(r):
            new = sum(1 for t in pos(r) if t in covered and covered[t] < min_per_target)
            return (new, -len(pos(r)))
        best = max(pool, key=score)
        if score(best)[0] == 0:  # all targets covered: add clear single/double-finding cases
            best = min(pool, key=lambda r: len(pos(r)))
        chosen.append(best)
        for t in pos(best):
            if t in covered:
                covered[t] += 1
        pool.remove(best)
    return [{"VolumeName": r["VolumeName"], "labels": pos(r)} for r in chosen]


def convert(src: Path, dst: Path, meta: dict | None, flip_ap: bool = False) -> dict:
    img = nib.load(str(src))
    data = np.asanyarray(img.dataobj).astype(np.float32)  # applies header scl_slope/inter if set
    info = {"header_zooms": [round(float(z), 4) for z in img.header.get_zooms()[:3]]}
    ok, reason = assess_hu(data)
    # Only rescale data that has no negative values (raw detector numbers, as in dataset/valid).
    # dataset/valid_fixed is already in HU with -8192 padding; rescaling it again pushes every voxel to air.
    if not ok and meta and float(data.min()) >= 0:
        # e.g. CT-RATE stores raw detector values (0..) with RescaleIntercept -8192 in the metadata
        slope = parse_floats(col(meta, "RescaleSlope"))
        inter = parse_floats(col(meta, "RescaleIntercept"))
        if slope and inter:
            data = data * slope[0] + inter[0]
            info["rescale_applied"] = [slope[0], inter[0]]
    # AFTER rescaling: scanner padding outside the reconstruction circle (raw 0 -> -8192 HU) is not
    # tissue. Values below -2000 HU are physically impossible, so they are set to air (-1024), counted.
    pad = data <= -2000
    if pad.any():
        info["padding_replaced"] = {"voxels": int(pad.sum()), "fraction": round(float(pad.mean()), 5),
                                    "values": sorted({int(v) for v in np.unique(data[pad])[:5]}), "set_to": -1024}
        data[pad] = -1024
    ok, reason = assess_hu(data)
    # sanity: a chest CT is never mostly padding or a single value
    if info.get("padding_replaced", {}).get("fraction", 0) > 0.5 or float(np.percentile(data, 95)) <= -1000:
        ok, reason = False, f"conversion sanity check failed: {info.get('padding_replaced', {}).get('fraction')} padding, 95th percentile {float(np.percentile(data, 95)):.0f} HU"
    info.update(hu_valid=ok, hu_reason=reason)
    affine = img.affine.copy()
    xy = parse_floats(col(meta, "XYSpacing")) if meta else []
    z = parse_floats(col(meta, "ZSpacing")) if meta else []
    if len(xy) >= 2 and z:
        meta_sp = np.array([xy[0], xy[1], z[0]])
        hdr_sp = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        info["metadata_spacing"] = [round(float(v), 4) for v in meta_sp]
        if np.any(np.abs(hdr_sp - meta_sp) / meta_sp > 0.02):
            dirs = affine[:3, :3] / np.where(hdr_sp > 0, hdr_sp, 1)
            affine[:3, :3] = dirs * meta_sp
            info["affine_source"] = "header directions x metadata spacing (header spacing disagreed)"
        else:
            info["affine_source"] = "header (agrees with metadata spacing)"
    else:
        info["affine_source"] = "header (no metadata spacing)"
    if flip_ap:
        # CT-RATE volumes carry a placeholder identity header, and their front/back (y) axis is stored
        # reversed: viewed front-up, the spine and the scanner table sit at the TOP. Checked on 2026-11-16 by
        # rendering valid_5 next to a known-good case and by the heart-vs-lung centroid test in
        # anatomy_contrast_check.py (failed on all 9 readable cases while left/right and top/bottom passed).
        # Only valid for an axis-aligned header, so refuse anything else.
        if not np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3]))):
            raise ValueError("flip_ap needs an axis-aligned header")
        data = np.ascontiguousarray(data[:, ::-1, :])
        info["ap_flip_applied"] = "array y axis reversed (CT-RATE front/back fix)"
    out = nib.Nifti1Image(np.round(data).astype(np.int16), affine)
    out.header.set_xyzt_units("mm")
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))
    info.update(shape=list(out.shape), spacing_mm=[round(float(v), 4) for v in np.sqrt((affine[:3, :3] ** 2).sum(axis=0))],
                orientation="".join(nib.orientations.aff2axcodes(affine)),
                intensity_range=[float(data.min()), float(data.max())])
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--source", choices=["fixed", "original"], default="fixed",
                    help="fixed = dataset/valid_fixed (headers rebuilt by the CT-RATE maintainers from the DICOM "
                         "metadata; default). original = dataset/valid (placeholder identity header: front/back "
                         "and left/right are not trustworthy)")
    ap.add_argument("--flip-ap", action="store_true",
                    help="experimental, superseded by --source fixed: reverse the front/back axis only")
    ap.add_argument("--normals", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--set-name", default="pilot10",
                    help="name stored with each case in cases.json (e.g. pilot10, test20)")
    ap.add_argument("--min-per-target", type=int, default=1,
                    help="greedy selection keeps adding cases until each target finding appears this often")
    ap.add_argument("--recon", choices=["first", "soft"], default="first",
                    help="first = _a_1 as before (pilot10, test20); soft = the standard-kernel reconstruction")
    ap.add_argument("--same-patients-as", default=None, metavar="SET",
                    help="re-fetch the scans of an existing set (e.g. test20) instead of selecting new patients")
    ap.add_argument("--exclude-used", action="store_true",
                    help="skip every patient already listed in data/raw/ct_rate/cases.json (fresh test set)")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download

    cache = OUT / "hf_cache"
    csvs = {k: Path(hf_hub_download(REPO, v, repo_type="dataset", cache_dir=str(cache))) for k, v in CSV_FILES.items()}
    labels, reports, metadata = (read_csv(csvs[k]) for k in ("labels", "reports", "metadata"))
    rep = {r["VolumeName"]: r for r in reports}
    met = {r["VolumeName"]: r for r in metadata}
    prev = {}
    if (OUT / "cases.json").exists():
        prev = json.loads((OUT / "cases.json").read_text(encoding="utf-8")).get("cases", {})
    lab_by = {r["VolumeName"]: r for r in labels}
    abn = [k for k in labels[0] if k != "VolumeName"]
    pos_of = lambda name: [a for a in abn if str(lab_by.get(name, {}).get(a, "")).strip() in ("1", "1.0")]  # noqa: E731
    if args.same_patients_as:
        ids = sorted(c for c, v in prev.items()
                     if v.get("set") == args.same_patients_as or args.same_patients_as in v.get("also_in_sets", []))
        if not ids:
            raise SystemExit(f"no cases with set={args.same_patients_as} in cases.json")
        sel = [{"VolumeName": f"{c}.nii.gz", "labels": pos_of(f"{c}.nii.gz")} for c in ids]
        print(f"re-fetching the {len(sel)} scans of set {args.same_patients_as}")
    else:
        used = {patient_of(c) for c in prev} if args.exclude_used else set()
        if used:
            print(f"excluding {len(used)} patients already used: {sorted(used)}")
        sel = select(labels, args.n, args.normals, args.seed, used, args.min_per_target)
        assert not used & {patient_of(s["VolumeName"]) for s in sel}, "selected an excluded patient"
    for s_ in sel:
        v, note = pick_recon(s_["VolumeName"], met, args.recon)
        s_["recon_note"] = note
        if v != s_["VolumeName"]:
            s_["VolumeName"] = v
            s_["labels"] = pos_of(v)  # labels are per volume; same study, so normally identical
    print(f"selected {len(sel)} cases:")
    for s in sel:
        print(f"  {s['VolumeName']}: {', '.join(s['labels']) or 'no labelled abnormality'}  [{s['recon_note']}]")

    split_dir = "valid_fixed" if args.source == "fixed" else "valid"
    print(f"volume source: dataset/{split_dir}/")
    set_tag = "" if args.set_name in ("pilot10", "test20") else f"{args.set_name}_"  # older sets keep old names
    run = ROOT / "runs" / f"P1_qc_ctrate_{set_tag}{len(sel)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True)
    cases, gt = [], {}
    for s in sel:
        name = s["VolumeName"]
        cid = name[:-len(".nii.gz")]
        pid = "_".join(cid.split("_")[:2])
        scan = "_".join(cid.split("_")[:3])
        rec = {"case_id": cid, "status": "ok", "issues": [], "dataset": "CT-RATE validation", "reconstruction": s["recon_note"],
               "contrast": "non-contrast (all CT-RATE scans)"}
        try:
            src = Path(hf_hub_download(REPO, f"dataset/{split_dir}/{pid}/{scan}/{name}", repo_type="dataset",
                                       cache_dir=str(cache)))
            dst = OUT / "volumes" / name
            info = convert(src, dst, met.get(name), flip_ap=args.flip_ap)
            rec["ct"] = {"file": str(dst.relative_to(ROOT)), "source_folder": f"dataset/{split_dir}", "sha256": sha256_file(dst),
                         "source_sha256": sha256_file(src), **info}
            rec["laterality"] = {"verified": False,
                                 "reason": "pending anatomical check after segmentation (heart left, liver right)"}
            if not info["hu_valid"]:
                rec["status"] = "warn"
                rec["issues"].append(f"not HU after conversion: {info['hu_reason']}")
            print(f"[{cid}] ok {info['shape']} {info['spacing_mm']} {info['orientation']} {info['hu_reason']}")
        except Exception as e:  # recorded, never dropped
            rec.update(status="failed", issues=[f"{type(e).__name__}: {e}"])
            print(f"[{cid}] FAILED {e}")
        cases.append(rec)
        r = rep.get(name, {})
        gt[cid] = {"labels": s["labels"], "set": args.set_name, "selection_seed": args.seed,
                   "reconstruction": s["recon_note"], "kernel": kernel_of(met.get(name)),
                   "clinical_information": col(r, "ClinicalInformation_EN"),
                   "technique": col(r, "Technique_EN"),
                   "findings": col(r, "Findings_EN"),
                   "impression": col(r, "Impressions_EN"),
                   "metadata": {k: met.get(name, {}).get(k) for k in ("Manufacturer", "PatientSex", "PatientAge",
                                                                   "ConvolutionKernel", "SeriesDescription")}}
    for k, v in prev.items():  # keep earlier sets; never silently drop a case
        v.setdefault("set", "pilot10")
    merged = dict(prev)
    for cid, v in gt.items():  # a case already in an earlier set keeps its set and gains this one
        if cid in merged:
            extra = merged[cid].setdefault("also_in_sets", [])
            if args.set_name != merged[cid].get("set") and args.set_name not in extra:
                extra.append(args.set_name)
        else:
            merged[cid] = v
    (OUT / "cases.json").write_text(json.dumps(
        {"source": f"https://huggingface.co/datasets/{REPO}", "licence": "CC BY-NC-SA 4.0 (research use, no redistribution)",
         "split": "validation", "labels_are": "report-derived (RadBERT), not image annotations",
         "sets": sorted({v.get("set", "pilot10") for v in merged.values()}
                        | {x for v in merged.values() for x in v.get("also_in_sets", [])}),
         "cases": merged}, indent=2, ensure_ascii=False), encoding="utf-8")
    (run / "run_manifest.json").write_text(json.dumps(
        {"run_id": run.name, "provenance": {"finished_utc": datetime.now(timezone.utc).isoformat(), "repo": REPO,
                                            "csv_sha256": {k: sha256_file(p) for k, p in csvs.items()}},
         "cases": cases}, indent=2), encoding="utf-8")
    bad = sum(c["status"] == "failed" for c in cases)
    print(f"done: {len(cases) - bad} ok, {bad} failed -> {run}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
