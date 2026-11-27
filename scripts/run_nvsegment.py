"""Second segmentation (nvsegment env): NV-Segment-CT (the VISTA3D network) on HU-valid chest CT, all supported classes.

Runs in environments/nvsegment, on this computer only. The masks are predictions for comparison and viewing next to
TotalSegmentator. Nothing here feeds a measurement, a checklist answer or a report, and no existing file is changed.

Two passes per scan (configs/nvsegment_ct_v1.json):
  everything  the vendor's "segment everything" list (117 classes) -> one label map, one label per voxel
  extra       the supported classes the vendor leaves out of that list (class 'lung tumor', other tumour classes,
              hepatic vessel, bone lesion, airway) -> one mask each, because they overlap the organs of the first pass

Preprocessing and inference follow vista3d_pipeline.py of the pinned model revision step for step, but the network is
built from the installed MONAI and the weights are read as safetensors: the repository's Python files are not executed.

Per case (runs/<run>/<case>/):
  nvseg_labels.nii.gz      uint8 label map on the SOURCE grid (source affine; nearest-neighbour inverse of the resampling)
  extra/<class>.nii.gz     uint8 0/1 masks of the extra classes (written also when empty)
  volumes.csv              every class asked for: voxels and mL (voxels * voxel volume / 1000); 0 = nothing predicted
  seg_manifest.json        geometry checks, timings, hashes
Run-level: run_manifest.json, batch_results.csv, failures.csv.

Usage (Git Bash):
    environments/nvsegment/Scripts/python.exe scripts/run_nvsegment.py --qc-run runs/<P1_qc_verified_...> --cases valid_301_a_2
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = ROOT / "models" / "nvsegment" / "hub"
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def check_config(cfg: dict) -> None:
    """The two passes must not overlap and every label must have a name."""
    ev, ex = cfg["passes"]["everything"]["labels"], cfg["passes"]["extra"]["labels"]
    if set(ev) & set(ex) or len(set(ev)) != len(ev) or len(set(ex)) != len(ex):
        raise ValueError("label lists overlap or repeat")
    if max(ev + ex) > 255 or min(ev + ex) < 1:
        raise ValueError("labels must fit a uint8 label map and must not be 0")
    missing = [x for x in ev + ex if str(x) not in cfg["class_names"]]
    if missing:
        raise ValueError(f"no class name for labels {missing}")


def volume_rows(labels: np.ndarray, extra: dict, cfg: dict, voxel_mm3: float) -> list[dict]:
    """One row per class asked for. mL = voxels * voxel volume in mm3 / 1000. Classes with 0 voxels are kept."""
    counts = np.bincount(labels.ravel(), minlength=256)
    rows = []
    for x in cfg["passes"]["everything"]["labels"]:
        n = int(counts[x])
        rows.append({"label": x, "name": cfg["class_names"][str(x)], "pass": "everything", "voxels": n, "ml": round(n * voxel_mm3 / 1000.0, 3)})
    for x in cfg["passes"]["extra"]["labels"]:
        n = int(np.count_nonzero(extra[x]))
        rows.append({"label": x, "name": cfg["class_names"][str(x)], "pass": "extra", "voxels": n, "ml": round(n * voxel_mm3 / 1000.0, 3)})
    return rows


def snapshot_dir(mcfg: dict) -> Path:
    local = MODEL_CACHE / ("models--" + mcfg["model_id"].replace("/", "--")) / "snapshots" / mcfg["revision"]
    if not (local / mcfg["weights_file"]).exists():        # read from disk only; setup_nvsegment.bat does the download
        raise FileNotFoundError(f"{local / mcfg['weights_file']} not found - run setup_nvsegment.bat first")
    return local


def load_network(mcfg: dict, local: Path):
    import monai.networks.nets
    import torch
    from safetensors.torch import load_file

    wpath = local / mcfg["weights_file"]
    digest = sha256_file(wpath)
    if digest != mcfg["weights_sha256"]:
        raise RuntimeError(f"weights hash {digest} differs from the config")
    net = monai.networks.nets.vista3d132(encoder_embed_dim=48, in_channels=1)
    state = {k[len("network."):] if k.startswith("network.") else k: v for k, v in load_file(str(wpath)).items()}
    net.load_state_dict(state, strict=True)                # every weight must match the network: no silent partial load
    net = net.eval().to(mcfg["device"])
    info = {"local_snapshot": str(local), "weights_sha256": digest, "parameters": int(sum(p.numel() for p in net.parameters())),
            "state_dict_keys": len(state), "gpu_memory_after_load_mb": round(torch.cuda.memory_allocated() / 2**20)}
    return net, info


def preprocessing(cfg: dict):
    import torch
    from monai.transforms import (CastToTyped, Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped, LoadImaged, Orientationd,
                                  ScaleIntensityRanged, Spacingd)
    p = cfg["preprocessing"]
    return Compose([
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys="image", device=cfg["model"]["device"], track_meta=True),
        Spacingd(keys="image", pixdim=p["spacing_mm"], mode="bilinear"),
        CropForegroundd(keys="image", allow_smaller=True, margin=p["crop_margin_voxels"], source_key="image"),
        ScaleIntensityRanged(keys="image", a_min=p["hu_min"], a_max=p["hu_max"], b_min=0, b_max=1, clip=True),
        Orientationd(keys="image", axcodes="RAS"),
        CastToTyped(keys="image", dtype=torch.float32),
    ])


def run_pass(net, image, labels: list[int], cfg: dict):
    """Scores for the given classes on the preprocessed scan: tensor [classes, X, Y, Z] (still in the model's grid)."""
    import torch
    from monai.inferers import SlidingWindowInfererAdapt
    from monai.utils import set_determinism

    inf = cfg["inference"]
    set_determinism(seed=inf["seed"])
    cv = torch.as_tensor(labels, device=image.device).unsqueeze(-1)
    inferer = SlidingWindowInfererAdapt(roi_size=inf["roi_size"], sw_batch_size=inf["sw_batch_size"], overlap=inf["overlap"],
                                        with_coord=True, padding_mode=inf["padding_mode"])
    with torch.no_grad(), torch.autocast("cuda", enabled=bool(cfg["model"]["amp"])):
        out = inferer(image.unsqueeze(0), net, transpose=True, point_coords=None, point_labels=None, class_vector=cv,
                      prompt_class=cv.clone(), prev_mask=None, labels=None, label_set=None)
    return out[0]


def to_source_grid(pred, data: dict, pre):
    """Inverse of the preprocessing, nearest neighbour: prediction back on the grid of the source file."""
    from monai.transforms import Invertd
    out = Invertd(keys="pred", transform=pre, orig_keys="image", nearest_interp=True, to_tensor=True)({"image": data["image"], "pred": pred})
    return out["pred"]


def segment_case(cid: str, ct_path: Path, cfg: dict, case_dir: Path, net, pre) -> dict:
    import torch
    from monai.apps.vista3d.transforms import VistaPostTransformd

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    src = nib.load(str(ct_path))
    shape, aff = tuple(int(x) for x in src.shape[:3]), np.asarray(src.affine, float)
    zooms = [float(z) for z in src.header.get_zooms()[:3]]
    voxel_mm3 = float(abs(np.linalg.det(aff[:3, :3])))
    data = pre({"image": str(ct_path)})
    model_grid = [int(x) for x in data["image"].shape[1:]]
    issues, checks = [], {}

    # pass 1: everything -> one label per voxel (vendor post-transform: highest positive score, else background)
    ev = cfg["passes"]["everything"]["labels"]
    t1 = time.time()
    scores = run_pass(net, data["image"], ev, cfg)
    post = VistaPostTransformd(keys="pred")({"pred": scores, "label_prompt": torch.as_tensor(ev)})
    del scores
    lab_t = to_source_grid(post["pred"], data, pre)
    inv_aff = lab_t.affine.cpu().numpy().astype(float)
    lab_f = lab_t.cpu().numpy()[0]
    del post, lab_t
    nan1 = int(np.isnan(lab_f).sum())
    labels = np.nan_to_num(lab_f, nan=0.0).round().astype(np.uint8)
    sec_everything = round(time.time() - t1, 1)

    # pass 2: extra classes -> one mask each (score > 0)
    ex = cfg["passes"]["extra"]["labels"]
    t2 = time.time()
    scores = run_pass(net, data["image"], ex, cfg)
    binm = (scores > 0).to(torch.float32)
    del scores
    ex_t = to_source_grid(binm, data, pre)
    ex_f = ex_t.cpu().numpy()
    del binm, ex_t
    nan2 = int(np.isnan(ex_f).sum())
    extra = {x: (np.nan_to_num(ex_f[i], nan=0.0) > 0.5).astype(np.uint8) for i, x in enumerate(ex)}
    del ex_f
    sec_extra = round(time.time() - t2, 1)
    torch.cuda.empty_cache()

    # geometry: same grid as the source file, checked before anything is written
    checks["shape_equal"] = labels.shape == shape and all(m.shape == shape for m in extra.values())
    checks["inverted_affine_max_abs_diff_mm"] = float(np.abs(inv_aff - aff).max())
    checks["affine_equal"] = bool(checks["inverted_affine_max_abs_diff_mm"] < 1e-2)
    checks["labels_only_from_list"] = bool(set(np.unique(labels).tolist()) <= set([0] + ev))
    checks["voxels_without_prediction_after_inverse"] = {"everything": nan1, "extra": nan2}
    for k in ("shape_equal", "affine_equal", "labels_only_from_list"):
        if not checks[k]:
            issues.append(f"geometry check failed: {k}")
    if issues:
        raise RuntimeError("; ".join(issues) + f" {checks}")

    hdr = src.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr.set_slope_inter(None, None)
    files = {}
    p = case_dir / "nvseg_labels.nii.gz"
    nib.save(nib.Nifti1Image(labels, src.affine, hdr), str(p))
    files[p.name] = sha256_file(p)
    (case_dir / "extra").mkdir()
    for x, m in extra.items():
        p = case_dir / "extra" / f"{slug(cfg['class_names'][str(x)])}.nii.gz"
        nib.save(nib.Nifti1Image(m, src.affine, hdr), str(p))
        files[f"extra/{p.name}"] = sha256_file(p)

    rows = volume_rows(labels, extra, cfg, voxel_mm3)
    with open(case_dir / "volumes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "name", "pass", "voxels", "ml"])
        w.writeheader()
        w.writerows(rows)
    found = [r for r in rows if r["voxels"] > 0]
    empty = [r["name"] for r in rows if r["voxels"] == 0]
    rec = {"case_id": cid, "status": "ok", "issues": issues, "ct_file": str(ct_path.relative_to(ROOT)), "ct_sha256": sha256_file(ct_path),
           "source_shape": list(shape), "source_spacing_mm": zooms, "source_axis_codes": "".join(nib.aff2axcodes(aff)),
           "voxel_volume_mm3": voxel_mm3, "model_grid_shape": model_grid, "checks": checks,
           "classes_asked": len(rows), "classes_found": len(found), "classes_empty": len(empty), "empty_class_names": empty,
           "extra_ml": {r["name"]: r["ml"] for r in rows if r["pass"] == "extra"},
           "seconds_everything": sec_everything, "seconds_extra": sec_extra, "seconds_total": round(time.time() - t0, 1),
           "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20), "files_sha256": files}
    (case_dir / "seg_manifest.json").write_text(json.dumps(rec, indent=2))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qc-run", required=True, type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--n", type=int)
    g.add_argument("--cases", nargs="+")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "nvsegment_ct_v1.json")
    args = ap.parse_args()

    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    check_config(cfg)
    qc = json.loads((ROOT / args.qc_run / "run_manifest.json").read_text())
    qc_cases = {c["case_id"]: c for c in qc["cases"]}
    eligible, skipped = [], []
    for cid in (args.cases if args.cases else list(qc_cases)):
        c = qc_cases.get(cid)
        if c is None or c["status"] == "failed":
            skipped.append({"case_id": cid, "reason": "absent or failed in Phase 1 QC"})
        elif not c["ct"]["hu_valid"]:
            skipped.append({"case_id": cid, "reason": f"non-HU: {c['ct']['hu_reason']}"})
        elif "laterality" in c and not c["laterality"].get("verified"):
            skipped.append({"case_id": cid, "reason": "left / right not verified from anatomy; the model takes it from the file"})
        else:
            eligible.append(cid)
    if args.n:
        eligible = eligible[:args.n]

    import importlib.metadata as md
    run_id = f"P2_nvsegment_{len(eligible)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    manifest = {"run_id": run_id,
                "provenance": {"started_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(), "qc_run": str(args.qc_run),
                               "config_file": str(cfg_path.relative_to(ROOT)), "config_sha256": sha256_file(cfg_path),
                               "python": platform.python_version(), "platform": platform.platform(),
                               "packages": {p: md.version(p) for p in ("torch", "monai", "nibabel", "numpy", "scipy", "safetensors", "einops")}},
                "config": cfg, "skipped": skipped, "cases": []}
    print(f"{run_id}: {len(eligible)} cases, {len(skipped)} skipped")
    try:
        import torch
        manifest["provenance"]["gpu"] = torch.cuda.get_device_name(0)
        manifest["provenance"]["cuda_runtime"] = torch.version.cuda
        net, manifest["model_info"] = load_network(cfg["model"], snapshot_dir(cfg["model"]))
        pre = preprocessing(cfg)
    except Exception as e:  # noqa: BLE001
        manifest["model_load_error"] = f"{type(e).__name__}: {e}"
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"MODEL LOAD FAILED: {manifest['model_load_error']}")
        traceback.print_exc()
        return 2

    for cid in eligible:
        print(f"[{cid}] ...", end=" ", flush=True)
        case_dir = run_dir / cid
        case_dir.mkdir()
        try:
            rec = segment_case(cid, ROOT / qc_cases[cid]["ct"]["file"].replace("\\", "/"), cfg, case_dir, net, pre)
        except Exception as e:  # noqa: BLE001
            rec = {"case_id": cid, "status": "failed", "issues": [f"{type(e).__name__}: {e}"], "traceback": traceback.format_exc()}
            (case_dir / "seg_manifest.json").write_text(json.dumps(rec, indent=2))
        manifest["cases"].append(rec)
        print(rec["status"], {k: rec[k] for k in ("classes_found", "classes_empty", "seconds_total", "peak_gpu_memory_mb") if k in rec}, "; ".join(rec["issues"]))

    manifest["provenance"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    cols = ["case_id", "status", "classes_asked", "classes_found", "classes_empty", "seconds_total", "peak_gpu_memory_mb"]
    with open(run_dir / "batch_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest["cases"])
    with open(run_dir / "failures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "stage", "reason"])
        for s in skipped:
            w.writerow([s["case_id"], "eligibility", s["reason"]])
        for r in manifest["cases"]:
            if r["status"] != "ok":
                w.writerow([r["case_id"], r["status"], "; ".join(r["issues"])])
    bad = sum(r["status"] != "ok" for r in manifest["cases"])
    print(f"done: {len(manifest['cases']) - bad} ok, {bad} failed -> {run_dir}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
