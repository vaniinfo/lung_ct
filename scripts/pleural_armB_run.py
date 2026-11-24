"""Arm B of the pleural-effusion comparison: run the published nnU-Net v1 model Task307_pleuralfinal (GPU).

Runs in the separate environment `environments/pleural_v1` (nnU-Net 1.7.1). That environment separates package
versions only; it is not a security boundary, and the checkpoints are pickle files that run code when loaded.

Driver: nnU-Net v1's own entry point (predict_from_folder) does not run on Windows: its preprocessing worker
processes cannot be started (PicklingError, recorded in runs/test_logs/armB_go_nogo.log). This script therefore
calls the SAME nnU-Net functions in the SAME order as nnunet.inference.predict.predict_cases, in one process:
load_model_and_checkpoint_files -> trainer.preprocess_patient -> predict_preprocessed_data_return_seg_and_softmax
for each fold -> mean of the five softmax maps -> save_segmentation_nifti_from_softmax. No nnU-Net code is changed
and no setting is chosen by us. Not verified: bit-identity with the multi-process entry point, which cannot run here.

Published inference configuration, nothing chosen by us:
  - nnU-Net v1, 3d_fullres, trainer nnUNetTrainerV2, plans nnUNetPlansv2.1 (working grid 2.0 mm, patch 128^3)
  - all five folds as an ensemble (paper: "An ensemble from the 5-fold cross-validation models was used")
  - checkpoint model_final_checkpoint, test-time mirroring on, step size 0.5 (nnU-Net v1 defaults; the archive's own
    validation_args.json shows do_mirroring true, step_size 0.5, use_gaussian true)
  - whole CT as stored; nnU-Net resamples it. No crop, no filter, no limit, no postprocessing of ours.
Known difference from the published pipeline: the authors downsampled scans with their own (unpublished) tool before
prediction; here nnU-Net's own resampling does it.

Label meaning is checked, not assumed: label 1 must be lung-sized (1 to 9 litres) or the scan is failed (--lung-guard stop).
With --lung-guard warn the mask is saved and scored as produced and the scan is flagged (used for LUNG1-104, so that a
bad output stays in the comparison and is not hidden by the guard).
Per case: runtime, peak GPU memory, label volumes, failure text. Failed cases are recorded, never dropped.

Usage (Git Bash): environments/pleural_v1/Scripts/python.exe scripts/pleural_armB_run.py --manifest runs/E17_.../pilot_manifest_armB.json --out runs/E17_.../pilot
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "pleural_task307"
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["RESULTS_FOLDER"] = str(MODELS)                       # nnU-Net v1 reads these at import
os.environ.setdefault("nnUNet_raw_data_base", str(MODELS / "_unused_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(MODELS / "_unused_preprocessed"))
MODEL_DIR = MODELS / "nnUNet" / "3d_fullres" / "Task307_pleuralfinal" / "nnUNetTrainerV2__nnUNetPlansv2.1"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lung-guard", choices=("stop", "warn"), default="stop",
                    help="stop: a scan whose label 1 is not lung-sized is failed and not saved; warn: it is saved, scored and flagged")
    args = ap.parse_args()
    import importlib.metadata as md
    import nibabel as nib
    import numpy as np
    import torch
    from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
    from nnunet.training.model_restore import load_model_and_checkpoint_files

    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    weights = sorted(MODEL_DIR.glob("fold_*/model_final_checkpoint.model"))
    if len(weights) != 5:
        raise SystemExit(f"expected five folds under {MODEL_DIR}, found {len(weights)}")
    log = {"arm": man["arm"], "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
           "packages": {p: md.version(p) for p in ("nnunet", "torch", "numpy", "nibabel", "batchgenerators")},
           "settings": {"folds": "all five (ensemble)", "checkpoint": "model_final_checkpoint", "mirroring": True, "step_size": 0.5, "mode": "normal", "mixed_precision": True,
                        "all_in_gpu": False, "use_gaussian": True, "postprocessing": "none (the archive's postprocessing.json lists no class)",
                        "driver": "single process, same nnU-Net functions and order as predict_cases (its worker processes do not start on Windows)"},
           "weights_sha256": {str(w.relative_to(MODEL_DIR)): sha256(w) for w in weights}, "plans_sha256": sha256(MODEL_DIR / "plans.pkl"), "cases": {}}
    torch.cuda.empty_cache()
    trainer, params = load_model_and_checkpoint_files(str(MODEL_DIR), None, mixed_precision=True, checkpoint_name="model_final_checkpoint")
    assert len(params) == 5
    sep = trainer.plans.get("segmentation_export_params") or {"force_separate_z": None, "interpolation_order": 1, "interpolation_order_z": 0}
    log["plans"] = {"segmentation_export_params": sep, "transpose_forward": trainer.plans.get("transpose_forward"), "mirror_axes": list(trainer.data_aug_params["mirror_axes"]),
                    "target_spacing": [float(v) for v in trainer.plans["plans_per_stage"][trainer.stage]["current_spacing"]], "num_classes": int(trainer.num_classes)}
    for c in man["cases"]:
        cid, rec = c["case_id"], {"status": "ok", "issue": ""}
        dst = ROOT / c["prediction"]
        try:
            with tempfile.TemporaryDirectory(dir=str(out)) as td:
                tin, tout = Path(td) / "in", Path(td) / "out"
                tin.mkdir()
                shutil.copy(ROOT / c["ct"], tin / f"{cid}_0000.nii.gz")
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                tout.mkdir()
                d, _, dct = trainer.preprocess_patient([str(tin / f"{cid}_0000.nii.gz")])
                rec["network_input_shape"] = [int(v) for v in d.shape[1:]]
                rec["source_spacing_zyx"] = [float(v) for v in dct["original_spacing"]]
                softmax = None
                for prm in params:
                    trainer.load_checkpoint_ram(prm, False)
                    sm = trainer.predict_preprocessed_data_return_seg_and_softmax(
                        d, do_mirroring=True, mirror_axes=trainer.data_aug_params["mirror_axes"], use_sliding_window=True, step_size=0.5,
                        use_gaussian=True, all_in_gpu=False, mixed_precision=True)[1]
                    softmax = sm if softmax is None else softmax + sm
                softmax /= len(params)
                tb = trainer.plans.get("transpose_backward")
                if trainer.plans.get("transpose_forward") is not None:
                    softmax = softmax.transpose([0] + [k + 1 for k in tb])
                save_segmentation_nifti_from_softmax(softmax, str(tout / f"{cid}.nii.gz"), dct, sep["interpolation_order"], getattr(trainer, "regions_class_order", None),
                                                     None, None, None, None, sep["force_separate_z"], sep["interpolation_order_z"])
                del softmax, d
                rec["seconds"] = round(time.time() - t0, 1)
                rec["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
                seg_img = nib.load(tout / f"{cid}.nii.gz")
                seg = np.asarray(seg_img.dataobj)
                vml = float(np.prod(seg_img.header.get_zooms()[:3])) / 1000.0
                rec["label_ml"] = {str(int(v)): float((seg == v).sum()) * vml for v in np.unique(seg) if v}
                lung_l = rec["label_ml"].get("1", 0.0) / 1000.0
                if not 1.0 <= lung_l <= 9.0 and args.lung_guard == "warn":
                    rec["issue"] = f"WARNING label 1 is {lung_l:.2f} L, not lung-sized; mask saved and scored as produced"
                elif not 1.0 <= lung_l <= 9.0:
                    raise RuntimeError(f"label 1 is {lung_l:.2f} L, not lung-sized: label meaning is not as assumed (1 lung, 2 pleural effusion)")
                dst.parent.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image((seg == 2).astype(np.uint8), seg_img.affine), dst)
                nib.save(nib.Nifti1Image((seg == 1).astype(np.uint8), seg_img.affine), dst.parent / "lung.nii.gz")
                rec["prediction_sha256"] = sha256(dst)
        except Exception as e:  # noqa: BLE001  recorded, never dropped
            rec.update(status="failed", issue=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
        log["cases"][cid] = rec
        (out / f"run_log_{man['arm']}.json").write_text(json.dumps(log, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
        print(cid, rec["status"], rec.get("seconds"), "s", rec.get("peak_gpu_memory_mb"), "MB", rec.get("label_ml"), rec["issue"], flush=True)
    failed = [k for k, v in log["cases"].items() if v["status"] != "ok"]
    print(f"done: {len(log['cases']) - len(failed)} ok, {len(failed)} failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
