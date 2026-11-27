"""Trace PleThora scans through every stage of the TotalSegmentator pleural_pericard_effusion run (segmentation env).

Diagnostic only. Same task, weights and settings as plethora_segment.py / the lung_ct agent. Nothing is tuned and no
limit is derived here: the probability levels below (0.5 / 0.25 / 0.1 / 0.01) only describe how far the model was
from outlining fluid; they are not proposed operating points.

Per scan, written to data/derived/plethora_trace/<PatientID>/:
  trace.json        crop box, class probabilities inside the reference, postprocessing check, mirroring variant
  ct_crop.nii.gz, ref_crop.nii.gz, p_pleural_pct.nii.gz, p_lung_pct.nii.gz   (crop grid, RAS) for overlays

Stages checked:
  crop       lobes from the 6 mm 'total' model -> box + 50 mm: is the whole reference inside the box?
  inference  class probabilities exported by nnU-Net on the crop grid (as run: no mirroring)
  postproc   argmax of those probabilities, put back by TotalSegmentator == the file the pipeline reads?
  variant    the same run with test-time mirroring switched on (TotalSegmentator runs every task without it)

Usage (Git Bash): environments/segmentation/Scripts/python.exe scripts/plethora_empty_mask_trace.py LUNG1-013 ...
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOTALSEG_HOME_DIR", str(ROOT / "models" / "totalsegmentator"))
TASK = "pleural_pericard_effusion"
P = ROOT / "data" / "raw" / "plethora"
OUT = ROOT / "data" / "derived" / "plethora_trace"
LEVELS = (0.5, 0.25, 0.1, 0.01)
CLASSES = ("background", "lung", "pleural_effusion", "pericardial_effusion")


def main() -> int:
    import nibabel as nib
    import numpy as np
    import totalsegmentator.nnunet as tsn
    from totalsegmentator.cropping import crop_to_bbox_nifti
    from totalsegmentator.python_api import totalsegmentator

    cfg = Path(os.environ["TOTALSEG_HOME_DIR"]) / "config.json"
    if not cfg.exists() or json.loads(cfg.read_text()).get("send_usage_stats") is not False:
        raise SystemExit(f"{cfg}: usage statistics must be switched off before running")

    cap = {}
    real_crop, real_predict = tsn.crop_to_mask, tsn.nnUNetv2_predict

    def spy_crop(img_in, mask_img, addon=(0, 0, 0), dtype=None, verbose=False):
        out, bbox = real_crop(img_in, mask_img, addon=addon, dtype=dtype, verbose=verbose)
        if "bbox" not in cap:  # first call = the CT itself
            cap.update(bbox=bbox, lobes=np.asarray(mask_img.get_fdata() > 0), crop_shape=out.shape)
        return out, bbox

    def spy_predict(*a, **k):
        a = list(a)
        if cap.get("mirroring") and int(a[2]) == 315:
            a[6] = True  # positional 'tta'
        return real_predict(*a, **k)

    tsn.crop_to_mask, tsn.nnUNetv2_predict = spy_crop, spy_predict

    for pid in sys.argv[1:]:
        t0 = time.time()
        out = OUT / pid
        out.mkdir(parents=True, exist_ok=True)
        ct_path = P / "volumes" / f"{pid}.nii.gz"
        ct = nib.load(ct_path)
        zooms = np.array(ct.header.get_zooms()[:3], float)
        vml = float(np.prod(zooms)) / 1000.0
        # reference (first-reviewer file), brought onto the CT's own voxel grid by axis reorientation only
        ref_img = nib.load(P / "effusions" / "Effusions" / pid / f"{pid}_effusion_first_reviewer.nii.gz")
        rc, cc = nib.as_closest_canonical(ref_img), nib.as_closest_canonical(ct)
        assert rc.shape == cc.shape and np.allclose(rc.affine, cc.affine, atol=1e-2), "reference and CT grids differ"
        back = nib.orientations.ornt_transform(nib.orientations.io_orientation(cc.affine), nib.orientations.io_orientation(ct.affine))
        ref = nib.orientations.apply_orientation(np.asarray(rc.dataobj) > 0, back)
        assert ref.shape == ct.shape
        ref_on_ct = nib.Nifti1Image(ref.astype(np.uint8), ct.affine)
        rec = {"patient_id": pid, "ct_shape": list(ct.shape), "spacing_mm": [round(float(z), 3) for z in zooms], "reference_ml": round(float(ref.sum()) * vml, 1)}

        res = {}
        for variant in ("as_run", "mirroring_on"):
            cap.clear()
            cap["mirroring"] = variant == "mirroring_on"
            with tempfile.TemporaryDirectory(dir=str(OUT)) as td:
                seg_dir, npz = Path(td) / "seg", Path(td) / "prob.npz"
                totalsegmentator(ct_path, seg_dir, quiet=True, task=TASK, fast=False, device="gpu", save_probabilities=npz)
                prob = np.load(npz)["probabilities"].astype(np.float32)          # (class, z, y, x) on the cropped RAS grid
                seg_file = np.asarray(nib.load(seg_dir / "pleural_effusion.nii.gz").dataobj) > 0
            bbox = cap["bbox"]
            ct_crop = nib.as_closest_canonical(crop_to_bbox_nifti(ct, bbox))
            ref_crop = np.asarray(nib.as_closest_canonical(crop_to_bbox_nifti(ref_on_ct, bbox)).dataobj) > 0
            prob = prob.transpose(0, 3, 2, 1)
            assert prob.shape[1:] == ct_crop.shape, (prob.shape, ct_crop.shape)
            arg = prob.argmax(0)
            pp = prob[2]
            # postprocessing check: argmax on the crop grid, put back the same way, against the file the pipeline reads
            file_crop = np.asarray(nib.as_closest_canonical(crop_to_bbox_nifti(nib.Nifti1Image(seg_file.astype(np.uint8), ct.affine), bbox)).dataobj) > 0
            n_ref = int(ref_crop.sum())
            r = {"pleural_ml_from_argmax_on_crop": round(float((arg == 2).sum()) * vml, 2), "pleural_ml_in_output_file": round(float(seg_file.sum()) * vml, 2),
                 "argmax_equals_output_file": bool(np.array_equal(arg == 2, file_crop)), "output_file_voxels_outside_crop": int(seg_file.sum() - file_crop.sum()),
                 "max_p_pleural_anywhere": round(float(pp.max()), 4),
                 "ml_with_p_pleural_at_least": {str(lv): round(float((pp >= lv).sum()) * vml, 1) for lv in LEVELS},
                 "inside_reference": {
                     "argmax_class_fraction": {CLASSES[c]: round(float((arg[ref_crop] == c).mean()), 3) for c in range(4)},
                     "p_pleural_max": round(float(pp[ref_crop].max()), 4), "p_pleural_p99": round(float(np.percentile(pp[ref_crop], 99)), 4),
                     "p_pleural_median": round(float(np.median(pp[ref_crop])), 5),
                     "fraction_with_p_pleural_at_least": {str(lv): round(float((pp[ref_crop] >= lv).mean()), 4) for lv in LEVELS},
                     "mean_p": {CLASSES[c]: round(float(prob[c][ref_crop].mean()), 4) for c in range(4)}}}
            res[variant] = r
            if variant == "as_run":
                lobes = cap["lobes"]
                box = np.zeros(ct.shape, bool)
                box[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = True
                idx = np.argwhere(ref)
                margins = [[int(idx[:, a].min() - bbox[a][0]), int(bbox[a][1] - 1 - idx[:, a].max())] for a in range(3)]
                rec["crop"] = {"bbox_voxels": bbox, "crop_shape": list(cap["crop_shape"]), "lobes_from_6mm_model_ml": round(float(lobes.sum()) * vml, 0),
                               "reference_fraction_inside_box": round(float((ref & box).sum()) / max(1, int(ref.sum())), 4),
                               "reference_voxels_cut_by_box": int((ref & ~box).sum()),
                               "smallest_margin_reference_to_box_face_mm": round(float(min(min(m) * zooms[a] for a, m in enumerate(margins))), 1),
                               "box_reaches_image_edge": [bool(bbox[a][0] == 0 or bbox[a][1] == ct.shape[a]) for a in range(3)],
                               "reference_fraction_inside_6mm_lobes": round(float((ref & lobes).sum()) / max(1, int(ref.sum())), 3)}
                rec["network_input"] = {"crop_voxels_before_resampling": list(ct_crop.shape), "model_spacing_mm": 1.5,
                                        "crop_size_at_model_spacing": [int(round(s * z / 1.5)) for s, z in zip(ct_crop.shape, ct_crop.header.get_zooms()[:3])],
                                        "patch_size": [128, 128, 128], "intensity_clip_hu": [-962, 110], "normalisation": "(HU - (-721.3)) / 262.6 after clipping"}
                nib.save(nib.Nifti1Image(np.asarray(ct_crop.dataobj).astype(np.int16), ct_crop.affine), out / "ct_crop.nii.gz")
                nib.save(nib.Nifti1Image(ref_crop.astype(np.uint8), ct_crop.affine), out / "ref_crop.nii.gz")
                nib.save(nib.Nifti1Image(np.round(pp * 100).astype(np.uint8), ct_crop.affine), out / "p_pleural_pct.nii.gz")
                nib.save(nib.Nifti1Image(np.round(prob[1] * 100).astype(np.uint8), ct_crop.affine), out / "p_lung_pct.nii.gz")
            else:
                nib.save(nib.Nifti1Image(np.round(pp * 100).astype(np.uint8), ct_crop.affine), out / "p_pleural_pct_mirroring.nii.gz")
            del prob, pp, arg
        rec["inference"] = res
        rec["seconds"] = round(time.time() - t0, 1)
        (out / "trace.json").write_text(json.dumps(rec, indent=1, default=lambda o: o.item()), encoding="utf-8")
        a = res["as_run"]
        print(f"{pid} ref {rec['reference_ml']} mL | in box {rec['crop']['reference_fraction_inside_box']} | as run {a['pleural_ml_in_output_file']} mL, "
              f"max p in ref {a['inside_reference']['p_pleural_max']}, argmax in ref {a['inside_reference']['argmax_class_fraction']} | "
              f"mirroring {res['mirroring_on']['pleural_ml_in_output_file']} mL | file==argmax {a['argmax_equals_output_file']} | {rec['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
