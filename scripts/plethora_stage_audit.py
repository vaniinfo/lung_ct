"""Exact stage-by-stage audit of the pleural_pericard_effusion run on PleThora scans (segmentation env). Diagnostic only.

Re-runs the network IN THIS PROCESS on the crop that TotalSegmentator used (crop box taken from the earlier trace), so
that every intermediate array can be counted without rounding:

  grid 0  source CT              as stored (0.977 x 0.977 mm in plane, 3 mm slices)
  grid 1  crop, RAS              the file TotalSegmentator hands to nnU-Net (same voxels, same spacing)
  grid 2  working grid           what the network sees: resampled by nnU-Net to the plans' spacing
  grid 3  crop grid again        logits resampled back, softmax, argmax
  grid 4  source grid            TotalSegmentator puts the crop back -> the file the pipeline reads

Checked: the mask rebuilt here equals the file written by the ordinary run (voxel for voxel). Recorded: grids,
spacing, every resampling call (order, order along the slice axis, whether the slice axis is handled separately),
class mapping, exact voxel counts and unrounded volumes per class and per probability level, on grids 2 and 3, for
the whole crop and inside EACH reviewer file separately. Probability levels are descriptive, not operating points.

Usage (Git Bash): environments/segmentation/Scripts/python.exe scripts/plethora_stage_audit.py --out runs/E16_.../stage_audit LUNG1-013 ...
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOTALSEG_HOME_DIR", str(ROOT / "models" / "totalsegmentator"))
P = ROOT / "data" / "raw" / "plethora"
TRACE = ROOT / "data" / "derived" / "plethora_trace"
SEG = ROOT / "data" / "derived" / "plethora"
MODEL = ROOT / "models" / "totalsegmentator" / "nnunet" / "results" / "Dataset315_thoraxCT" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
LEVELS = (0.5, 0.25, 0.1, 0.05, 0.01)


def main() -> int:
    import nibabel as nib
    import numpy as np
    import torch
    import nnunetv2.preprocessing.resampling.default_resampling as rs
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from totalsegmentator.cropping import crop_to_bbox_nifti
    from totalsegmentator.map_to_binary import class_map

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    calls = []
    real = rs.resample_data_or_seg

    def spy(data, new_shape, is_seg=False, axis=None, order=3, do_separate_z=False, order_z=0, **kw):
        calls.append({"in_shape": [int(v) for v in data.shape], "out_shape": [int(v) for v in new_shape], "is_seg": bool(is_seg), "order_in_plane_or_3d": int(order),
                      "slice_axis_handled_separately": bool(do_separate_z), "separate_axis": None if axis is None else [int(a) for a in np.atleast_1d(axis)],
                      "order_along_separate_axis": int(order_z) if do_separate_z else None})
        return real(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z, **kw)

    rs.resample_data_or_seg = spy
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True, device=torch.device("cuda"),
                           verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(str(MODEL), use_folds=None, checkpoint_name="checkpoint_final.pth")
    pm, cm, lm = pred.plans_manager, pred.configuration_manager, pred.label_manager
    labels = pred.dataset_json["labels"]
    idx = {v: k for k, v in labels.items()}
    common = {"network_class_index_to_name": {str(k): idx[k] for k in sorted(idx)}, "totalsegmentator_output_files": class_map["pleural_pericard_effusion"],
              "plans_spacing_zyx_mm": [float(v) for v in cm.spacing], "patch_size": [int(v) for v in cm.patch_size], "transpose_forward": list(pm.transpose_forward),
              "normalization": cm.normalization_schemes, "intensity_properties": pm.foreground_intensity_properties_per_channel,
              "resampling_settings_from_plans": {"data": cm.configuration.get("resampling_fn_data_kwargs"), "probabilities": cm.configuration.get("resampling_fn_probabilities_kwargs")},
              "folds_used": [str(f) for f in (pred.list_of_parameters and range(len(pred.list_of_parameters)))], "mirroring": False, "tile_step_size": 0.5}
    (out / "model_and_classes.json").write_text(json.dumps(common, indent=1, default=str), encoding="utf-8")

    for pid in args.cases:
        ct = nib.load(P / "volumes" / f"{pid}.nii.gz")
        bbox = json.loads((TRACE / pid / "trace.json").read_text(encoding="utf-8"))["crop"]["bbox_voxels"]
        zooms = [float(z) for z in ct.header.get_zooms()[:3]]
        vox_mm3 = float(np.prod(zooms))
        crop = nib.as_closest_canonical(crop_to_bbox_nifti(ct, bbox, dtype=np.int32))
        cc = nib.as_closest_canonical(ct)
        back = nib.orientations.ornt_transform(nib.orientations.io_orientation(cc.affine), nib.orientations.io_orientation(ct.affine))
        refs = {}
        for f in sorted((P / "effusions" / "Effusions" / pid).glob("*.nii.gz")):
            name = f.name[len(pid) + 1:-7].replace("effusion_", "")
            rc = nib.as_closest_canonical(nib.load(f))
            assert rc.shape == cc.shape and np.allclose(rc.affine, cc.affine, atol=1e-2), f.name
            r_on_ct = nib.orientations.apply_orientation(np.asarray(rc.dataobj) > 0, back)
            refs[name] = (int(r_on_ct.sum()), np.asarray(nib.as_closest_canonical(crop_to_bbox_nifti(nib.Nifti1Image(r_on_ct.astype(np.uint8), ct.affine), bbox)).dataobj) > 0)
        calls.clear()
        with tempfile.TemporaryDirectory(dir=str(TRACE)) as td:
            f_in = Path(td) / "s01_0000.nii.gz"
            nib.save(crop, f_in)
            data, props = NibabelIOWithReorient().read_images([str(f_in)])
        pp = cm.preprocessor_class(verbose=False)
        d2, _, props = pp.run_case_npy(data, None, props, pm, cm, pred.dataset_json)
        n_pre = len(calls)
        logits = pred.predict_logits_from_preprocessed_data(torch.from_numpy(d2)).cpu()
        p2 = torch.softmax(logits.float(), 0).numpy()
        a2 = p2.argmax(0)
        wv = float(np.prod(cm.spacing))
        seg3, p3 = convert_predicted_logits_to_segmentation_with_correct_shape(logits, pm, cm, lm, props, return_probabilities=True)
        seg3 = np.asarray(seg3).transpose(2, 1, 0)            # reader order (z, y, x) -> RAS crop (x, y, z)
        p3 = np.asarray(p3, dtype=np.float32).transpose(0, 3, 2, 1)
        assert seg3.shape == crop.shape
        tool = np.asarray(nib.load(SEG / pid / "pleural_pericard_effusion" / "pleural_effusion.nii.gz").dataobj) > 0
        tool_crop = np.asarray(nib.as_closest_canonical(crop_to_bbox_nifti(nib.Nifti1Image(tool.astype(np.uint8), ct.affine), bbox)).dataobj) > 0

        def counts(p, vol_mm3, mask=None):
            v = p if mask is None else p[mask]
            return {f"p>={lv}": {"voxels": int((v >= lv).sum()), "ml": float((v >= lv).sum()) * vol_mm3 / 1000.0} for lv in LEVELS}

        rec = {"patient_id": pid,
               "grid_0_source": {"shape": list(ct.shape), "spacing_mm": zooms, "axis_codes": "".join(nib.aff2axcodes(ct.affine)), "voxel_mm3": vox_mm3},
               "grid_1_crop": {"bbox_voxels_on_source_axes": bbox, "shape": list(crop.shape), "spacing_mm": [float(z) for z in crop.header.get_zooms()[:3]], "axis_codes": "RAS",
                               "as_read_by_nnunet": {"array_shape_zyx": list(data.shape[1:]), "spacing_zyx_mm": [float(s) for s in props["spacing"]]},
                               "nnunet_nonzero_crop": {"shape_before": list(props["shape_before_cropping"]), "bbox": props["bbox_used_for_cropping"],
                                                       "shape_after": list(props["shape_after_cropping_and_before_resampling"])}},
               "grid_2_working": {"shape_zyx": list(d2.shape[1:]), "spacing_zyx_mm": [float(v) for v in cm.spacing], "voxel_mm3": wv,
                                  "input_resampling_calls": calls[:n_pre], "input_value_range_after_normalisation": [float(d2.min()), float(d2.max())],
                                  "argmax_voxels_per_class": {idx[c]: int((a2 == c).sum()) for c in sorted(idx)},
                                  "argmax_pleural_ml": float((a2 == 2).sum()) * wv / 1000.0, "max_p_pleural": float(p2[2].max()),
                                  "p_pleural_levels": counts(p2[2], wv)},
               "grid_3_crop_after_resampling_back": {"output_resampling_calls": calls[n_pre:], "resampled_quantity": "logits (softmax and argmax are applied afterwards)",
                                                     "voxel_mm3": vox_mm3, "argmax_voxels_per_class": {idx[c]: int((seg3 == c).sum()) for c in sorted(idx)},
                                                     "argmax_pleural_ml": float((seg3 == 2).sum()) * vox_mm3 / 1000.0, "max_p_pleural": float(p3[2].max()),
                                                     "p_pleural_levels": counts(p3[2], vox_mm3)},
               "grid_4_pipeline_file": {"pleural_voxels": int(tool.sum()), "pleural_ml": float(tool.sum()) * vox_mm3 / 1000.0, "voxels_outside_crop_box": int(tool.sum() - tool_crop.sum()),
                                        "voxels_differing_from_mask_rebuilt_here": int((tool_crop != (seg3 == 2)).sum())},
               "inside_each_reviewer_file": {}}
        for name, (n_src, rm) in refs.items():
            n = int(rm.sum())
            rec["inside_each_reviewer_file"][name] = {
                "voxels_on_source_grid": n_src, "voxels_inside_crop": n, "ml": n * vox_mm3 / 1000.0,
                "argmax_voxels_per_class": {idx[c]: int((seg3[rm] == c).sum()) for c in sorted(idx)},
                "max_p_pleural": float(p3[2][rm].max()), "argmax_position_of_max_p_xyz_crop": [int(v) for v in np.argwhere(rm & (p3[2] == p3[2][rm].max()))[0]],
                "p_pleural_levels": counts(p3[2], vox_mm3, rm)}
        (out / f"{pid}.json").write_text(json.dumps(rec, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
        g3, g4 = rec["grid_3_crop_after_resampling_back"], rec["grid_4_pipeline_file"]
        print(pid, "working grid", rec["grid_2_working"]["shape_zyx"], "| max p", round(g3["max_p_pleural"], 4), "| vox p>=0.1:", g3["p_pleural_levels"]["p>=0.1"]["voxels"],
              "p>=0.5:", g3["p_pleural_levels"]["p>=0.5"]["voxels"], "| argmax pleural vox", g3["argmax_voxels_per_class"]["pleural effusion"],
              "| file vox", g4["pleural_voxels"], "| differ", g4["voxels_differing_from_mask_rebuilt_here"], "| in-resample", calls[:n_pre], flush=True)
        del logits, p2, p3
    return 0


if __name__ == "__main__":
    sys.exit(main())
