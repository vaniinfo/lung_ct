# Mask-derived airway estimates for bronchiectasis research (exploratory)

Date: 2026-11-23. Status: **synthetic-tube tests pass and four scans have been run as a pipeline check; not validated
against any airway reference.** Research use only. First-run notes: docs/STATUS.md.

This stage produces **mask-derived estimates**: numbers computed from the shape of TotalSegmentator's predicted airway
mask. They are not airway-diameter measurements made on the CT image, and they are not validated. The stage does not
detect or diagnose bronchiectasis.

**What the current data can and cannot show.** 85 local scans have an airway mask; 14 carry a report-derived
bronchiectasis label, and a radiologist has reviewed 4 scans (2 graded present). This is enough to check that the
pipeline behaves as intended. It cannot establish bronchiectasis detection performance.

## 1. What this is and is not

| | |
|---|---|
| Is | an estimation stage: lumen area, equivalent diameter and tapering per airway branch, estimated from the predicted mask, in physical units, with quality flags |
| Is not | AirQuant. AirQuant is a MATLAB tool; this project has no MATLAB and does not run it. It is a methods reference only. Our results are "lung_ct airway calibre v1" and must not be called AirQuant results |
| Is not | a bronchiectasis detector, a diagnostic model or a set of clinical thresholds |

## 2. Stages (kept separate)

| Stage | Code | Output |
|---|---|---|
| 1. Airway segmentation | `scripts/segment_totalseg.py` (TotalSegmentator `lung_vessels`) | `lung_airways.nii.gz` on the CT grid (a model prediction) |
| 2. Airway tree (baseline, unchanged) | `src/lung_ct/airway/tree.py`, `scripts/airway_tree.py` | `runs/P5_airway_*`: branches, generation, side, length, tortuosity |
| 3. Airway calibre (new) | `src/lung_ct/airway/calibre.py`, `scripts/airway_measure.py` | `runs/P6_airway_measures_*` |
| 4. Group comparison (new) | `src/lung_ct/airway/cohort.py`, `scripts/airway_group_compare.py` | `runs/E20_airway_group_compare_*` |
| 5. Classification | not implemented | would be its own stage with its own metrics |

How data flow: CT volume (NIfTI, HU) -> stage 1 writes the airway mask on the CT grid -> stage 2 reorients the mask to
RAS, keeps the largest connected piece, resamples it to isotropic voxels (nearest neighbour), thins it to a skeleton,
builds the branch graph (Skan), prunes and contracts it -> stage 3 reuses that grid and those branches.
The CT intensities are not used by stages 2 and 3; only the mask and its affine are.

## 3. Required inputs

| Input | Where | Note |
|---|---|---|
| Segmentation run manifest | `runs/P2_seg_*/run_manifest.json` | each case needs `status`, `output_dir` |
| Airway mask | `<output_dir>/lung_vessels/lung_airways.nii.gz` | spacing is read from the NIfTI affine |
| Lobe masks (optional) | `<output_dir>/total/lung_*_lobe_*.nii.gz` | missing: lobe is not assigned and the case is flagged |
| Labels (stage 4) | CSV `case_id,label[,patient_id][,label_source]` | 1 = bronchiectasis, 0 = control |

## 4. Outputs and units

`airway_branch_measures.csv`, one row per tree branch:

| Column | Unit | Meaning |
|---|---|---|
| `branch_id`, `generation`, `kind`, `side`, `parent_branch_id` | none | same ids and generations as the tree stage |
| `length_mm`, `usable_length_mm`, `trim_proximal_mm`, `trim_distal_mm` | mm | centreline length; length left after trimming both ends |
| `lumen_volume_mm3` | mm³ | mask volume assigned to the branch (untrimmed) |
| `mean_area_mm2` | mm² | assigned volume in the trimmed part / usable length |
| `area_equivalent_diameter_mm` | mm | diameter of a circle with that area |
| `inscribed_diameter_median_mm` | mm | QC cross-check only (largest sphere that fits). **Not an airway diameter; do not report it as one** |
| `calibre_ratio` | none | equivalent / inscribed diameter; about 1 for a round tube |
| `intratapering_pct_per_mm` | % per mm | narrowing along the branch; positive = narrows distally |
| `intertapering_pct` | % | narrowing against the parent branch |
| `lobe`, `lobe_fraction` | none | majority lobe under the centreline; `central` for trachea and main bronchi |
| `flags` | none | see section 6 |

`airway_case_features.csv`: medians per generation (1 to 8), per lobe and over all measured branches.

**Voxel-unit outputs to be aware of.** The new files have none. The baseline tree stage has four counts in voxels,
which are not sizes and must not be compared between scans with different spacing: `n_skeleton_voxels`
(`airway_branches.csv`), `input_voxels`, `kept_component_voxels`, `discarded_voxels` (`airway_summary.json`).
Its lengths, endpoint distances and coordinates are in mm.

## 5. Definitions and assumptions

- **Spacing** comes from the NIfTI affine. Analysis runs on an isotropic grid whose voxel edge is the smallest input
  spacing. Labels are resampled with nearest neighbour only.
- **Ownership.** Each lumen voxel is assigned to the nearest centreline voxel. For a straight tube, volume owned
  divided by centreline length equals the cross-sectional area. This assumes a roughly tubular lumen.
- **Trimming.** Each branch end is trimmed by at least 2 mm or one inscribed radius, because voxels near a bifurcation
  belong to more than one airway.
- **Tapering** is fitted on the logarithm of the equivalent diameter in 3 mm windows; at least three windows are needed.
- **Parameters** are in `calibre.DEFAULTS`. They are reporting conventions chosen before any result was seen. They are
  not validated and not clinical thresholds.
- Mask voxels of branches that are not in the tree (cycle-closing branches) are assigned to the nearest tree branch.

## 6. Quality flags

| Flag | Level | Raised when |
|---|---|---|
| `missing_or_invalid_spacing` | case | spacing is zero, negative or not a number |
| `spacing_may_be_a_default` | case | 1 mm identity affine: the header may not carry the true spacing |
| `anisotropic_voxels`, `thick_slices` | case | coarsest / finest spacing above 2; coarsest spacing above 1.5 mm |
| `broken_airway_mask` | case | mask volume outside the largest piece above 5% of the kept volume |
| `airway_mask_touches_image_edge` | case | the tree may be cut off there (common at the top of the trachea) |
| `truncated_scan` | case | the lung outline touches the image edge |
| `incomplete_or_ambiguous_centreline` | case | tree flags: side not assigned, root below top of mask, cycles left, fewer than 3 branches |
| `terminal_branch` | branch | the branch ends where the segmentation stopped, not at an anatomical end |
| `short_usable_length` | branch | under 3 mm left after trimming |
| `small_relative_to_voxel` | branch | diameter under 3 x the coarsest input spacing: partial-volume and stair-step artefact likely |
| `non_circular_or_merged_lumen` | branch | calibre ratio above 1.5: flattened lumen, touching airways or a leak |
| `no_parent_diameter`, `not_measured` | branch | no value could be computed |

Branches flagged `not_measured`, `short_usable_length` or `non_circular_or_merged_lumen` are left out of case features.

## 7. Broncho-arterial ratio: out of scope

TotalSegmentator gives artery and vein masks, but this project has no reliable way to pair each airway with its
adjacent artery (same generation, same orientation, artery not merged with a vein). Without that pairing the ratio is
not meaningful. It is not implemented. It would need its own pairing method and its own validation.

## 8. Comparing labelled cases and controls

`scripts/airway_group_compare.py` describes how measurements differ between two labelled groups.

- Splits are made per **patient** and written once; the file is never changed afterwards.
- Scans of one patient are combined by median, so a patient counts once.
- Default: train split only. Validation is added with `--use train validation`.
- The **test split is locked**. It is read only with `--frozen-plan <file>`: a written plan (features, direction and
  any threshold) fixed on train and validation. The plan's hash is recorded.
- Output per feature: group sizes, medians, median difference, rank-biserial value and an uncorrected p value.
- This is **not classification performance**. No threshold is chosen and no sensitivity or specificity is computed.
  If a classifier is added it must report its metrics separately and be labelled exploratory until experts have
  validated the labels and the measurements.
- **Measurement agreement** (does our diameter match a reference diameter?) is a different question from group
  difference and from classification. It needs reference measurements (section 10) and is not addressed by this script.

## 9. Blockers and limits

1. **AirQuant runtime.** MATLAB only; not available here. Not a dependency.
2. **No validation against real airways.** Synthetic tubes are recovered (6 mm tube: 5.87 to 6.08 mm), but no dataset with airway
   reference annotations is used, so agreement with real airway diameters is unknown.
3. **Labels.** The local bronchiectasis labels are CT-RATE labels derived from report text: 14 of 85 scans. They are not
   expert image annotations, and the group is too small for a meaningful three-way split. A result on them is a pipeline
   check only. The clinician reviewer graded 4 scans (2 present).
4. **Segmentation dependence.** Every value depends on the predicted mask, slice thickness, reconstruction and
   inspiration. Fewer or shorter detected branches do not indicate airway loss.
5. **Small airways.** Bronchiectasis often shows in airways only a few voxels wide, where this method is least reliable.
6. **Generation** is graph depth in our tree, not an anatomical generation label.

## 10. Validation plan (not started)

| Step | What | Passes when |
|---|---|---|
| 1 | Synthetic tubes: `tests/test_airway_calibre.py` | known diameters and tapering are recovered at two voxel sizes |
| 2 | Baseline unchanged | `n_tree_branches`, `max_generation`, `total_centreline_mm` equal in P5 and P6 summaries |
| 3 | Visual check on real CT | branch diameters overlaid on CT for a few cases; reviewed by a person |
| 4 | Parameter sensitivity | trims and window varied; unstable features are marked and not compared between patients |
| 5 | Measurement agreement | against a dataset with airway reference annotations (for example ATM'22 or AeroPath) or a phantom |
| 6 | Expert-labelled cohort | bronchiectasis labels from expert image reading; patient-level split frozen before analysis |
| 7 | Test split | read once, with a frozen plan |

## 11. Commands (1 and 2 run on 2026-11-23; 3 run on four CT-RATE scans only; 4 not run)

```bash
# 1. synthetic-tube and split tests for the new code only
python -m pytest tests/test_airway_calibre.py -q
# 2. baseline tests, to confirm nothing else changed
python -m pytest tests/test_airway_tree.py -q
# 3. measurements on the 10 baseline scans (new folder; the P5 baseline is not touched)
python scripts/airway_measure.py --seg-run runs/P2_seg_10cases_<ts>
#    then compare n_tree_branches / max_generation / total_centreline_mm with runs/P5_airway_10cases_20261108_092858/summary.csv
# 4. pipeline check of the comparison on CT-RATE report labels (not expert labels)
python scripts/airway_measure.py --seg-run runs/P2_seg_20cases_<ts>
python scripts/airway_group_compare.py --labels-from-ctrate runs/airway_labels_ctrate.csv
python scripts/airway_group_compare.py --measure-run runs/P6_airway_measures_<ts> --labels runs/airway_labels_ctrate.csv \
    --splits runs/airway_splits_seed20261123.json --seed 20261123
```
