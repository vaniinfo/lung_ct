# Plan: compare the current pleural-effusion model with one alternative (version 3, 2026-11-23)

Status: **storage exception approved by the project lead (about 13 GB peak, 7.6 GB retained); pilot run and
technically accepted; fixed development comparison run.** Results: `runs/E17_pleural_model_comparison_prep_20261123/REPORT.md`.
Sections 1 to 8 are the plan as fixed before Arm B results; section 9 records what happened. No training, no threshold change, no density or shape rejection rule, in any step.
Research use only. Background: E14 and E16 reports under `runs/`. Preparation files:
`runs/E17_pleural_model_comparison_prep_20261123/`.

## 1. Question

On development cases, does one other published model outline the pleural fluid that the current model misses or
mislabels, without new failures of its own? Descriptive only. It cannot show that either model is accurate.

## 2. The two arms (frozen)

| | Arm A (current) | Arm B (alternative) |
|---|---|---|
| Model | TotalSegmentator 2.18.0, task `pleural_pericard_effusion` (315), fold 0 | `Task307_pleuralfinal`, nnU-Net v1, `nnUNetTrainerV2__nnUNetPlansv2.1`, 3d_fullres |
| Source | installed | Zenodo 21918112 (Shan Yang, 2026-08-13), CC BY 4.0; paper Sexauer et al., Invest Radiol 2022;57:552, PMC9390225 |
| Working grid | 1.5 mm, patch 128 cubed | **2.0 mm**, patch 128 cubed (read from `plans.pkl`) |
| Classes | lung, pleural effusion, pericardial effusion | two classes; 1 = lung and 2 = pleural effusion is probable (cross-validation Dice 0.98 and 0.58 to 0.71), **confirmed on the first pilot scan or the run stops** |
| Inference | as in E14: one fold, no mirroring, lobe crop + 50 mm | as published: five-fold ensemble, final checkpoint, mirroring on, step 0.5, whole scan |
| Training data | 600 published cases, 216 with a pleural label | 224 scans (160 with effusion, plus scans without), University Hospital Basel, Siemens, 120 kVp, 1.0 mm soft kernel, 2016 to 2021 |

Honest expectation, stated before any run: Arm B's grid is coarser than Arm A's, so it is not obviously better
placed for thin layers. It is still the one justified alternative: a published, effusion-only model with a
five-fold ensemble, and the only other public pleural-effusion model with weights that was found.

The checklist rule is the same for both arms (`checklist_v3_8`, limits unchanged and provisional).

## 3. What was inspected (read-only)

| Item | Finding | Source |
|---|---|---|
| Paper | training and test from one hospital, 3 Siemens scanners; input "a soft tissue kernel (30f) of 1.0 mm"; "An ensemble from the 5-fold cross-validation models was used for inference"; Python 3.7 | PMC9390225 full text |
| PleThora in the paper | "External validation of the segmentation performance was performed using data from the publicly available PleThora project (n = 34)"; median Dice 0.71, mean volume 383 mL | same |
| Archive | 750 entries, 2,360,306,044 bytes zipped, 2,539,672,668 unpacked; five folds, each with `model_final_checkpoint.model` (250 MB) and `model_best.model`; 672 small validation masks; training logs dated 2021-02-26 to 28 | `armB_archive_listing.json` (file list read by byte range) |
| Plans | 224 training cases; target spacing 2.0 x 2.0 x 2.0 mm; CT normalisation; no postprocessing rule (`for_which_classes: []`) | `armB_plans_summary.json` (restricted unpickler; no class in the file was refused) |
| Repository | post-processing and classification scripts only; no inference script, no licence file; scripts read a "downsampling/raw.nii", so the authors downsampled scans with their own tool before prediction | `github.com/usb-radiology/pleuraleffusion` |

### Prior evaluation exposure is not training overlap

| | Arm A | Arm B |
|---|---|---|
| PleThora or NSCLC-Radiomics scans in **training** | not indicated by any source; not excluded (files are resampled, cannot be matched by grid) | not indicated: the paper describes PleThora as external, and the plans list 224 local case numbers |
| PleThora scans used for **evaluation by the authors** | unknown | **yes, 34 scans**; which ones is not stated in the main text. Their mean volume (383 mL) is far above our six misses (9 to 150 mL) |

- Consequence: on PleThora, Arm B's authors have seen results on up to 34 of our 77 effusion scans. That is exposure,
  not contamination, but it is one more reason the result is descriptive.
- Not identified: the 34 scans. The supplement was not retrieved. If the list is obtained, results are also shown
  with those 34 separated.

## 4. Requirements

### 4a. Compatibility

| Item | State |
|---|---|
| Framework | nnU-Net v1 (PyPI `nnunet` 1.7.1, Apache 2.0, source package only, 277 KB). Not loadable by our nnunetv2 2.8.1 |
| Environment | new isolated `environments/pleural_v1` on D:, Python 3.11.8. Never mixed into `segmentation` |
| PyTorch | v1 loads checkpoints with plain `torch.load`; PyTorch 2.6 changed that default, so the plan pins PyTorch 2.5.1 (CUDA 12.4) and NumPy below 2. **To verify at install** |
| Windows | **main risk**: nnU-Net v1 was developed for Linux. Whether its inference worker processes run on Windows is unverified. Go / no-go is step 2 of the pilot. No workaround is assumed |
| Checkpoints | pickle files: loading runs code. Loaded only in the isolated environment, after the archive MD5 matches, with no network use during the run |
| Input | whole CT as stored (HU, any spacing); nnU-Net resamples to 2.0 mm. Difference from the published pipeline: the authors' own downsampling step is not published |
| Our data | PleThora scans are 3 mm, kernels B19f to B41f; CT-RATE scans differ again. The paper's 1.0 mm B30f input is not matched. Recorded as a limitation, not adjusted for |

### 4b. Provenance

| Item | State |
|---|---|
| Integrity | Zenodo states MD5 `52f3dab3815733018367a698c77400b0`. Verified after download; SHA-256 of the five checkpoints and of `plans.pkl` recorded in the run log |
| Licence | CC BY 4.0 on the Zenodo record; cite Sexauer et al. 2022 |
| Relation to Arm A's training set | unknown (same group; 224 against 600 cases). Recorded as a limitation |

### 4c. Storage: exception requested

Project rule: "Max 20GB new data, 30GB model weights for initial scope."

| Item | Additional disk | Kept afterwards | Budget line |
|---|---|---|---|
| Weights archive (zip) | 2.36 GB | 0 (deleted after MD5 check and extraction) | model weights |
| Extracted files needed (5 final checkpoints, their `.pkl`, `plans.pkl`) | 1.25 GB | 1.25 GB | model weights |
| Environment `pleural_v1` (PyTorch with CUDA; the `segmentation` env is 5.5 GB) | about 6 GB | about 6 GB | none (environments are not budgeted) |
| Package download cache, on D:, cleared afterwards | up to about 3 GB | 0 | none |
| Predicted masks, 213 scans | about 0.3 GB | about 0.3 GB | derived data, not "new data" |
| New raw data | 0 | 0 | data: stays at 20 GB of 20 GB |
| **Total** | **about 13 GB at peak** | **about 7.6 GB** | |

**Requested exception:** model weights are already at 56 GB against the 30 GB limit. Permission is requested to add
**3.61 GB at peak, 1.25 GB permanently** to model weights, and to create one further environment of about 6 GB on D:
(6.6 TB free). No new raw data. Nothing is downloaded until this is approved.

## 5. Pilot (14 scans)

| Set | Scans | Reference |
|---|---|---|
| The six empty-mask misses | LUNG1-013, -113, -170, -195, -249, -299 | first-reviewer file (official) |
| The four traced controls | LUNG1-016, -285 (good overlap), LUNG1-301, -051 (thin, found in part) | first-reviewer file |
| Graded CT-RATE development cases | valid_301_a_2, valid_541_a_2, valid_832_a_2, valid_988_a_2 | none; clinician grade for pleural effusion shown beside the output (present, absent, present, absent) |

Steps:

1. Download, verify MD5, extract the needed files, delete the zip.
2. Build the environment; run one synthetic volume. **Go / no-go for Windows.**
3. Run the 14 scans with `scripts/pleural_armB_run.py` (written, not yet run). Published configuration, no tuning.
4. Recorded per scan: seconds, peak GPU memory, label volumes, failure text, mask hash. Failures are kept in the table.
5. Metrics with `scripts/pleural_compare_metrics.py` (tested on Arm A).

Arm A on the same 14 scans, already computed (`runs/E17_.../pilot/metrics_A_totalsegmentator_315.csv`):

| Set | Result |
|---|---|
| Six misses | 0 voxels on all six; Dice 0 (empty predictions included) |
| Controls | 110.8 mL (71% of the reference outlined), 142.5 mL (84%), 2.3 mL (17%), 3.5 mL (8%) |
| CT-RATE | 262.1 mL (grade present), 0 (absent), 18.4 mL (present), 0 (absent) |

The pilot is successful if Arm B runs on all 14 without a setting chosen by us and label 1 is lung-sized. Its
segmentation results do not decide success: a pilot in which Arm B also finds nothing on the six is still a
successful pilot.

## 6. Fixed development comparison (after a successful pilot)

| Set | Scans | Reference | Treated as |
|---|---|---|---|
| PleThora, coded effusion | 77 | first-reviewer file | reference-based metrics |
| PleThora, coded no effusion | 78 | none | **not confirmed negatives**: predictions described only |
| CT-RATE development runs | 58 | none; 4 with a clinician grade | agreement between arms only |

Rules, fixed now and implemented in `scripts/pleural_compare_metrics.py`:

- **Empty predictions are included** in every segmentation metric: Dice 0, share outlined 0.
- All 77 effusion scans are in the primary analysis, the six misses and LUNG1-143 included.
- Second and third reviewer files are reported in their own columns, never merged.
- No Dice and no false-positive rate on scans without an expert outline.
- Unrounded volumes in the CSV. Counts with scan identifiers. No significance test, no single score.
- Paired tables, A against B: the six misses; the 17 under-outlined; the 10 positive outputs on scans coded no
  effusion and the 16 over-outlined; the 15 good-overlap controls; 4 x 4 mask-state tables for 77, 78 and 58 scans.
- Sensitivity lines: without LUNG1-143; and, if the list is obtained, without the 34 scans of Arm B's paper.

## 7. How the result may be read

- May say: "on these development scans Arm B outlined fluid on n of the 6 scans where Arm A outlined none".
- May not say that Arm B is better, more sensitive or more accurate; may not be used to choose limits.
- A clear difference leads to a separate plan on scans not yet inspected. That plan is not part of this one.

## 8. Stop conditions

- The storage exception is not granted.
- The archive MD5 does not match.
- nnU-Net v1 does not run on this machine without changes to its code.
- Label 1 is not lung-sized on the first pilot scan.
- Arm B needs a setting chosen by us.

Sources: https://zenodo.org/records/21918112 , https://github.com/usb-radiology/pleuraleffusion ,
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9390225/ , https://pypi.org/project/nnunet/ ,
https://zenodo.org/records/20272295 .

## 9. Execution record (2026-11-23)

- **Provenance of the checkpoints.** Chain: paper ("The algorithms are openly available at
  https://github.com/usb-radiology/pleuraleffusion.git") -> repository README ("nnU-Net model download link:
  https://doi.org/10.5281/zenodo.21918112") -> Zenodo record 21918112 by Shan Yang, a co-author. Archive MD5 computed
  here equals the value on the record (`52f3dab3815733018367a698c77400b0`); SHA-256
  `3834a92ea9a6410bfc1ba678575fc7394a2e287c5da16cbacd71fb37c2344215`; hashes of the 11 extracted files in
  `armB_checkpoint_provenance.json`. The MD5 shows the file is the one Zenodo serves. It says nothing about what the
  checkpoint code does. **The separate Python environment is not a security sandbox**: it separates package versions
  only, and the pickle checkpoints ran with this user account's full rights.
- **Windows.** nnU-Net v1's own entry point failed (its worker processes cannot start: PicklingError; log
  `runs/test_logs/armB_go_nogo.log`). No nnU-Net code was changed. `scripts/pleural_armB_run.py` calls the same
  nnU-Net functions in the same order as `predict_cases`, in one process: same preprocessing function, five folds,
  mirroring, step 0.5, final checkpoints, nnU-Net's own export. Not verifiable here: bit-identity with the
  multi-process entry point. Recorded as a deviation in how the code is driven, not in the model configuration.
- **Published preprocessing.** nnU-Net's preprocessing from the model's own plans (2.0 mm, CT normalisation) was
  used unchanged. The authors' own downsampling tool, applied before nnU-Net in their pipeline, is not published and
  was not reproduced.
- **Metric handling** (unit-tested, `tests/test_pleural_compare_metrics.py`): empty prediction on a non-empty
  reference = Dice 0, kept in; missing reference = no segmentation score, not a negative; verified empty reference =
  no Dice (never 1 for empty on empty), predicted volume reported in its own group. No scan in these sets has a
  verified empty reference.
- **Improved / worsened rule**, fixed in `scripts/pleural_compare_report.py` before the pilot ran: Dice difference
  of at least 0.10 in either direction; otherwise "similar".
- **Pilot acceptance** and the full results: see the report.

## 10. Full comparison record (2026-11-23, evening)

- Run on the fixed 213 scans after the pilot passed its technical criteria; the set was not changed after the pilot.
- 212 scans completed in the main run. LUNG1-104 was stopped by the runner's lung-size guard (label 1 = 18.0 L on an
  89 cm scan); it was re-run with `--lung-guard warn`, saved, scored and flagged. No scan is missing from the scores.
- Technical checks on 213: grid 213, orientation 213, volume 213; the too-strict class criterion flags 21 (reported).
- New material finding: without a chest crop the alternative labels tissue outside the chest as effusion on 5 of the
  7 scans of 50 cm or more. This did not change feasibility of the comparison; it limits use of the model as released.
- Result and limitations: `runs/E17_pleural_model_comparison_prep_20261123/REPORT.md`.
- Disk retained about 6.7 GB (allowance 7.6 GB); peak allowance of 13 GB not exceeded.

## 11. Closed (2026-11-23)

- Conclusion: the comparison does not support replacing the current pipeline. Not a finding of equivalence; not a
  validation of the current model. The six target misses remain unresolved.
- What was compared is two inference pipelines as implemented here (chest crop versus whole scan; authors'
  downsampling step not reproduced): REPORT.md revision 2, section 1.
- Primary analysis keeps all 77 expert-reference scans; the result without LUNG1-104 is a labelled sensitivity analysis.
- The 21 class-check warnings were reviewed per scan (`full/class_review*`); freeze record in `freeze/`.
- No further model runs or training without a new experiment brief.
