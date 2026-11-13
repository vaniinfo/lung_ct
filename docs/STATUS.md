# Project Status

Last updated: 2026-11-08

## Phase 0: Environment Bootstrap
- **Status**: DONE
- **Implemented**: Hardware detection, project structure, PROJECT_GUIDELINES.md, `.venv` (core) with CUDA PyTorch
- **Tested**: RTX 3090 24GB visible; `torch 2.14.1+cu126`, `torch.cuda.is_available() == True`; 13 unit tests pass (`pytest -q`)
- **Blocked**: none
- **Done**: separate `environments/segmentation` and `environments/vlm` environments in use (Phases 2-3)

## Phase 1: Data Ingestion and QC
- **Status**: DONE
- **Implemented**:
  - `scripts/download_covid_ct.py`: pinned Zenodo MD5 verification (incl. pre-existing files), atomic `.part` downloads, zip path-traversal guard
  - `lung_ct.ingest.nifti_loader.assess_hu`: explicit HU classification (display-scaled / unsigned / out-of-bounds / no-air -> invalid; >3071 reported, not invalidating)
  - `lung_ct.qc.laterality`: left/right label verification from world-space (RAS) centroids
  - `lung_ct.qc.qc_views`: reorients to canonical RAS before display, so screen labels are correct for any on-disk orientation; aspect = vertical/horizontal spacing; refuses overlay on geometry mismatch
  - `scripts/batch_qc.py`: 1 -> 5 -> 20 batch QC with per-case manifest + summary.csv; failures recorded, not dropped
- **Fixed defects found during review**:
  - QC sagittal labels were swapped (A/P) and coronal/sagittal aspect was inverted; axial labels were only correct by coincidence for LAS input. E0 run `runs/E0_coronacases_001_1794109858/qc` PNGs predate the fix - sagittal images there are mislabelled.
  - HU rule rejected genuine HU volumes with metal/dense contrast (max > 5000); now reported as a note.
- **Tested** (31 unit tests passing; `runs/P1_qc_20cases_20261107_223515`):
  - Archives match Zenodo MD5; 20 CT + 20 lung + 20 infection masks
  - All 20 cases: CT/mask geometry match, lung labels exactly {0,1,2}, laterality verified (label 1 = patient left), orientation LAS
  - **coronacases_001-010 (10 cases): genuine HU** (int16, air -1021/-1023, lung median -737 to -868 HU). 4 contain a few voxels > 3071 (<=0.0013%).
  - **radiopaedia_* (10 cases): display-scaled uint8 0-255 -> excluded from all HU-dependent analysis.** Reference volumes/burden still computed (geometry-only).
  - Reference infection outside reference lung: 3 radiopaedia cases, 1-104 voxels (boundary disagreement, <=0.03%)
  - radiopaedia_29_86490_1: tiny but non-empty infection mask (163 voxels, 0.014%)
- **Blocked**: none
- **Implication for Phase 2+**: HU-dependent metrics (density, thresholds) only on the 10 coronacases. TotalSegmentator on 8-bit radiopaedia volumes is out-of-distribution; report separately if run at all.

## Phase 2: Segmentation and Quantification
- **Status**: DONE for the 10 HU-valid cases (lung lobes, lung/airway/vessel masks)
- **Implemented**:
  - `scripts/segment_totalseg.py` (segmentation env): TotalSegmentator 2.18.0 `total` (roi_subset: 5 lobes + trachea) and `lung_vessels` (airways, airway wall, arteries, veins), normal resolution; class map checked up front; per-case `seg_manifest.json` with versions, weight SHA-256, settings, runtime, GPU memory; skips non-HU cases unless `--include-non-hu`
  - `lung_ct.evaluation.lung_seg` + `scripts/evaluate_segmentation.py` (core env): lobe volumes, predicted L/R lung vs reference (Dice, signed volume error), predicted laterality, airway/vessel volumes, overlay QC. Missing/empty/off-grid masks reported, never counted as 0 mL
  - Weights copied to `models/totalsegmentator` (`TOTALSEG_HOME_DIR`), usage statistics disabled in that config (the C: user config had `send_usage_stats: true`)
- **Fixed**: Windows nnU-Net worker crash ("OpenBLAS error: Memory allocation still failed") -> BLAS/OpenMP threads capped to 1 before numpy import. Old `run_segmentation.py` (mixed envs, missing mask -> 0 mL) removed; its incomplete case-001 output was replaced.
- **Tested** (36 unit tests; `runs/P2_seg_10cases_20261107_225107`, `runs/P2_eval_10cases_20261107_230212`):
  - 10/10 coronacases segmented, all 10 expected structures present, ~62-74 s/case on RTX 3090, peak in-process GPU allocation 1.8-5.6 GB
  - Lung Dice vs expert masks: left 0.974-0.985, right 0.976-0.987; volume error -1.2% to +0.8%; predicted laterality verified in all 10
  - Lowest Dice in the two highest-burden cases (003: 29.6%, 010: 17.7%) - observation only, n=10
- **Not done / limits**:
  - No reference for lobes, airways or vessels in this dataset -> those volumes are unvalidated predictions
  - 10 radiopaedia (8-bit, non-HU) cases not segmented (out-of-distribution); decide whether to run them as a separately-reported robustness check
  - Overlay QC uses one colour for all lobes (checks lung coverage, not lobe boundaries)
- **Next**: Phase 3 - MedGemma 1.5 4B CT inference (vlm env, HF token via env var only)

## Phase 3: MedGemma CT Inference
- **Status**: DONE (acceptance: 1 + 5 cases executed, raw outputs preserved, 0 failures). Execution only - no accuracy claim.
- **Implemented**:
  - Official notebook pinned: Google-Health/medgemma `ea92d0f` (`docs/upstream/`, sha256 recorded in config). Discrepancy noted: notebook markdown says green window 135..215 HU, code uses -135..215; code followed.
  - `configs/medgemma_ct_v1.json` (frozen): model `google/medgemma-1.5-4b-it` @ `91850547`, bf16, device_map auto, no quantization; representation; prompt v1; greedy, max_new_tokens 1024
  - `lung_ct.vlm.ct_representation`: notebook sampling formula (85 slices, index 0 skipped as in notebook), 3-window RGB, JPEG q75; explicit reorientation to radiological layout (rows A->P, cols R->L) from the affine, superior->inferior order, per-slice world coordinates + image hashes; refuses non-HU volumes
  - `lung_ct.vlm.prompting`: notebook message layout (instruction, image + `SLICE n` x N, query); neutral prompt (no case IDs/labels); strict JSON schema (assessability, opacity_presence, distribution{laterality, craniocaudal, axial}, description, limitations); no repair/coercion; consistency warnings flagged only; no retry
  - `scripts/run_medgemma.py`: raw output saved before parsing; token counts, runtime, peak GPU memory, device map/offload, weight SHA-256, git commit, config/prompt hashes; failures.csv; `--dry-run`
- **Tested**: 51 unit tests passing. Dry run `runs/P3_vlm_dry_10cases_20261108_071944`: 10/10 HU cases -> 85 images each; 10 radiopaedia cases skipped (non-HU). Visual check of coronacases_001 slice 45: anterior top, patient left on image right (correct).
- **Runs**:
  - **Official baseline: `runs/P3_vlm_5cases_20261108_073705`** (coronacases_001-005, torchvision fast `Gemma3ImageProcessor`, as in the notebook)
  - Superseded (PIL processor fallback, torchvision missing): `runs/P3_vlm_1cases_20261108_072222`, `runs/P3_vlm_5cases_20261108_072921` - kept as a named processor-sensitivity comparison, not the baseline
  - Weights `google/medgemma-1.5-4b-it` @ `91850547` (SHA-256 in manifest), bf16, fully on cuda:0, no offload/quantization; 8.2 GB after load. vlm env: torch 2.14.1+cu126, torchvision 0.29.1+cu126, transformers 5.22.2
  - Per case: 85 images, 22,801 input tokens (21,760 image tokens), 93-141 output tokens, 25-28 s generate, peak allocated 17.4 GB / reserved 20.0 GB
  - Parse: 5/5 schema-valid (whole_text JSON), no truncation, no consistency warnings; greedy output repeatable (case 001 byte-identical across two PIL runs)
- **Observations (unevaluated model output, not findings)**:
  - All 5 cases got identical structured fields (assessable / present / bilateral / lower_predominant / peripheral) -> low output variance; whether this reflects the images or a prior is for the evaluation stage (all-COVID set cannot test specificity)
  - Case 004 states the limitation "The images show a single slice through the chest", which is factually wrong (85 slices were given) -> unsupported model statement for the reviewer sheet
  - Processor sensitivity (fast vs PIL image processor, same images/prompt): structured fields identical in 5/5; raw text differs in 3/5 (001, 004, 005). 004 gains a "small area of consolidation in the left lower lobe" claim; 005 loses the "single slice" limitation. Free-text details are not stable under minor preprocessing changes.
- **Next**: Phase 4 - case reports, reviewer_template.csv, E2 on the 5 baseline cases
## Phase 4: Reviewable Outputs
- **Status**: DONE for E0-E3 (E4 optional, not run). Awaiting clinician review of `reviewer_template.csv`.
- **Implemented**:
  - `lung_ct.reporting.case_report`: standalone HTML (base64 images, all model text HTML-escaped), separate REFERENCE / PREDICTED / MODEL OBSERVATION sections, stage-status table, contact sheet of the exact 85 model-visible images, reviewer rows (clinician columns blank)
  - `lung_ct.metrics.volumes.infection_by_side`: expert infection annotation volume per lung, using labels verified from world coordinates in Phase 1
  - `scripts/build_reports.py`: reads QC/seg/eval/VLM runs (input manifest SHA-256 recorded); writes per-case qc.json, metrics.json, vlm_raw/parsed/result, overlays, case_report.html; run_manifest.json, batch_results.csv, failures.csv, reviewer_template.csv, feasibility_summary.md
- **Runs**: E3 VLM `runs/P3_vlm_10cases_20261108_083032` (10/10 schema-valid; cases 001-005 byte-identical to the 5-case baseline run); reports `runs/P4_report_20cases_20261108_083529` (20 reports, no external references checked, 213 MB)
- **Measured (see feasibility_summary.md)**:
  - E0 coronacases_001: all stages ok
  - E1 first 5 HU CTs: lung Dice L 0.977-0.984, R 0.979-0.987; volume error L -0.03 to +0.79 %, R -0.40 to +0.45 %
  - E2 same 5: 5/5 schema-valid, raw outputs preserved, reviewer sheet generated
  - E3 20 CTs: 10 HU-valid processed by all stages (seg 10/10, VLM 10/10 valid); 10 display-scaled excluded from HU stages (listed in failures.csv). Seg 61-74 s/case, peak 5.6 GB; MedGemma 25-27 s/case, peak 17.4 GB
- **Observations for review (not evaluated results)**:
  - All 10 valid outputs share the same (assessable, present, bilateral); 1 distinct combination
  - Laterality vs reference infection annotation: coronacases_006 annotation is right-only (L 0 mL, R 130 mL) and 009 nearly right-only (L 2 mL, R 103 mL), but the model said "bilateral" for both. The annotation is not an opacity label, so this needs clinician review, not automatic scoring.
  - "Single slice" limitation (factually wrong, 85 slices given) appears in 004 and 009
- **Not done**: E4 (CT-RATE); formal evaluation stage (concordance scoring) - should be predefined before running; reports are large (QC PNGs embedded at full size)
- **Next**: clinician review of reviewer_template.csv; optionally E4

## Phase 5: Airway Extension (Python)
- **Status**: DONE for scope step 1 (lung volumes, airway centrelines, branch lengths, tortuosity) on the 10 HU cases. Unvalidated measurements.
- **Plan change (2026-11-08, user decision)**: MATLAB is unavailable; MATLAB and AirQuant removed from implementation dependencies (original project plan and PROJECT_GUIDELINES.md updated). Python pipeline: nibabel/SimpleITK geometry, scikit-image skeletonisation, Skan branch graph, explicit definitions. AirQuant = methodological reference only; outputs labelled "lung_ct airway tree v2 ... NOT AirQuant". Diameter/tapering = later validated extension; wall thickness deferred.
- **Implemented** (`lung_ct.airway.tree`, `scripts/airway_tree.py`; core env + scikit-image 0.26, skan 0.13.1, SimpleITK 2.5.7):
  - Input: predicted TotalSegmentator `lung_airways` (includes trachea); canonical RAS; largest 26-component, fragments reported; isotropic nearest-neighbour resampling
  - Lee 3D skeleton -> `skan.Skeleton(spacing=iso)`; Skan paths give pixel paths, raw path length (`skan_path_length_mm`) and endpoint coordinates
  - Explicit rules: spur pruning (terminal < max(2 mm, 1.5 x local EDT scale)); contraction of junction-junction paths < max(local scale, 1.75 voxels) and of loops with perimeter < 2*pi x local scale; branch = chain between endpoints/bifurcations; root = most superior endpoint; generation = depth in shortest-path tree (trachea 0); side from world x of generation-1 branches
  - length_mm = Gaussian-smoothed centreline length (raw Skan length overestimates oblique paths: tortuosity 1.23 for a straight tube at 0.7x0.7x1.5 mm); tortuosity = length_mm / endpoint_distance_mm
  - EDT used only as an internal local scale for thresholds; no diameter column is output (test enforces)
  - Lung volumes (predicted and reference, Dice) carried from the Phase 2 evaluation into the Phase 5 summary
  - QC image: centreline by generation on a CT coronal minimum-intensity slab and on the mask MIP (radiological display; points mapped through world coordinates)
- **Validation (67 tests)**: synthetic Y-bifurcation at 1 mm isotropic and 0.7x0.7x1.5 mm stored LAS (topology, lengths within ~1-2 mm, tortuosity ~1.0, side from world x); two-level bifurcation (generations {0:1, 1:2, 2:2}, 5 branches, 3 terminal); curved tube tortuosity 1.07-1.16 vs true 1.11; spur pruned (kept when pruning disabled); fragment removal; hole in trachea does not create a false bifurcation; raw Skan length bias documented; SimpleITK LPS vs nibabel RAS coordinates agree after explicit sign flip
- **Defects found and fixed during development**: staircase length bias (-> smoothed centreline); small loops inside the trachea (cases 002/005) created false bifurcations and shifted generations (-> loop contraction + degree-2 merging, regression test)
- **Run** `runs/P5_airway_10cases_20261108_085954` (10/10 ok, 2.2-5 s/case), CT overlays inspected for 001 and 010:
  - Lung volume predicted 2286-4778 mL (Dice vs reference 0.975-0.985)
  - All 10: 1 trachea, 2 generation-1 branches, 4-5 generation-2, 8-10 generation-3 branches; right generation-1 branch shorter than left in 9/10 (R 16-30 mm, L 46-59 mm)
  - Tree branches 141-376, terminal 70-175, max generation 11-13, total centreline 1515-5048 mm, median internal tortuosity 1.013-1.030
  - Remaining cycles 2-29 per case, mostly where ~2-voxel peripheral branches touch in the predicted mask -> flagged
  - coronacases_010: left generation-1 branch only 14.1 mm - a short bud on the left main bronchus near the carina in the predicted mask survives pruning; left-side generations in 010 likely shifted by one; flagged for review (not tuned away)
- **Limits / not claims**: no airway reference in this dataset; all airway values depend on the TotalSegmentator mask, slice thickness (1-1.5 mm), inspiration and reconstruction; fewer branches do not indicate airway loss; peripheral counts/generations are sensitive to pruning/contraction parameters
- **Next**: (1) parameter-sensitivity sweep (prune/contract factors) and a visual review of more CT overlays; (2) diameter/tapering only with a validation design (phantom or annotated data, e.g. ATM22); (3) optional NaviAirway comparison on annotated data

## Validation round 1 (2026-11-08) - baseline frozen, airway stability, review targets
- **Baseline**: tag `baseline-feasibility-v1` (`e62ce04`). All runs below from clean commits (no `+dirty`); current reports from `6821032`.
- **Current artifacts**: reports `runs/P4_report_20cases_20261108_092959` (supersedes `..._083529`); lobe overlays `runs/P2_eval_10cases_20261108_092055` (metrics identical to `..._230212`); airway `runs/P5_airway_10cases_20261108_092858`; sensitivity `runs/P5_sensitivity_10cases_20261108_092327`
- **Airway parameter sensitivity** (one-at-a-time: prune radius factor 1.0/2.0, prune min 1/4 mm, contraction 0.5/1.5, loop factor pi/3pi, smoothing 0.5x/2x; ratings predeclared: counts gen 0-3 must be identical, else <=5% stable, <=15% moderate):
  - Stable: generation 0-1 structure (10/10), trachea length (8 stable / 2 moderate), main bronchus lengths (right 9/1/0; left 7/2/1 - case 010 unstable)
  - Moderate: total centreline length (10/10 moderate, 6-9%), median length of generation 2-4 branches (2/7/1)
  - Unstable: total and terminal branch counts (7/10 unstable, ~20% range, driven by prune_min_mm), max generation (5/10), cycles (10/10), generation-2 counts (4/10) and generation-3 counts (5/10)
  - Tortuosity: rated on excess over 1 -> unstable in 10/10 (see corrected wording and ranges in "Clinician review preparation" below); case ranking only moderately preserved (Spearman >= 0.83 all internal branches, >= 0.56 generations 2-4)
  - Case 010: left main bronchus 14.1 mm at default vs 54.0 mm with prune_radius_factor 2.0 -> a short spur on the left main bronchus; left-side generations at default are shifted
- **Visual review of all 10 airway overlays** (CT minIP slab + mask MIP): proximal tree plausible in all (midline trachea, carina, main and lobar bronchi). Generation numbers beyond ~3 are graph depth, not anatomical generation: in 003, 005, 006, 008, 009 branches a few cm from the hilum carry generation 8-13 (side buds/residual loops add depth). Reported as such in case reports.
- **Reliability marking**: each case report lists every airway metric with its range across settings and stability rating; batch_results.csv lists unstable metrics per case.
- **MedGemma review targets** (rule-based, `review_targets.csv`, also in reviewer_template.csv `automated_flags`; flags prioritise review, not correctness):
  - input_statement_contradiction ("single slice" with 85 images): 004, 009
  - laterality_vs_reference_annotation (model "bilateral", annotation >95% one side): 004 (left-only), 005, 006, 009 (right) - annotation is not exhaustive
  - text_changes_with_image_processor (vs PIL-processor run, available for 001-005 only): 001, 004, 005; structured fields unchanged
  - all 10 valid outputs share one structured combination (assessable/present/bilateral)
- **Reporting fixes**: Phase 5 airway section in case reports; one colour per lobe with legend, sagittal views through each lung centre, readable titles; stage outcomes categorised (execution failure: none; planned exclusion: 10 display-scaled cases at segmentation/eval/airway/VLM; QC warning: 10)
- **Tests**: 71 passing (adds sensitivity rating rules incl. tortuosity excess, review-flag rules, outcome categories)
- **Next**: superseded by "Clinician review preparation" below

## Clinician review preparation (2026-11-08)
- **Status**: review package READY; awaiting clinician. No model added, no prompt change, no airway measurement added; no model inference rerun. Frozen tag `baseline-feasibility-v1` (`e62ce04`) and its outputs untouched.
- **Code**: `d86bf63` (clean). Report `runs/P4_report_20cases_20261108_095004`; package `runs/P6_clinician_review_v1_20261108_095014` (1.4 GB; includes byte-identical CT copies). Builder: `scripts/build_review_package.py`.
- **Reliability wording** (generated from the per-case ratings, the single source for case reports and the summary): "Central-airway measurements were generally stable across tested settings; anatomical accuracy remains unvalidated." Exception: coronacases_010 left main bronchus 14 mm at default vs 54 mm with prune_radius_factor 2.0. Parameter stability is reported separately from anatomical accuracy everywhere.
- **Tortuosity** (median per case, range across 10 cases): smoothing 0.5x 1.034-1.067 (minus 1: 0.034-0.067); default 1.013-1.030 (0.013-0.030); 2x 1.005-1.012 (0.005-0.012); generations 2-4 similar. The current tortuosity implementation is too sensitive to centreline smoothing for patient comparisons.
- **Flag rules**: `configs/review_flag_rules_v1.json` (sha256 b9154d50de83...), thresholds recorded (unilateral fraction 0.05; "single slice" regex). EXPLORATORY: written after inspecting this pilot, not prospectively defined criteria. Kinds: input_description_error ("single slice": 004, 009 - incorrect description of the supplied model input); review_flag (laterality vs non-exhaustive annotation: 004, 005, 006, 009 - not confirmed errors); processing_sensitivity (text changed with image processor: 001, 004, 005; only 001-005 comparable).
- **Review package** (all 10 HU-valid cases):
  - View 1 (blinded): neutral codes R01-R10, randomised (seed 20261108); full examination (all axial slices, lung and mediastinal windows, scrollable local HTML) plus original CT NIfTI for multiplanar review; no model output, annotation, flags or case names (checked by text/filename scan)
  - View 2 (comparison): verbatim and parsed model response, scrollable reference-annotation overlay on the full CT, flags (highlighted only here), link to the exact model-input contact sheet (kept separately in `model_input_contact_sheets/`), unblinding key
  - Blank sheets: `scoring_view1_independent.csv` (assessability, opacity presence, laterality, cranio-caudal and axial distribution, other findings, comments), `scoring_view2_comparison.csv` (per item: agreement, error type incl. unsupported claim / omission / incorrect input description, comments), `data_dictionary.csv`
  - Central-airway landmark sheet: coronacases_010 (exception), 009 (shortest right main), 006 (longest left main), 001 (E0 reference); rules defined by the analyst after the sensitivity sweep; per segment the exact endpoints and centreline (coronal + axial crops at each endpoint; 010 also shows the alternative-setting endpoint); case IDs only (no View 1 codes); explicitly not a cohort validation
- **Checks**: 78 tests (adds viewer radiological orientation for 4 storage orientations, aggregate/central statement consistency, versioned flag rules); package scan: no View 1 leaks, no external references, CT copies hash-identical, all judgement cells blank, every source slice rendered; re-analysis of 001/010 identical to committed airway run
- **Decisions needed (clinician / professor)**: see end of this section in the delivery message; recorded here:
  1. Who reviews (number of readers, specialty/seniority) and whether a second reader or consensus is required
  2. Whether the reference for View 2 should remain the COVID-19 infection annotation or be supplemented by the reader's own View 1 labels as the primary reference
  3. Error taxonomy and agreement categories in the data dictionary - accept or amend before scoring starts
  4. Whether laterality/distribution thresholds (e.g. "predominant") need explicit definitions for readers
  5. Central-airway landmarks: confirm intended landmarks (carina; right upper lobe bronchus origin; left secondary carina) and whether tracheal length should be reported at all given the mask-limited proximal end
  6. Whether the 4-case airway subset is sufficient for this round or all 10 should be reviewed
  7. After review: whether airway work proceeds (anatomical labelling vs graph depth, annotated airway dataset) and whether E4 (CT-RATE reports) is wanted
  8. Data sharing: the package contains public, de-identified CT copies (dataset licence applies); confirm how it is transferred to the reviewer

## Review package revision v1.1 (2026-11-08, documentation/layout only)
- **Status**: package `runs/P6_clinician_review_v1_20261108_095014` revised to v1.1; ready for handoff 1. No inference rerun, no clinical judgement populated, no image/CT/viewer re-rendered.
- **Commits**: build commit `d86bf63` (generated all content; recorded in the preserved, byte-identical `analyst_only/package_manifest.json`); revision commit `7c37b51` (`scripts/revise_review_package.py`, recorded in `analyst_only/revision_v1.1.json` and `REVISION_LOG.md`); the final package commit is the commit that adds this STATUS entry. The build manifest is intentionally not rewritten so generated content stays traceable to the code that produced it.
- **Integrity**: SHA-256 inventories before/after (`analyst_only/file_inventory_v1.0_build.json`, `..._v1.1_revised.json`): 7786 files byte-identical (moved only); 22 documentation/template files modified (10 comparison.html, 10 annotation_viewer.html, View 2 index.html, View 2 scoring sheet); all relative links and viewer image sets resolve.
- **Changes**:
  1. View 1 lock: reviewer saves the unchanged View 1 file as `scoring_view1_independent_<reviewer_id>_<YYYYMMDD>.csv` and sends it; analyst records its SHA-256 in `analyst_only/view1_lock_record.csv` before releasing handoff 2; later changes go to `view1_revisions.csv` (handoff 2)
  2. Primary reference (infection annotation vs reader View 1) and distribution definitions (zones, "predominant") marked PENDING clinician/professor approval in READMEs, data dictionaries and every View 2 page
  3. not_applicable (question does not apply given an earlier answer) vs indeterminate (applies but cannot be decided), with consistency rules; error_type accepts multiple codes separated by ";" without spaces ("none" alone)
  4. Omissions judged against the actual prompt; verbatim prompt v1 (sha256 26d7c99692e2) shown on every View 2 page, absent from handoff 1 (scanned)
  5. New View 2 column `visible_in_model_input`: yes | no | uncertain | not_checked
  6. Staged layout: `handoff_1_blinded/` (View 1 only) -> `handoff_2_comparison/` -> `handoff_3_airway/`; `analyst_only/` holds build manifest, case key, lock record, revision log, inventories and superseded v1.0 documents; handoff 1 scanned for manifest, key, airway material, prompt and identifying text
- **Provenance note**: the MedGemma run shown in View 2 (`runs/P3_vlm_10cases_20261108_083032`) was produced from `ad20e6f+dirty` (Phase 3 code before its commit `30895e5`); inference not rerun.
- **Decisions still pending** (unchanged): readers; primary reference; distribution definitions; scoring categories; airway landmarks and tracheal length; airway subset size; post-review direction; package transfer.
