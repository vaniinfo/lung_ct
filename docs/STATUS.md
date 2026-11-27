# Project Status

Last updated: 2026-11-22

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

## CT-RATE condition checklist: test20 result (2026-11-17)
- **Status**: checklist_v2 run on 20 fresh CT-RATE cases (set `test20`) DONE and scored. Earlier agent / checklist history (2026-11-14 to 2026-11-17) is in the revision log of `docs/PROJECT_PLAN.md`; the pipeline summary is in `docs/HANDOFF.md`.
- **Implemented**: `run_ctrate_test20.bat` (download, segmentation, anatomy check, COLIPRI, MedGemma and Hulu-Med readers, checklist_v2 agent, scoring). Renderer fix in `scripts/agent_render_checklist.py`: "please check" and "limitations" items written by MedGemma as `{text, evidence}` objects are now accepted (display only; no rule or threshold changed).
- **Tested** (101 unit tests passing):
  - All 20 cases completed every stage, 0 failed: `runs/P1_qc_ctrate_20cases_20261117_181838`, `runs/P2_seg_20cases_20261117_182540`, `runs/P1_qc_verified_P1_qc_ctrate_20cases_20261117_181838_191500` (20 verified: left/right from anatomy, all non-contrast), `runs/P3_colipri_20cases_20261117_192152`, `runs/P3_vlm_20cases_20261117_192335`, `runs/P3_hulumed_20cases_20261117_194041`, 20 `runs/A1_agent_valid_*_20261117_19*` / `_20*` folders
  - Scores `runs/E6_checklist_scores_20261117_205523/scores.md` (360 answers; 59 labelled findings): **234 agree, 11 missed, 15 extra, 100 open**. Pilot (10 cases, checklist_v1): 128 / 4 / 11 / 37. Agreement where the agent gave present/absent: 234 of 260 (90%); pilot 128 of 143 (90%). Open answers rose from 21% to 28%.
  - Per condition (agree / missed / extra / open, labelled in brackets): pleural effusion 19/1/0/0 (3); pericardial effusion 18/0/0/2 (3); cardiomegaly 18/1/0/1 (3); medical material 18/1/0/1 (1); consolidation 16/1/1/2 (3); increased lung density 16/1/2/1 (4); hiatal hernia 16/0/0/4 (1); coronary calcification 16/1/0/3 (2); interlobular septal thickening 14/0/0/6 (4); enlarged lymph nodes 14/2/0/4 (4); emphysema 13/2/5/0 (4); aortic wall calcification 13/0/7/0 (2); mosaic attenuation 13/0/0/7 (3); atelectasis 12/0/0/8 (3); fibrotic changes 9/1/0/10 (5); peribronchial thickening 7/0/0/13 (1); bronchiectasis 2/0/0/18 (3); lung nodule 0/0/0/20 (10)
  - Report writer: MedGemma wrote 15 of 20 reports; rule-based fallback for valid_491, valid_572, valid_1200, valid_313, valid_126
- **Observations (not acted on; test20 must not be used for tuning)**:
  - 4 of the 5 emphysema extras and 5 of the 7 aortic calcification extras are on scans with image noise (E69) of 48-57 HU, just below the 60 HU cut-off above which checklist_v2 measures on the smoothed copy. Scans above 60 HU were mostly clean.
  - Extras are judged against text-derived labels; a finding the radiologist did not mention may still be real (aortic calcification especially). Not yet checked by eye.
  - Most misses sit in two heavily diseased cases: valid_572 (consolidation, increased density, lymph nodes) and valid_72 (coronary calcification, lymph nodes; plus extra consolidation and density).
- **Run defects**:
  - The first pass stopped at the 5th case (valid_471) on the renderer crash; the batch file then scored the old pilot page list, so `runs/E6_checklist_scores_20261117_200945` is a duplicate of the pilot scores, not a test20 result. After the fix the remaining 15 cases were run with the same reader and classifier runs and all 20 run folders were scored directly.
  - `run_ctrate_test20.bat` still continues to scoring when the agent step fails (not fixed). `runs/test_logs/agent_demo_pages.txt` lists only the last 15 cases.
  - The download used the Hugging Face login saved on the PC (`HF_TOKEN` was not set).
- **Blocked**: none
- **Next**: review the 26 missed / extra answers on the `review_checklist.html` pages and with the clinician reviewer; check the AI reader answers on test20; nodule detector, airway tree on CT-RATE (bronchiectasis), lymph-node tool; noise rule change defined in advance and judged on a third fresh case set

## Location-first viewer, grading form v2, checklist_v3_4 (2026-11-22)
- **Status**: implemented and verified on the 58 already-seen soft-kernel runs. Presentation and review-form update only. NOT a validation of any threshold: The clinician reviewer's preference for descriptive, location-first wording says nothing about whether the cut-offs are right. Nothing committed.
- **Trigger**: reviewer feedback relayed 2026-11-22 ("we don't quantify ground-glass, consolidations or density; we describe them, and location matters as much as extent").
- **Implemented**:
  - Measurements (extra rows only): `lung_ct.agent.conditions.lobe_distribution` / `fissure_distance_mm`; evidence rows `E83`, `E89`, `E91` = full per-lobe distribution (mL, % of that lobe, % of the selected region, share within 5 mm of a predicted fissure, share outside lobe outlines) of the ground-glass-range region, the consolidation-range region and dense lung next to pleural fluid. Enabled by `measurements.lobe_distribution` (absent before v3_4, so older configs give the same rows as before). `scripts/add_lobe_distribution.py` appends them to finished runs after recomputing every existing row and requiring an exact match.
  - `configs/checklist_v3_4.json` (PROVISIONAL CANDIDATE; default for new runs is still v3_2): decision rules, cut-offs, gates and measurement settings identical to v3_3. New `location_reporting` block = `location_rule_v1_provisional`: a lobe is named when it holds >= 10% of the selected region; "lobe assignment uncertain" when >= 50% of the region is within 5 mm of a fissure, >= 10% is outside every lobe outline, the lungs are poorly inflated, or pleural effusion is present. All limits were fixed before any per-lobe distribution was computed; not fitted, not validated. Must not be tuned on fresh cases.
  - `lung_ct.agent.checklist.location_report` / `add_locations`: new item fields `location_report`, `location_text`; statements lead with the lobes and call the region "HU-selected"; for "cannot tell" it is a "candidate region, not a confirmed finding". The older `location` string is still computed and stored unchanged. Every percentage in a generated sentence names its denominator.
  - Viewer v2 (`build_viewer_page.py --viewer v2`, `scripts/viewer_v2.py`, build id `checklist_viewer_v2_location_first`): location first, no number under the answer, click-to-open "Numbers" box, measurement panel always on the page (per-lobe tables with explicit denominators; thirds of lung height labelled geometric, not segments). Viewer v1 is still the default so `build_meeting_viewers.bat` reproduces the first pack.
  - Grading form `grades_v2` (`scripts/viewer_grading_v2.py`): finding and location are separate fields; lobes, "Indeterminate" or "Not assessed" (default); pleural effusion by side; severity / confidence choices unchanged (no new extent words). Form-v1 grades load with location `not_recorded_form_v1`. `scripts/collect_grades.py` accepts v1 and v2 files together.
  - Staged page (`--blinded`): step 1 = scan and form only; lock needs a name and an answer for every condition; then the tool's answers, locations, measurements and outlines appear; later changes are stored with a timestamp next to the locked answers; CSV has both. No CT-RATE report or labels in staged pages. Hidden content is inside the HTML file: a reading procedure, not a security barrier.
  - Finding-mask audit: `scripts/build_blinded_viewer.py` is now `blinded_viewer_v2`; stage 1 also records "is the finding there" (required) and the reader's own location (lobes / not in a lobe / indeterminate / not assessed), locked before lobe outlines are revealed. Record `audit_review_v2` (`lung_ct.audit.schema`; v1 records readable, location `not_recorded_form_v1`). Protocol draft `configs/audit_review_protocol_v0_1_draft.json` (v0 draft kept unchanged; still NOT frozen). Checks, thresholds, sampling and blinding list unchanged.
- **Tested**: 169 unit tests pass (new: `tests/test_location_v3_4.py`, one test in `tests/test_audit.py`). Browser click-through self-tests in headless Edge (form logic, staging, CSV columns) passed for the location-first page, the staged page and the audit review page.
- **Verification** (`runs/E13_location_update_verification_20261122_081803/verification.md`, PASSED):
  - Evidence: 1334 existing measurement rows recomputed from the CT (58 runs x 23 rows), 0 different (`runs/E12_lobe_distribution_20261122_073038`, `_072853`, `_081718`).
  - Code: v3_3 checklists built by the current code are identical to the files written before the update, 58 of 58.
  - Config: v3_4 vs v3_3 - status, display, measured value, basis, flags, classifier score and old location string identical for all 1160 answers. Scores vs report labels identical (found 28, missed 26, extra 16, correct absent 563, cannot tell / assess 411).
  - Text that changed: 140 statements = 55 location statements (20 on "present" answers, 35 candidate regions on "cannot tell" answers) + 85 denominator wordings (emphysema 52, ground-glass 33).
- **Observations (seen cases; hypothesis only)**:
  - 38 of the 55 location statements are flagged "lobe assignment uncertain": poorly inflated lungs 24 times, pleural effusion 13, near a fissure 1. The flag is common because diseased scans are often shallow-breath or have effusions; whether it is useful to a reader is untested.
  - The 10% rule names 1 to 5 lobes (1: 6, 2: 8, 3: 8, 4: 10, 5: 4 of 36 single-region statements). Whether readers agree with the named lobes is unknown: no clinician lobe grade exists yet.
- **Run defect**: my own consistency check in `add_lobe_distribution.py` compared a rounded total with a rounded stored value and failed valid_885_a_2 (334.5 vs 335 mL) although all 23 rows matched; check corrected (tolerance 0.51 mL) and the case re-run.
- **Artifacts**: `runs/M2_location_first_20261122/` (index, four location-first pages, one staged example; development / usability cases only), `runs/A2_audit_inventory_usability_20261120_190509/review_v2_<case>.html`, `runs/E7_reviewer_grades_20261122_075615` (the three received form-v1 grade files collected).
- **Not done**: lobe naming rule not validated; no anatomical segments (TotalSegmentator 2.18 has none for the lung); no extent words; effusion size words (trace / mild / moderate / large) not added to the form; staged page not tried by a reader; `docs/PAPER_NOVELTY.md` status table not updated; default config unchanged (v3_2).
- **Also on this working tree (2026-11-21)**: `scripts/segment_extra_task.py` + `scripts/nodule_candidates.py`: TotalSegmentator `lung_nodules` run on the 58 scans (`runs/P2_seg_lung_nodules_*`, `runs/E11_nodule_candidates_20261121_205951`). Exploratory: 18 of 23 nodule-labelled scans and 13 of 35 unlabelled scans get a candidate; 21 of 22 candidates >= 10 mm are in consolidation-labelled scans. No checklist rule uses it.
- **Next**: reader usability trial of the staged page and the lobe form on development cases; then freeze form, rule and protocol versions before any fresh evaluation case is opened.

## Viewer v3: reliable local-file opening (2026-11-22)
- **Status**: implemented and tested on the development workstation only. The clinician reviewer reported that the page "keeps crashing"; the crash has NOT been reproduced here and the new file has NOT been tried on his computer. Do not call it fixed.
- **Scope**: loader and payload embedding only (`scripts/viewer_v3.py`, `build_viewer_page.py --viewer v3`, build id `checklist_viewer_v3_lazy`). Still one standalone HTML opened from file://; no server, no install, no browser setting. Layout, answers, measurements, grading form `grades_v2`, CSV and staging are those of viewer v2. Viewer v1 / v2 still build as before; the v2 files in `runs/M2_location_first_20261122/` are kept.
- **Defaults verified, not changed** (already in the pages since 2026-11-21): axial view on opening, plain CT, overlays only after a click, coronal / sagittal / 3-plane available, no 3D render pane.
- **Inspection of the 186 MB valid_832 page**: CT = 182.1 MB of base64 (497 x 502 x 540 int16 voxels, 0.82 x 0.82 x 0.75 mm, gzip-6); eight overlays = 0.18-0.28 MB each as base64 but 134.7 MB each once decoded (uint8 on the full grid); viewer code 2.3 MB. All payloads were inside one JavaScript literal, which stays in memory for the life of the page. Every overlay that was clicked stayed loaded.
- **NiiVue 0.69.0, read from the bundled code**: the CT is uploaded as an RGBA8 3D texture (4 bytes per voxel = 539 MB for this scan) and the first overlay adds a second texture of the same size. With an overlay present, each `updateGLVolume()` (overlay on / off, window change, contrast drag) creates a new overlay texture without deleting the previous one. NiiVue has no handler for WebGL context loss.
- **Changes**:
  - CT coded losslessly (prediction from three decoded neighbours, zigzag, 1- or 3-byte codes, gzip): 182.1 -> 125 MB of base64; page 186.1 -> 128.9 MB. The browser checks a checksum and a SHA-256 of the decoded voxel bytes against values computed from the original array at build time. No resampling, no cropping beyond the existing body crop; NIfTI header embedded unchanged. (As in every earlier page, values are clipped to -1024..3071 HU before embedding.)
  - Payloads in `<script type="application/octet-stream">` blocks; the CT is decoded block by block and the blocks are removed from the page as they are consumed.
  - One overlay at a time: the previous one is removed with `removeVolume`, its references dropped, and the overlay texture reset to 2 x 2 x 2; the old overlay texture is handed to NiiVue's `rgbaTex` so it is deleted before a new one is made.
  - Loading progress on the scan; error states for CT failure, overlay failure (the queue continues), unhandled errors / promise rejections and WebGL context loss (message with a reload button). The grading form and the "Numbers" buttons are set up before the scan loads and keep working if the viewer cannot start.
- **Tested** (170 unit tests incl. a lossless round-trip test; `scripts/test_viewer_page.py` opens the unmodified file from file:// through the DevTools protocol): 5 runs, 78 checks, all pass - `runs/V1_viewer_test_{new,old}_*_20261122_09*`. Chrome 157 and Edge 158, software rendering (Microsoft Basic Render Driver) and, for Chrome, the RTX 3090.
  - Start-up: NiiVue holds 1 volume (the CT), 0 overlays decoded, axial, lung window, no render pane, scan verified by SHA-256.
  - Scrolling, window buttons, coronal / sagittal / 3 planes, 24 overlay switches (never more than CT + 1 overlay), "Show in scan", overlay off, grade entry, CSV export, damaged overlay reported and next overlay still loads, simulated context loss shows the message and grading still works.
- **Before / after, valid_832, development workstation** (working set summed over the browser's processes; software rendering unless stated):

  | | old page (v2) | new page (v3) |
  |---|---|---|
  | file size | 186.1 MB | 128.9 MB |
  | browser launch to scan shown, Chrome / Edge | 11.1 s / 11.9 s | 6.4 s / 9.3 s |
  | memory after start-up, Chrome / Edge | 3.7 GB / 3.9 GB | 3.3 GB / 3.5 GB |
  | memory after every overlay was used once, Chrome / Edge | 9.3 GB / 9.4 GB | 5.7 GB / 5.8 GB |
  | Chrome on the RTX 3090: RAM after start-up / after overlays | 2.6 GB / 5.2 GB | 2.2 GB / 2.8 GB |
  | Chrome on the RTX 3090: graphics memory after start-up / with an overlay | about 1.1 GB / 1.5-1.9 GB | about 1.2 GB / 1.7 GB |

- **What the numbers say**: memory while overlays are used is much lower and stays flat (same after 8 and after 24 switches). Start-up memory is almost unchanged: it is set by NiiVue's textures for a 135-million-voxel scan, not by the loader. The growth of the old page came mainly from every clicked overlay staying loaded; the texture leak found in the code was not shown separately to be a cause.
- **Limits**: tested headless on one workstation (128 GB RAM, RTX 3090). Not tested: his computer, a laptop with integrated graphics, low RAM, a real mouse session. With software rendering the browser kept about 5.5 GB after overlays were switched off again; on the graphics card it returned to the start-up level.
- **Open decision**: if his computer fails at start-up, the remaining lever is fewer voxels. A crop to the lungs plus 20 or 40 mm would keep 35% or 49% of the voxels (texture 187 or 266 MB instead of 539 MB) with unchanged HU values inside the crop, but it removes chest wall and upper abdomen. Not done ("no cropping yet").
- **Artifacts**: `runs/M2_location_first_20261122/*_v3.html` (index links to them; the v2 files are kept), test folders above.
- **Correction after testing on integrated graphics (2026-11-22, later)**: the reviewer uses an Intel Evo laptop (integrated graphics, shared memory). The earlier before / after table was measured with software rendering and on the RTX 3090 and must not be read as representative of his machine. Re-run on this workstation's integrated adapter (AMD Radeon Graphics, `scripts/test_viewer_page.py --integrated-gpu`, `runs/test_logs/igpu_probe.py`; Edge 158):
  - Plain CT at start-up: graphics process 2.9 GB, browser total 4.2 GB.
  - Four window changes (Lung / Soft tissue / Bone / Lung) with NO overlay: graphics process 2.9 -> 18.4 GB.
  - Then one overlay after another: 23-33 GB, then the graphics process died and the page lost its WebGL context (`runs/V1_viewer_test_new_edge_igpu_20261122_093410`, 6 checks failed). The old page shows the same collapse of the graphics process (`runs/V1_viewer_test_old_edge_igpu_20261122_093520`).
  - The new page handled it as designed (message shown, no further overlay actions, grading and CSV still worked), but the scan viewer was gone.
  - This reproduces a crash mechanism on integrated graphics: NiiVue 0.69.0 rebuilds full-size 3D textures on every refresh, and on this adapter the memory is not returned. A laptop with 8-16 GB would fail after one or two window changes. It is NOT confirmed that this is what happens on his Intel laptop (different vendor and driver), and viewer v3 does NOT solve it.
  - Consequence: for an integrated-graphics laptop the WebGL viewer itself is the problem, not the loader. Candidate fix (not built, needs a decision): a small 2D slice viewer drawn on an ordinary canvas from the decoded CT (no WebGL, no 3D textures; about 0.3-0.5 GB), keeping axial / coronal / sagittal, scrolling, window presets, one overlay at a time, the same lossless data, grading form and CSV. Left / right display would have to be re-verified from the affine.

## Viewer v4: 2D canvas reviewer page (2026-11-22)
- **Status**: built and tested on the development workstation; NOT yet opened on the clinician reviewer's Intel Evo laptop. A short usability check on his computer is required before the next grading batch. Full report: `docs/VIEWER_V4_TEST_REPORT.md`. Approved by the user on 2026-11-22; supersedes the requirement to keep NiiVue. All earlier pages are kept.
- **Viewer v3 remains unsuitable to send as a confirmed fix** for integrated graphics (`runs/M2_location_first_20261122/NOT_A_CONFIRMED_FIX.txt`).
- **Implemented**: `scripts/viewer_v4.py` (`build_viewer_page.py --viewer v4`, build `checklist_viewer_v4_canvas2d`): no WebGL, no viewer library; one 2D slice drawn on a canvas; buffers reused (one grey look-up table, one image buffer per view); outlines stored as their bounding box, decoded on request, one at a time; lossless CT coding of v3 with SHA-256 per 8 MB piece; loading and error states. Axial first, plain CT, coronal and sagittal, wheel / keys, window presets, contrast by right-drag or Shift+drag, zoom, pan, cross-hair, "Show in scan". No linked three-pane view. Layout, answers, measurements, grading form `grades_v2`, staging and CSV unchanged.
- **Geometry**: voxels are reordered to RAS+ at build time from the affine (permutation / flip only), the canonical affine and its inverse are embedded for physical positions, spacing gives the aspect ratio, and oblique scans above 0.5 degrees are refused with an explicit error (never assumed axis-aligned). Display convention is fixed and printed on the image edges (R / L / A / P / S / I).
- **Tested**: 175 unit tests (`tests/test_viewer_v4.py`: 48 axis orders / flips with hand-built affines, anisotropic spacing, oblique refusal, lossless coding, exact outline bounding box). Browser: `scripts/test_canvas_viewer.py`, 26 runs, 558 checks, 557 passed (one defect found and fixed: wheel notch moved 1.2 images). Chrome 157 and Edge 158 on the integrated adapter, file://, all four meeting cases, the staged page, and three size cases (smallest 512 x 512 x 200, largest 768 x 768 x 253, one 1024 x 1024 matrix). Drawn slices equal SimpleITK's reading of the original files voxel for voxel; drawn outlines equal the original masks; CSV round trip identical.
- **Memory (private bytes, all browser processes, integrated graphics)**: empty browser 0.43 GB (Chrome) / 0.70 GB (Edge); with a page 0.65-0.89 GB (Chrome) / 0.91-1.14 GB (Edge) after start-up; peak while loading up to 1.34 GB; no continuing growth after 200 window changes, 570 scroll steps, 30 view switches and 56 outline switches (retained change -220 to +153 MB). v3 on the same adapter: 4.2 GB at start-up, 18 GB after four window changes, then the graphics process died.
- **Integrity check**: per-piece, whole-volume and no SHA-256 gave the same peak within run-to-run variation; per-piece kept.
- **Limits**: no human interactive session yet (an automated visible-window session stalled and was stopped); one workstation with 128 GB RAM and an AMD integrated adapter, not an Intel laptop; touchpad gestures tested only as synthetic wheel events; all real scans are LPS and non-oblique, other orientations covered by the phantom only; pre-existing -1024..3071 HU clip of the display array.
- **Run defects**: two page builds failed on a false alarm in my own check (random base64 scan data contained the letters "niivue"); check corrected to look at code only and the pages rebuilt.
- **Artifacts**: `runs/M3_canvas_viewer_20261122/` (index + 4 case pages + staged example), `runs/V2_canvas_size_pages/`, `runs/V2_canvas_test_*`, `build_canvas_viewers.bat`.
- **Next**: usability check on the clinician reviewer's laptop (steps at the end of the report); then decide whether the grading batch uses v4.
- **Viewer v4, revision 2 (2026-11-22, later)**: added "1:1 pixels" (one image pixel per screen pixel), a "Smooth" toggle for pixel inspection, zoom read-out as "% of fit" with screen pixels per image pixel; grading form `grades_v2` now shows the lobe / side choices before an answer is given (greyed until Present or Can't tell; saved data and CSV unchanged). All eight pages rebuilt; 17 browser runs, 462 checks, all passed (`runs/V2_canvas_test_r3_*`); 176 unit tests. **Run defect**: one intermediate build had an unescaped apostrophe in the grading script, which removed the grading column from every page; caught by the browser tests, fixed, and `tests/test_viewer_v4.py` now syntax-checks every embedded script with Node. The open case pages are for usability and discussion; independent clinician grading must use the staged build (`--blinded`).
- **Viewer v4, revision 3 (2026-11-22)**: larger text for the reviewers (right-hand panel scaled to 120 % by default, "A- / A+" control 100-180 % remembered per browser, larger viewer buttons, bold edge labels, wrapping status line, scan and panel share the width equally); index page reduced to the open-pages section with larger text. Five pack pages rebuilt; 10 browser runs (Edge and Chrome, integrated graphics), 296 checks, all passed (`runs/V2_canvas_test_r4_*`). The three size-test pages in `runs/V2_canvas_size_pages/` are still revision 2.
- **Viewer v4, revision 4 (2026-11-22)**: The clinician reviewer used the v4 page for valid_832 on his laptop (Intel Core Ultra 7 155H, 16 GB): "Much better. There were no crashes." One session, one scan. He asked for the three-plane view back ("essential"), so a linked "3 planes" mode was added (axial left, coronal top right, sagittal bottom right, one cross-hair position, wheel and zoom act on the pane under the pointer; the page still opens in the single axial view). Four pack pages rebuilt, valid_832 now pre-filled with his grades (form grades_v2, lobes included); 10 browser runs (Edge and Chrome, integrated graphics, incl. the phantom), all checks passed (`runs/V2_canvas_test_r5_*`). `runs/grades_received/` holds his four CSVs; `runs/M3_canvas_viewer_20261122.zip` rebuilt. Grades so far over 4 scans (open pages, development only): firm answers agreed 34 of 39; 4 of the 5 misses came from the classifier's "not suggested". No rule changed.
- **Viewer v4, revision 5 (2026-11-22)**: adjustable split between the scan and the right-hand panel (drag the bar, 25-75 %, remembered per browser, double-click = half and half). Four pack pages rebuilt; quick test on two of them (valid_832 in Edge, valid_541 in Chrome, integrated graphics): 70 checks, all passed (`runs/V2_canvas_test_r6_*`); the other two pages were rebuilt from the same code but not re-tested. Zip rebuilt.
- **Viewer v4, revision 6 (2026-11-22)**: under the checklist the cards are now in the order radiologist report (CT-RATE) -> measurement panel -> tool's report; the measurement panel is collapsible (closed at first; opens from its heading or from the 'Open measurement panel' button). v4 pages only. Four pack pages rebuilt; quick test on valid_832 (Edge) and valid_541 (Chrome), integrated graphics: 72 checks, all passed (`runs/V2_canvas_test_r7_*`). Zip rebuilt.

## Development audit on the four graded cases (2026-11-22)
- **Status**: milestone complete; nothing frozen; waiting for clinician review. Report: `runs/D1_dev_audit_20261122/DEV_AUDIT_REPORT.md`, questions: `clinician_questions.md` in the same folder. DEVELOPMENT DATA: the four cases were graded on open pages (not blinded); no performance claim.
- **Provenance defect found and fixed**: automated browser tests had written 30 CSV files named after the clinician reviewer into the Downloads folder (pre-filled pages keep the reviewer name); in the valid_301 ones the test had added "RLL" to ground-glass. Moved to `runs/test_logs/test_generated_csv_NOT_from_reviewer/` with `PROVENANCE.json`. Tests now block downloads and use the name "AUTOMATED TEST - not a reviewer". The four genuine files in `runs/grades_received/` are unchanged (hashes recorded). The clinician reviewer recorded no location for valid_301.
- **Implemented**: `scripts/dev_audit_reconcile.py` (80-row reconciled table from run artifacts, file manifest, recomputed reader-vs-checklist table, policy replay), `scripts/dev_audit_density.py` (stage-by-stage rebuild of the consolidation / ground-glass measurement that reproduces the stored values, one-step-off ablations, comparison with HoloCT and lung_nodules outlines, overlays), `scripts/dev_audit_candidates.py` (nodule candidates and pericardial outlines with exploratory attributes; two standalone review pages). `configs/checklist_v3_5.json` + `classifier_can_establish_absence` in `lung_ct.agent.checklist` (policy `classifier_policy_v2_no_absence`; provisional, default config unchanged).
- **Tested**: 177 unit tests (new: v3_5 policy test); stage rebuild equals stored E85 / E80 / E88 on all four cases; review pages click-tested in Edge.
- **Findings**: reader vs checklist (v3_4, 72 answers): present/present 5, present/absent 4, present/uncertain 21, absent/present 0, absent/absent 27, absent/uncertain 15. All four incorrect negatives come from the COLIPRI low-score branch (two after an undecided measurement). v3_5 replay: 4 of 4 withdrawn (not detected), 15 of 27 correct negatives become unresolved; 403 answers change over the 58 runs. valid_832 consolidation: not lost by the lung outline; 7.3 of HoloCT's 10.8 mL are removed as "lung next to pleural fluid" and 2.2 mL by the posterior band (same pattern on valid_301). valid_832 ground-glass region: 1023 pieces, 976 under 0.1 mL. Nodule candidates: scan-level agreement on 4 of 4, but no candidate in RUL on valid_832 or valid_541 and the large LLL piece on valid_832 is dense lung. Pericardial outlines: 9.5-28 mL on the three reader-positive scans, 0.7 mL on the reader-negative one; size words do not follow volume.
- **Deferred, documented only**: coronary calcium, bronchiectasis / wall thickening, lymph nodes (axillary vs mediastinal), hiatal hernia, devices.
- **Next**: clinician answers and the two candidate reviews; then freeze the selected changes and evaluate on fresh patients with staged pages.
- **Development audit, revision 2 (2026-11-22)**: (1) provenance finalised: `runs/grades_received/AUTHENTICATED_EXPORTS.json` lists the four accepted grade exports with SHA-256; `dev_audit_reconcile.py`, `dev_audit_candidates.py` and `collect_grades.py` refuse any other grades file; all summaries regenerated from them (`runs/D1_dev_audit_20261122/`, `runs/E7_reviewer_grades_20261122_182303/`). The valid_301 reader-location comparison is WITHDRAWN (the "RLL" came from a test-generated file). (2) v3_5 stays provisional; on the four cases it withdraws 4 incorrect negatives and raises abstention from 36 to 55 of 72 answers. (3) Development-only tagged inventory `dense_region_inventory_dev_v1` (`scripts/dev_audit_retained_regions.py`): dense lung after vessel / airway removal and opening is split into baseline_measured / fluid_adjacent / posterior / fluid_adjacent_and_posterior; baseline reproduced and untouched; no tag is assigned to a diagnosis; valid_832: 14.6 / 1.0 / 15.8 / 26.8 mL, valid_301: 133.8 / 68.0 / 1.8 / 178.2 mL, valid_541 and valid_988: about 0. Review page `retained_regions/dense_region_review.html` (18 regions + 4 scan-level cards; what the region represents, outline adequacy and condition presence asked separately; tag hidden). (4) Nodule and pericardial review continue; statements about what a candidate is are now review hypotheses (`review_hypotheses` column), nothing deleted, no threshold derived. (5) E79 documented: 2 x the largest 3-D distance-transform value inside the predicted outline = diameter of the largest inscribed sphere; a property of the mask, not a caliper measurement, not validated; from v3_5 new runs drop the small / moderate / large wording (existing evidence and pages are not rewritten). Viewer unchanged. 177 unit tests pass.
- **Development audit, revision 3 (2026-11-22)**: (1) Policy replay reconciled: 403 answers move absent -> cannot tell over all 20 conditions x 58 runs; 397 of them are in the 18 conditions that have a report label (16 label-present + 381 label-absent) and 6 are ground-glass, which has no label. `runs/D1_dev_audit_20261122/policy_transition_table.csv`; `dev_audit_reconcile.py` asserts that all totals balance and that the only transition is absent -> cannot tell. (2) One consolidated, staged review page `review_package/development_adjudication_package.html` (`scripts/dev_audit_review_package.py`, 46 cards: dense regions, nodule candidates and missed nodules, pericardial outline): tool information, model identity, analyst hypotheses and tags, checklist and HoloCT answers and the reviewer's earlier grades are hidden until the initial assessment is locked; a separate post-reveal assessment is exported next to it; identity, outline adequacy, location and presence are separate questions; described on the page as development adjudication. Click-tested in Edge, 19 checks passed; not yet used by a person. The three separate review pages are superseded (kept). (3) Dense-region selection recorded per scan and tag (rule, regions selected / omitted, mL on cards, coverage): valid_832 baseline 71.6% and posterior 60.1% coverage (card limit reached), all other non-empty tags 81-100%; the page states that the cards do not represent every region. (4) `runs/M4_canvas_viewer_corrected_20261122/` + zip: corrected copies of the four v4 pages (`build_viewer_page.py --corrected-wording`, `lung_ct.agent.wording`): no pericardial "greatest thickness" / small-moderate-large wording, no inspiration claims; viewer script and embedded scan data verified byte-identical to M3; 4 browser runs, 144 checks passed. `runs/DISTRIBUTION_MANIFEST.json` (`scripts/write_distribution_manifest.py`) marks M1, M2, M3 and the separate review pages as superseded; nothing deleted. (5) Inspiration audit `inspiration_statement_audit.md`: the "inflation" gate uses segmented lung volume (< 3000 mL) and median density inside the outline (> -780 HU) only; no independent assessment of inspiratory adequacy exists; flagged on 16 of 58 scans, 11 of which also show large dense tissue or pleural fluid; wording made neutral in v3_5 (`neutral_wording`) and in the corrected copies; gate, limits and answers unchanged. 178 unit tests pass. Still provisional: v3_5. Next: clinician review, then freeze and evaluate on fresh patients.

## Classifier policy adopted; pleural-effusion outline checked on PleThora (2026-11-23)

- **Decision (project lead, 2026-11-23)**: `checklist_v3_5` is adopted for the next iteration: a classifier score can no longer establish absence. Classifier scores stay in the output as supporting evidence (basis text, `classifier_p`, flags) and cannot set or override any answer. `checklist_v3_4` is kept unchanged for comparison. Not validated: still to be evaluated on fresh patients with staged pages.
- **Implemented**: `configs/checklist_v3_5.json` status + `adoption` block; default of `scripts/agent_run_case.py --checklist-config` is now v3_5; the 58 existing runs rescored (`checklist_checklist_v3_5.json` next to the v3_4 file, list in `runs/test_logs/pages_58runs_with_v3_4.txt`): 403 answers absent -> cannot tell, no other change, no measured value changed, no answer decided by the classifier. Existing reviewer packs (M4) still show v3_4 answers and were not rebuilt.
- **Tested**: 179 unit tests (new: under v3_5 the answers are identical for classifier scores absent / 0.01 / 0.5 / 0.99, the score is still reported, v3_4 behaviour unchanged).
- **E14, external check of the pleural-effusion outline** (`runs/E14_plethora_pleural_effusion_20261123_112031/REPORT.md`; scripts `plethora_fetch.py`, `plethora_segment.py`, `plethora_compare.py`): PleThora expert outlines on NSCLC-Radiomics planning CT; all 78 effusion scans + 78 random no-effusion scans (seed 20261123), 4.5 GB as NIfTI; 1 scan excluded (LUNG1-095, non-uniform slices). Existing limits (present >= 20 mL, none < 5 mL), nothing tuned. Effusion scans (77): present 54 (70%), undecided 10, none 13 (17%; 6 with 0 mL outlined; largest missed 149.6 mL); by reference volume 5-20 mL 1/1/5, 20-50 mL 8/7/4, 50-150 mL 26/2/4, >= 150 mL 19/0/0. No-effusion scans (78): none 62 (79%), undecided 6, present 10 (13%, 22-261 mL). Volume: median absolute difference 41.6 mL, Spearman 0.77; median Dice 0.38 against 0.91 between experts.
- **Reading**: supports keeping an undecided measurement as "cannot tell" (tool volumes of 5-20 mL: 10 scans with effusion, 6 without) and shows that "none measured" is not proof of absence. 19 of 19 scans with a reference volume of 150 mL or more were answered present (detection only; says nothing about mask accuracy). The mL value is an estimate from a predicted mask, not a measurement.
- **Limits**: lung-cancer planning scans, not CT-RATE; reference excludes fluid overlapping the tumour; (corrected in revision 2, below) the TotalSegmentator effusion training set is published on Zenodo; overlap with this collection is unlikely but not excluded; cause of over-outlining not examined. Limits were not changed.
- **Next**: clinician review of the adjudication package; decide whether the pleural-effusion "none measured" wording should change; look at the over-outlined and missed PleThora scans before drawing conclusions about cause.

## checklist_v3_6: dense-lung evidence preservation, pleural-effusion wording (2026-11-23)

- **Implemented**: `dense_lung_inventory_v2` (`lung_ct.agent.conditions.dense_lung_inventory_rows`, rows E92 total mL, E93 by lobe, E94 spatial tags): consolidation-range lung is no longer removed for lying within 15 mm of the predicted pleural fluid or in the posterior band; both are spatial tags (`fluid_adjacent`, `posterior`, both, none) that name no diagnosis. Regions are called "dense-lung candidates". E88 renamed for new runs (no "compressed lung"). `configs/checklist_v3_6.json` (derived from v3_5; default of `agent_run_case.py`): atelectasis is no longer answered "present" from E88; consolidation decision unchanged (baseline E85, same limits; limits NOT applied to E92); `lung_ct.agent.checklist.add_dense_candidates` attaches the candidates as evidence text and flags, never a status. Pleural effusion wording by mask state (`mask_wording`): empty mask "no fluid outlined", non-zero under 5 mL "low volume outlined", neither worded as excluding an effusion, volumes called estimates from predicted masks; limits and stored statuses unchanged. Viewer builder uses neutral overlay labels when the config has the inventory. `docs/DENSE_LUNG_EVIDENCE.md`: Fleischner definitions (consolidation, atelectasis, ground-glass, air bronchogram), dependent-opacity guidance, limits of non-contrast attenuation, sources and what was not retrieved.
- **Unchanged / preserved**: historical baseline rows E85 / E88 / E90 and all limits; evidence.json, checklist files v3_2 to v3_5 and all viewer packs are not rewritten; v3_4 and v3_5 configs untouched.
- **Tested**: 183 unit tests (new `tests/test_v3_6.py`: tags partition the region and the untagged part equals the baseline region; no limit or measurement setting changed; inventory rows change no status; atelectasis not named from proximity; pleural wording by mask state; v3_5 wording unchanged).
- **Replay E15** (`runs/E15_dense_lung_inventory_replay_20261123/REPORT.md`, `scripts/dense_lung_inventory_replay.py`): 58 development runs, 56 computed (2 without a baseline consolidation measure: image too grainy), baseline E85 / E88 reproduced exactly on all 56. Baseline 535 mL in total against 2,749 mL retained; tags: none 537 mL (2,813 regions), fluid-adjacent 1,039 mL (632), posterior 154 mL (684), both 1,019 mL (150). Candidates outside the baseline on 18 runs (>= 5 mL on 9, >= 30 mL on 7). Status changes v3_5 -> v3_6: atelectasis present -> cannot tell on 5 runs (valid_102, 202, 301, 500, 72); nothing else. Pleural effusion display: 47 empty masks, 3 low-volume, 1 between limits, 5 present.
- **Open, on purpose**: consolidation "none measured" from the baseline coexists with >= 5 mL of tagged candidates on 2 runs (valid_500_a_2 67.6 mL, valid_885_a_2 6.1 mL); flagged on the item, not resolved. Ground-glass still leaves out the posterior band. Stored status "absent" for pleural effusion now means "below the reporting limit of the predicted mask".
- **Pages**: no pack shows v3_5 or v3_6 yet. M1-M4 and the adjudication package show the old policy (`runs/DISTRIBUTION_MANIFEST.json`, `answer_policy`).
- **Next**: expert assessment of the tagged regions (adjudication package, section A) or suitable annotated data before any presence / absence rule uses the inventory; decide whether to build a v3_6 pack.

## E14 revision 2: PleThora benchmark verified, failure analysis (2026-11-23)

- **Report**: `runs/E14_plethora_pleural_effusion_20261123_112031/REPORT.md` (revision 2; revision 1 kept as `REPORT_rev1.md`), `failure_analysis.csv` (43 inspected scans), `failure_groups.csv` (all 155), `overlays/` (43 PNG), `data/raw/plethora/PROVENANCE.md`. Scripts `plethora_failure_analysis.py`, `plethora_failure_table.py`, `plethora_fetch_gtv.py`. No limit changed.
- **Verification**: (a) "no effusion" (code 0) is NOT a confirmed absence: the PleThora paper says a medical student identified the 78 effusion scans; no physician read of the other 324 is described. Tool "present" on those scans = false-positive candidate. (b) Reference outlines exclude effusion overlapping the tumour by design; trace fluid was not handled by a fixed rule (stated by the authors); which reviewer file is the official one is not documented (spreadsheet volume matches the first file on 47 scans, third on 15, second on 5, none on 10). (c) Geometry: all 155 scans on one grid after reorientation only; right / left thoracic volumes match the spreadsheet within 3% on 154; LUNG1-053 stays in, LUNG1-095 stays out. (d) TotalSegmentator 2.18.0, task 315 `Dataset315_thoraxCT`, 3d_fullres fold 0, checkpoint hash recorded; the crop model's hash is not. (e) Metrics written out as coded. An earlier statement that the training data are unpublished was wrong: the 600-case set is on Zenodo (record 20272295, University Hospital Basel author); neither it nor the two cited papers mentions NSCLC-Radiomics / TCIA / PleThora, so overlap is unlikely but not excluded (not downloaded and compared).
- **Detection (kept separate from mask accuracy)**: effusion scans 54 present / 10 undecided / 13 none; no-effusion-coded scans 10 / 6 / 62; 19 of 19 at >= 150 mL answered present. **Mask and volume**: median Dice 0.38 (0.91 between reviewers); only 15 of the 54 "present" answers have Dice >= 0.70.
- **Failure groups** (rules fixed before viewing overlays): effusion scans: empty mask 6, threshold abstention 10, undersegmentation 17, over-outlining 16, good-overlap control 15, other 13. No-effusion-coded: empty 45, low non-zero 17, abstention 6, false-positive candidate 10.
- **Observations from overlays (analyst, not a radiologist)**: all 6 empty-mask positives are thin posterior crescents over many slices (the 150 mL case spans 61 slices); under-outlining mostly on thin layers; on 6 of 10 false-positive candidates 96-100% of the tool mask lies inside the tumour outline, 3 more sit on soft tissue next to it, and LUNG1-138 looks like a real effusion on a scan coded "no effusion"; over-outlining is mostly fluid plus tumour.
- **Recommendation**: keep both limits (moving them fixes none of the causes); wording by mask state (done in v3_6); next, describe the predicted mask as a candidate (mean attenuation, crescent vs compact shape, contact with other outlined structures) with no cut-off, and evaluate such descriptors only on scans not yet inspected; radiologist questions: LUNG1-138, the six empty-mask scans, LUNG1-128.
- **Not verified**: institution and composition of the 600 training cases; which reviewer file is primary; whether thin layers are fluid or pleural thickening; whether tumour confusion matters on CT-RATE. A left / right error in the first pass of the failure analysis was caught by the spreadsheet cross-check and corrected before the report was written.

## E16: root cause of the six empty-mask misses; checklist_v3_7 (2026-11-23)

- **Report**: `runs/E16_empty_mask_root_cause_20261123/REPORT.md` (stage table, density table, which errors need a model change, overlays for ten scans).
- **Traced** (six misses + four controls, from source DICOM to saved mask): DICOM re-downloaded, 0 voxels differ from our NIfTI, HU / spacing / position correct; 100% of each reference inside the actual crop box (nearest face 15-24 mm); saved mask identical to the argmax of the exported probabilities; mirroring changes nothing. **The loss is in the network**: highest P(pleural effusion) inside the reference 0.03-0.09% on three scans, 2%, 17%, 43% on the others; 0 mL even at a 10% level.
- **Two causes** (hypotheses, analyst not a radiologist): (a) LUNG1-170, -013, -249: the inner part of the reference is 59-76% fat-range density (mean -60 to -105 HU); (b) LUNG1-113, -195, -299: layers of 2.8-4.4 mm. Water-density test (edited CT in memory only): -170, -249, -195 then outlined 44%, 44%, 36%; -013, -113, -299 still 2-6%.
- **Training set** (Zenodo 20272295; labels read by byte range, 88 MB on disk; 40 images read to memory only): 600 cases at 1.5 mm, 216 with a pleural label, median 179 mL, median inscribed sphere 21.8 mm; typical labels at water density (inner mean 5.5 HU, fat-range share 3%). Overlap with NSCLC-Radiomics cannot be screened by grid (all resampled); still not excluded.
- **Needs a model change**: thin layers (empty masks and the 17 under-outlined scans); soft tissue outlined as fluid. **Decision policy is enough**: "none" on effusion scans and the 5-20 mL band (both done). **Needs a radiologist first**: the three fat-density rims, LUNG1-138.
- **checklist_v3_7** (`configs/checklist_v3_7.json`, default of `agent_run_case.py`; v3_6 and earlier untouched): a predicted pleural-fluid mask below the reporting limit is "cannot tell" (display "cannot tell (no fluid outlined)" / "(low volume outlined)"), never "absent"; `historical_rule_answer` keeps the answer of the unchanged 20 / 5 mL rule; limits named provisional reporting limits, not validated diagnostic boundaries (`limits_note`). 58 development runs rescored (`checklist_checklist_v3_7.json`, additive): only change pleural effusion absent -> cannot tell on 51 runs.
- **Tested**: 187 unit tests (new `tests/test_v3_7.py`: no limit or measurement changed, only pleural effusion differs from v3_6, historical answer kept, report never lists it as not detected).
- **Provenance completed**: hash of the 6 mm cropping model (`Dataset298`, fold 0, `checkpoint_final.pth`) SHA-256 `3489323a4b759506ab46b489eb78b72142f2143aea6e6d828604d754bd7a9270`; actual crop boxes saved in `data/derived/plethora_trace/`.
- **Not done on purpose**: no training, no new model, no shape / density / tumour filter, no limit moved, no viewer pack rebuilt (no page shows v3_5, v3_6 or v3_7).
- **Open**: whether "present" from a predicted mask alone stays a definitive answer; which reviewer file is primary; overlap with the training set; whether the pattern occurs on CT-RATE.
- **Next**: radiologist questions (fat-density rims on -170 / -013 / -249; thin layers -113 / -195 / -299; LUNG1-138); only then compare another or a fine-tuned model on scans not yet inspected.

## E16 revision 2: audit completed; checklist_v3_8; comparison plan (2026-11-23)

- **Report**: `runs/E16_empty_mask_root_cause_20261123/REPORT.md` (revision 2; revision 1 kept as `REPORT_rev1.md`). Plan: `docs/PLEURAL_MODEL_COMPARISON_PLAN.md`.
- **Corrections to revision 1 (and to the section above)**: (a) "0 mL even at a 10% level" was a rounding artefact: exact counts at P >= 0.1 on the crop grid are 4 voxels (0.011444 mL) on LUNG1-195 and 16 voxels (0.045776 mL) on LUNG1-299, 0 on the other four. (b) Postprocessing is not loss-free: on LUNG1-299 the network outlined 9 voxels (0.030375 mL) on the working grid and resampling back left 0. (c) Thinness and intensity mismatch are hypotheses, not causes; the water-density intervention is exploratory and supports no decision.
- **Exact stage audit** (`scripts/plethora_stage_audit.py`, `runs/E16_empty_mask_root_cause_20261123/stage_audit/`): network re-run in-process on the crop TotalSegmentator used; rebuilt mask identical to the pipeline file on 8 of 10 scans, 1 voxel different on LUNG1-016 and LUNG1-285 (run-to-run GPU variation). Class mapping 0 background, 1 lung, 2 pleural effusion, 3 pericardial effusion.
- **Network input**: source 0.977 x 0.977 mm, 3.0 mm slices. nnU-Net resamples in plane to 1.5 mm with cubic interpolation (coarser than the source) and along the slice axis to 1.5 mm by nearest neighbour (every 3 mm slice used twice; 3.0 / 0.977 = 3.07 is above nnU-Net's anisotropy limit of 3). The working grid adds no through-plane detail. Output: logits linear in plane, nearest neighbour along the slice axis.
- **Official PleThora reference resolved**: `*_effusion_first_reviewer.nii.gz` of `PleThora-Effusions-June-2020.zip` (TCIA version 2, 2020-06-26) = primary reviewer Rad4, re-reviewed and corrected; second = Rad3, third = RO1 (matched through the spreadsheet's Dice columns). The April 2020 spreadsheet was not updated and is stale for the 35 first files dated 2020-05-28; spreadsheet volumes are not the reference. Alternative reviewer files are reported separately.
- **Alignment by image content** (`scripts/plethora_alignment_check.py`, 172 files): bone-or-air share of the outline as stored 0.9% (first, median) against 41.7% when mirrored left-right; best of 147 positions within 1 voxel of the stored one on 74 of 77 first files; six misses rank 1 to 7. Oddity: LUNG1-143 (front-back mirror scores lower).
- **Training-density provenance**: HU statements rest on 40 of 216 labelled training images (20 thinnest labels + every 10th of the rest), read by byte range from Zenodo 20272295 into memory, CRC-32 checked against the archive directory, listed in `training_labels/training_density_sample_provenance.csv`. Images are 1.5 mm resampled copies without acquisition information. Archive MD5 not verified (no full download). Label files alone support geometry only.
- **checklist_v3_8** (`configs/checklist_v3_8.json`, default of `agent_run_case.py`; v3_7 and earlier untouched): a positive answer from the predicted mask alone is headlined "Pleural effusion suggested by segmentation" (display "suggested by segmentation"); `provisional_threshold_result` stores the answer of the unchanged 20 / 5 mL rule separately. Stored status values unchanged. 58 development runs re-scored (`checklist_checklist_v3_8.json`, additive): no status change against v3_7, 6 runs carry the new headline.
- **Tested**: 190 unit tests (new `tests/test_v3_8.py`).
- **Comparison plan (not started)**: current model against one alternative, `Task307_pleuralfinal` (Zenodo 21918112, nnU-Net v1, 2.36 GB, CC BY 4.0), on the 155 PleThora and 58 CT-RATE development scans. No TotalSegmentator inference option addresses a failure found. Nothing downloaded.
- **Blocked / open**: (1) radiologist reading of the six regions and LUNG1-138; (2) cause not isolated (hypotheses A and B); (3) training-archive MD5 unverified, raw data at 20 of 20 GB; (4) training HU scale and acquisition undocumented; (5) training overlap with NSCLC-Radiomics not excluded; (6) PleThora: LUNG1-143, 16 third files against 15 stated, unconfirmed "no effusion" codes; (7) 1-voxel run-to-run differences; (8) alternative model: nnU-Net v1 compatibility unverified, pickle checkpoints, which 34 PleThora scans its paper used, model storage 56 GB against a 30 GB budget; (9) no viewer pack shows v3_5 to v3_8; (10) not checked on CT-RATE.
- **Next**: approval for section 4 of the plan (compatibility, provenance, storage) before any download; radiologist questions.

## E16 closed; E17 comparison preparation (read-only) complete (2026-11-23)

- **E16 closed** as a development failure audit with unresolved causal hypotheses (`runs/E16_empty_mask_root_cause_20261123/REPORT.md`, revision 3). Wording: the six scans are misses against the published expert reference; additional adjudication may clarify reference disagreements. All six stay in the primary reference-based analysis (no independently established data problem). Open items are split into actionable prerequisites (8a) and documented limitations (8b). The training archive will not be downloaded for its MD5 alone.
- **LUNG1-143 resolved without touching its reference**: no geometry problem. The bone-or-air alignment score is blind to a copy that lands in soft tissue; the front-back mirrored copy falls into the anterior chest wall. Stored outline: 59.7% inside the PleThora thoracic-cavity outline (mirrored 0.2%), mean -17 HU, Dice 0.448 with the model's mask. Sensitivity (E14, first-reviewer reference): with it 77 scans, 54 / 10 / 13, median Dice 0.3763; without it 76, 53 / 10 / 13, 0.3749. No case is recorded as unresolved.
- **v3_8 meaning through report generation, CSV export and scoring** (`lung_ct.agent.checklist.assessment_category`, `export_row`, `scripts/export_checklist_csv.py`; changed: `build_checklist_prompt`, `validate_checklist_report`, `scripts/score_checklist.py`, `scripts/agent_render_checklist.py`): "suggested" is its own category, never scored or written as a confirmed finding; the provisional threshold result is exported in the column `provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT` and is never scored; the report writer is told to use the fixed sentence and the validator flags a plain "pleural effusion" finding or "no pleural effusion". `scripts/dev_audit_reconcile.py` (written for v3_2 to v3_5) now refuses a checklist with suggested items instead of counting them as present. Found and fixed here: before this change the scorer and the checklist page counted a suggested effusion as a confirmed "present". 58 development runs: pleural effusion 6 suggested (all 6 labelled in the CT-RATE report), 52 cannot tell, 0 absent (`runs/E6_checklist_scores_v3_8_dev58_20261123_141447/`).
- **Alternative model inspected, nothing downloaded** (`runs/E17_pleural_model_comparison_prep_20261123/`, `scripts/pleural_armB_inspect.py`): Task307_pleuralfinal is nnU-Net v1, 3d_fullres, 2.0 mm working grid (coarser than the current 1.5 mm), five folds, 224 training scans from University Hospital Basel (Siemens, 1.0 mm soft kernel), two classes. Paper: external PleThora evaluation on 34 scans (median Dice 0.71, mean volume 383 mL); the 34 are not listed. This is prior evaluation exposure, not training overlap; no source indicates PleThora scans in its training.
- **Pilot prepared, not run**: 14 scans (six misses, four traced controls, four graded CT-RATE cases); manifests, `scripts/pleural_armB_run.py` (not yet run), `scripts/pleural_compare_metrics.py` (run for the current model: 0 voxels on the six, Dice 0 with empty predictions included). Rules fixed: empty predictions included; scans without an expert outline are not confirmed negatives.
- **Tested**: 191 unit tests.
- **Blocked on the project lead**: storage exception (plan, section 4c): model weights are at 56 GB against a 30 GB limit; requested +3.61 GB at peak, +1.25 GB permanently, plus one environment of about 6 GB on D:. No new raw data (20 of 20 GB).
- **Risks before the pilot**: nnU-Net v1 on Windows unverified; checkpoints are pickle files; PyTorch must be pinned below 2.6; the paper's input (1.0 mm B30f) differs from our scans.
- **Next, after approval**: download and verify, build `environments/pleural_v1`, synthetic go / no-go, pilot, then the fixed development comparison (155 PleThora + 58 CT-RATE).

- **Viewer pack M5 (current policy, built, not sent)**: `runs/M5_policy_v3_8_20261123/` (four graded development cases, `checklist_v3_8`), built with `scripts/build_policy_pack.py`, checked with `scripts/check_policy_pack.py` (102 of 102 checks, headless Edge from file://, automated test exports marked as not from a reviewer). Every page names the policy and pack and says it is an open development page. Pleural effusion shown: valid_301 "suggested by segmentation", valid_832 "cannot tell", valid_541 and valid_988 "cannot tell (no fluid outlined)"; the provisional threshold result sits apart, labelled as not the current answer. Earlier grades appear read-only with the policy they were made on (v3_2, v3_3, v3_4) and are not copied into the new form; the form's storage key and CSV carry the policy and pack. M1-M4, the adjudication package and all received grades are unchanged (hash lists in `runs/test_logs/M5_hashes_*.txt`; M4 zip re-verified after the rebuild). `runs/DISTRIBUTION_MANIFEST.json` names the policy of every pack and review.
- **v3_8 files were rewritten the same day (documented revision 2)**: two stored strings reworded (lobe-uncertainty reason no longer says "pleural effusion present (compressed lung)"; basis of a suggested answer no longer says "present >= 20, absent < 5"). All 58 `checklist_checklist_v3_8.json` files were overwritten; 6 changed content (the six suggested runs), 52 were written back with identical bytes. Revision 1 was not under version control (runs/ and the config are untracked). Record: `runs/E17_pleural_model_comparison_prep_20261123/v3_8_revision_record/` (SHA-256 of every current file, reconstructed revision-1 files; the reconstruction matches all 4 revision-1 hashes that had been recorded before the rewrite, the other 54 are reconstructions, not preserved copies). `configs/checklist_v3_8.json` was also edited (one key, one change note); its first revision was not preserved. M5 was rebuilt from revision 2. No status changed. Earlier statements that the M5 hash check covered "checklist v3_2-v3_8 files unchanged" apply to the time before this rewrite only.
- **Not done for M5**: full interaction and memory test, no human look at the pages, `collect_grades.py` not tested with the new CSV columns, nothing claimed about the reviewer's laptop.

## E17 pleural-effusion pipeline comparison closed (2026-11-23)

Report (revision 2): `runs/E17_pleural_model_comparison_prep_20261123/REPORT.md`. Plain summary of E14 to E17:
`docs/PLEURAL_EFFUSION_LEARNINGS.md`. Development data; not a validation result.

**Decision**
- **Current model retained.** The comparison does not support replacing the current pipeline.
- **No clear replacement benefit demonstrated.** This is not a finding that the two models are equivalent, and it
  does not validate the current model.
- **The six target misses remain unresolved** (LUNG1-013, -113, -170, -195, -249, -299): missed by both pipelines,
  cause not established.
- **No further model runs or training are planned without a new experiment brief.**

**Implemented**
- Two inference pipelines as implemented here, on a fixed set of 213 scans (77 PleThora with expert outline, 78 PleThora
  coded no effusion, 58 CT-RATE). Current: TotalSegmentator effusion task, chest crop (lobes + 50 mm). Alternative:
  `Task307_pleuralfinal` (nnU-Net v1, five folds, released settings), whole scan as stored, single-process driver on
  Windows, authors' unpublished downsampling step not reproduced.
- Scripts: `pleural_armB_run.py` (`--lung-guard stop|warn`), `pleural_armB_qc.py`, `pleural_armB_class_review.py`,
  `pleural_armB_outside_chest.py` (descriptive), `pleural_compare_metrics.py`, `pleural_compare_report.py`,
  `pleural_compare_freeze.py`.
- Freeze record `runs/E17_.../freeze/`: case manifest with hashes of every CT, prediction and reference file; model
  and config identity; preprocessing (published versus ours); three environment listings; count verification; hashes
  of metrics and report.

**Tested**
- Primary analysis, all 77 expert-outlined scans, empty predictions = Dice 0: median Dice 0.376 current / 0.414
  alternative; empty 6 / 5; Dice >= 0.7 on 16 / 19; median absolute volume error 47.7 / 42.2 mL; mean 77.8 / 94.7 mL;
  6 improved, 4 worsened, 67 similar (0.10 Dice step fixed in advance). All summary counts recomputed from the
  per-case table: no mismatch.
- Labelled sensitivity analysis without LUNG1-104 (89 cm scan): median Dice 0.377 / 0.414, mean absolute volume error
  78.8 / 75.1 mL. LUNG1-104 stays in the primary analysis.
- "5 empty instead of 6" is not recovery: 3 targets still empty, 3 targets non-empty but off the fluid (best Dice
  0.003), 2 other scans newly empty.
- Grid, orientation and volume checks pass on 213 of 213. Output-class mapping verified on all scans (label 1 agrees
  with the current pipeline's lung label, median Dice 0.983; label 2 at most 0.3% air-like). The heuristic class check
  gave 21 warnings, each reviewed: 20 diseased lung correctly outlined, 1 real error (LUNG1-104). The heuristic missed
  5 further lung-label errors (LUNG1-038, -073, -133, -362, -128).
- Alternative labels tissue outside the chest on 5 of 7 scans of 50 cm or more; reaches 20 mL on 15 of 78 scans coded
  no effusion (current: 10).
- valid_832: 18.4 mL current, 20.2 mL alternative; a 1.7 mL difference across the provisional 20 mL limit, not a
  confirmed improvement.
- Unit tests: 202 pass.

**Blocked / open**
- No verified-negative scans: specificity cannot be stated. E16 hypotheses (intensity mismatch, thinness) unseparated.
- M5 viewer pack built and machine-checked, not sent; no person has looked at the pages.
- v3_8 revision 1: original bytes not preserved. `v3_8_revision_record/rev1_reconstructed/` holds reconstructions
  (marked by a README in that folder); 4 of 58 match hashes recorded before the rewrite, 54 cannot be verified.
- `D:\t`: empty folder made by this experiment as a short temp path for a package build. Not deleted: the earlier
  delete command was refused by the command tool's safety check on that path; not retried or worked around.

**Next actions**
- None on models without a new experiment brief. Candidate briefs are listed at the end of
  `docs/PLEURAL_EFFUSION_LEARNINGS.md`.

## Pack M6: v3_8 pages with the reviewer's earlier grades filled in (2026-11-23)

Asked for by the project owner. Folder `runs/M6_policy_v3_8_prefilled_20261123/` (4 pages + index). M5 is unchanged
(hashes compared before and after) and stays the pack with an empty form.

**Implemented**
- `build_policy_pack.py --prefill-earlier-grades --reference-columns` (both off by default; M5 rebuilds as before).
- Form filled from the authenticated earlier grade file of each case. The page names the policy those grades were made
  on (v3_2, v3_3 or v3_4) and asks the reviewer to check each one against this page.
- Every exported row has `answer_origin` (`carried_over_from_earlier_grade_unchanged`, `changed_on_this_page`,
  `entered_on_this_page`), `carried_from_policy`, `carried_from_saved_utc`.
- One always-visible reference column: CT-RATE report label ("not mentioned" is not "absent"; derived from the report
  text, not from image annotation). The comparison tag uses the current assessment, so "suggested by segmentation" and
  "cannot tell" are never counted as agreement. A separate "reviewer's earlier grade" column was built first and removed
  at the project owner's request; the earlier grades remain in the form and in the collapsed read-only card.

**Tested**
- `scripts/check_prefilled_pack.py`, Edge: 40 of 40 checks (form equals earlier file, columns visible, origin columns
  correct untouched and after one change, own storage key). Output `runs/V4_M6_viewer_check_20261123/`.
- Import: an M6 export is read as its own group (policy v3_8, pack M6) with no problems reported. Unit tests 202 pass.

**Limits**
- A row marked carried over unchanged is the earlier grade, made against another page's wording. It is not a new
  assessment of v3_8 and must not be counted as one. Showing earlier answers can also nudge the reviewer to keep them.
- Development pages, open. Four cases. Not sent; no person has looked at the pages. Chrome not run for M6.

## Policy v3_9: atelectasis rule restored; pack M7 (2026-11-23)

Project-owner decision: the v3_6 withdrawal of the atelectasis rule is reversed. `configs/checklist_v3_9.json` is the
default of `scripts/agent_run_case.py`. v3_8 files and packs M5 / M6 are unchanged (hashes compared).

**Implemented**
- Atelectasis is "present" when dense lung within 15 mm of the predicted pleural fluid (E88) is 30 mL or more, as on
  the pages up to v3_5. Below 30 mL: "cannot tell" (the rule never says "absent").
- The dense-lung candidate paragraph is no longer appended to the atelectasis answer. Nothing else changed.
- Pack `runs/M7_policy_v3_9_prefilled_20261123/`: v3_9 pages, earlier grades filled in, CT-RATE report-label column.

**Tested**
- 58 development scans re-scored (`runs/E18_atelectasis_rule_restored_v3_9_20261123/rescore_v3_9.json`): 5 answers
  change, all atelectasis "cannot tell" -> "present" (valid_102, 202, 301, 500, 72); no other answer or text changes.
- Against CT-RATE report labels: 9 scans labelled atelectasis; rule present on 3 of them and on 2 unlabelled scans
  (valid_301, graded present by the reviewer; valid_72); 53 "cannot tell".
- `tests/test_v3_9.py` (5 tests); all unit tests 207 pass. M7 browser check (Edge): 40 of 40.

**Known limits accepted with this decision**
- The answer depends on the predicted pleural-fluid mask, which the same page only calls "suggested by segmentation";
  a mask on tumour gives a false atelectasis. Tumour or consolidation next to fluid is counted as atelectasis.
- Only atelectasis next to fluid can be found (6 of 9 labelled scans stay "cannot tell").
- Decided after seeing the reviewer's grades on the four open development cases: they are not a test of this change.
- Not validated; development data only. Not committed; M7 not sent.

## Policy v3_10: lung nodule from segmentation candidates, four graded cases; pack M8 (2026-11-23)

Project-owner decision. `configs/checklist_v3_10.json` = v3_9 plus the lung nodule answer. Scope: the four graded
development cases only. The default for new runs stays v3_9, because the agent run does not yet compute the nodule rows.
Packs M5, M6, M7 unchanged (M7 hashes compared; one M7 page rebuilt byte-identically with the changed code).

**Implemented**
- `scripts/nodule_evidence_rows.py`: rows E97 (candidate count) and E98 (lobe and equivalent-sphere diameter of each
  candidate) from the stored TotalSegmentator `lung_nodules` mask. No new inference; evidence.json not rewritten.
- One or more candidates: "Lung nodule suggested by segmentation" (stored status present, current assessment suggested).
  No candidate: "cannot tell (no candidate outlined)". Never "present", never "absent". No size limit, no filter.
- Viewer: outline "Nodule candidates (segmentation model)"; "Show in scan" on the row jumps to the largest candidate.
- Pack `runs/M8_policy_v3_10_prefilled_20261123/` (earlier grades filled in, CT-RATE report-label column).

**Tested**
- Four cases: 301 six candidates (3.2 to 29.7 mm), 541 three (2.3 to 3.1 mm), 832 seven (2.0 to 18.9 mm), 988 none.
  Three answers move "cannot tell" -> suggested; no other answer or text changes.
- `tests/test_v3_10.py` (5 tests); all unit tests 212 pass. M8 browser check (Edge) 40 of 40; row, outline and jump
  inspected on 301 and 832 (`runs/V6_M8_viewer_check_20261123/`).

**Limits**
- Candidates are unvalidated model output. On 301 the three largest (30, 14, 12 mm) exceed the 8 mm the report gives
  for the largest nodule, so they are probably consolidation or mass. On the 58 scans (E11) 13 of 35 scans without a
  nodule label also get a candidate. Sizes are equivalent-sphere diameters of the outline, not nodule diameters.
- Chosen after seeing the reviewer's grades on these four open cases: they are not a test of the change.
- The other 54 development scans are not re-scored with v3_10. Not committed; M8 not sent.

## Mask-derived airway calibre estimates for bronchiectasis research: code written, NOT RUN (2026-11-23)

Description: `docs/AIRWAY_BRONCHIECTASIS_MEASURES.md`. Nothing below has been executed: no test, no measurement run.
Only a syntax check (py_compile) was done. The Phase 5 baseline code and `runs/P5_airway_*` are unchanged.

**Implemented (new files only)**
- `src/lung_ct/airway/calibre.py`: per tree branch, lumen area (mm2), area-equivalent diameter (mm), tapering along the
  branch (% per mm) and against the parent (%), lobe, quality flags; case-level quality flags; case features.
- `scripts/airway_measure.py` -> `runs/P6_airway_measures_*` (runs the unchanged tree stage, then the measurements).
- `src/lung_ct/airway/cohort.py`, `scripts/airway_group_compare.py`: patient-level splits written once, descriptive
  group comparison, test split locked behind a frozen plan. No classifier, no threshold.
- `tests/test_airway_calibre.py`: synthetic tubes and split rules (12 tests, not run).

**Tested**: nothing.

**Wording rule**: these values are mask-derived estimates, not validated airway-diameter measurements from the CT.
The current data (14 report-derived positive labels among 85 scans; 4 radiologist-reviewed scans) can check pipeline
behaviour and cannot establish bronchiectasis detection performance.

**Blocked**
- AirQuant needs MATLAB (unavailable); it stays a methods reference and is not integrated.
- No dataset with airway reference annotations is in the project: measurement agreement cannot be assessed.
- No expert-labelled bronchiectasis cohort: local labels are CT-RATE report-derived (14 of 85 scans).
- Broncho-arterial ratio: out of scope (no reliable airway-artery pairing).

**Next actions (project owner)**
- Run the commands in section 11 of the document, starting with the synthetic-tube tests; tolerances in those tests are
  first guesses and may need review against the measured values.

## Mask-derived airway estimates: first run on the four graded cases (2026-11-23)

Runs: `runs/P6_airway_measures_3cases_20261123_211533` (valid_301, 541, 988), `runs/P6_airway_measures_1cases_20261123_211613` (valid_832).
Pipeline check only. Four scans cannot show anything about bronchiectasis.

**Tested**
- `tests/test_airway_calibre.py` 13 pass, `tests/test_airway_tree.py` 15 pass (baseline unchanged). Tolerances were not changed
  after the first run. One fixture was moved off the voxel lattice: for a perfectly symmetric synthetic tube at 0.8 mm the
  baseline skeletoniser (scikit-image Lee thinning) returned an empty skeleton and the tree stage failed with an unclear error.
  Baseline code not changed; recorded as a known weakness of the tree stage for symmetric synthetic input.
- Synthetic tubes: 6 mm tube estimated 5.87 to 6.08 mm at 1 mm and at 0.7 x 0.7 x 1.5 mm voxels; cone with true narrowing
  1.53 % per mm estimated 1.49.
- Four cases ran (11 to 18 s each): 58 to 311 tree branches; 55 to 272 with an estimate.

**Findings (pipeline behaviour)**
- valid_832: `broken_airway_mask` (69 pieces; 36.0 mL outside the largest piece, 21.9 mL kept). The kept piece is the right
  lung tree only, without the trachea. Its generations and estimates are not usable. The flag caught it.
- valid_301: only 58 branches, to generation 7 (lungs with effusion and consolidation): little airway was segmented.
- valid_541 and valid_988: complete-looking trees (311 and 260 branches); trachea estimate 12.3 and 12.8 mm.
- 28% to 74% of branches are `small_relative_to_voxel`; on valid_541 (1.25 mm slices) 230 of 311.
- Tapering along a branch is noisy: most branches have fewer than three 3 mm windows.
- No pattern separates the 2 reviewer-present from the 2 reviewer-absent scans, and none was expected from four scans.

**Not done**: visual overlay of the estimates on CT; parameter sensitivity; comparison with any reference; group comparison
script (not run: four scans). No P5 baseline exists for these CT-RATE scans, so the P5-versus-P6 equality check was not possible here.

## CT-CLIP zero-shot on the four graded cases (hiatal hernia question) (2026-11-23)

Run `runs/E21_ctclip_zero_shot_4cases_20261123_215255`. Exploratory; four scans; scores are uncalibrated and no threshold was applied.

**Implemented**
- `scripts/ctclip_fetch.py`, `fetch_ctclip.bat` (gated checkpoint; token from the environment only), `scripts/run_ctclip.py`
  (published preprocessing and prompts of the CT-CLIP repository, commit a2a155c; one scan at a time).
- `models/ct_clip` 2.1 GB (CT-CLIP_v2.pt 1.77 GB, text encoder, code), `environments/ct_clip` 4.9 GB (torch 2.5.1+cu124).
- Array layout checked against one original CT-RATE file (identical; local file already in HU): `models/ct_clip/orientation_check_valid_301.json`.
  The raw copy was deleted after the check.

**Tested**
- 4 of 4 scans ran (3 to 5 s, 3.9 GB GPU). Checkpoint loaded with 0 missing keys and 1 unexpected key (`position_ids`, a buffer
  newer transformers versions no longer store).
- Hiatal hernia score: valid_832 0.79, valid_301 0.54, valid_541 0.48, valid_988 0.23. Reviewer: present on 832 and 301, absent on
  541 and 988. Same order as the reviewer, but 301 and 541 are 0.06 apart.
- Scores move together per scan: valid_832 is 0.61 to 0.90 on all 18 conditions and valid_988 is 0.11 to 0.38 on all 18. The hiatal
  hernia score is not specific to hiatal hernia on these scans.
- COLIPRI (current classifier) on the same scans: 0.06, 0.005, 0.94, 0.004 (301, 541, 832, 988).

**Limits**: four scans; development cases; CT-RATE validation split (not in CT-CLIP's training); fine-tuned CT-CLIP variants not tried.
Not wired into the checklist. Not committed.

## CT-CLIP fine-tuned variants; all three variants on the 85 local scans (pilot) (2026-11-23)

Runs: `runs/E21_ctclip_{base,vocabfine,classfine}_85cases_*`, four-scan runs `runs/E21_ctclip_{zero_shot,vocabfine,classfine}_4cases_*`,
table `runs/E21_ctclip_85cases_comparison.csv`. Pilot only: 85 scans from 70 patients, CT-RATE validation split, development data.

**Implemented**: `scripts/run_ctclip.py --variant base|vocabfine|classfine` (published preprocessing; ClassFine = image encoder +
linear layer, sigmoid, as in the repository's `ct_lipro_inference.py`). Checkpoints 1.77 GB each in `models/ct_clip`.

**Tested**
- 85 of 85 scans ran for each variant; all checkpoints loaded with 0 missing keys.
- Runner check against report labels (rank statistic, "not mentioned" as comparison group, so not accuracy): 0.9 or more for
  cardiomegaly, arterial and coronary calcification, pericardial and pleural effusion, medical material in all variants. The
  runner reproduces sensible model behaviour.
- Shared score component (not shown to measure abnormality; exact definition in the frozen report): share of raw-score variance in one component 0.89 zero-shot, 0.77 VocabFine, 0.97 ClassFine (all 85 scans).
  The hiatal hernia score correlates 0.95 / 0.77 / 0.93 (Spearman) with the mean of the other 17 scores.
- Hiatal hernia: 4 scans carry the report label (3 patients) and 4 have a reader grade; no accuracy figure is possible.
  valid_301 (label yes, reader present): percentile among the 85 scans 68 zero-shot, 46 VocabFine, 80 ClassFine; COLIPRI 0.06.
  valid_1057 (both reconstructions, label yes): 86th percentile or higher in every model; COLIPRI 0.97 and 0.99.
- COLIPRI's rank statistic against report labels is equal or higher than the CT-CLIP variants for most conditions on these scans.

**Training / selection exposure**: the scans are from the CT-RATE validation split. The repository's fine-tuning scripts train on the
data folder given and save every epoch without a validation step; the base trainer evaluates on the validation split during
training. Gradient training on these scans is not indicated; use of the split for monitoring or checkpoint selection cannot be
excluded, and it is the split the authors report results on. Not an independent test.

**Conclusion (wording revised 2026-11-24)**: these pilot results do not demonstrate sufficient condition-specific benefit to justify adopting a
CT-CLIP variant for hiatal hernia. (An earlier version of this entry tied the conclusion to one case, valid_301; that criterion is withdrawn.)

## CT-CLIP pilot frozen; hiatal hernia reader study proposed (2026-11-24)

Frozen report: `runs/E21_ctclip_pilot_20261124/REPORT.md` (hashes in `SHA256SUMS.json`). Proposal: `docs/HIATAL_HERNIA_READER_STUDY_PROPOSAL.md`.

**Decision**: development pilot only. COLIPRI stays. No hiatal-hernia threshold, no score tuning, no ranking rule.
These pilot results do not demonstrate sufficient condition-specific benefit to justify adopting a CT-CLIP variant for hiatal hernia.

**Implemented**: `scripts/ctclip_runner_check.py` (bounded implementation check), `scripts/ctclip_pilot_analysis.py` (one reconstruction per
patient), `scripts/hh_reader_study_sample.py` (draft sampling table, `runs/E22_hh_reader_feasibility_DRAFT/`).

**Tested (one scan, valid_301_a_2)**
- Checkpoint identity: size and SHA-256 of all three files reproduce the repository's Git-LFS pointer ids.
- Loading: every model tensor equals the checkpoint tensor; one unused checkpoint entry (`position_ids`); the authors' strict loader fails on it
  with transformers 4.44.2, so loading is non-strict (recorded deviation).
- Condition order equals the authors' lists and the label-file columns. ClassFine order rests on code, not data.
- Normalisation: softmax over the two prompts of one condition only; sigmoid per output for ClassFine.
- Authors' preprocessing on the original CT-RATE file and authors' softmax give tensors and scores identical to our runner (difference 0).
- Not reproduced: the authors' inference class / data loader and their software versions.

**Descriptive analysis, 70 patients (one reconstruction each)**: share of the first component, raw / logit / standardised / standardised logit:
zero-shot 0.90 / 0.89 / 0.88 / 0.88; VocabFine 0.79 / 0.81 / 0.66 / 0.66; ClassFine 0.97 / 0.98 / 0.86 / 0.86. No CT-CLIP variant is consistently
above COLIPRI against report labels. Hiatal hernia: 3 labelled patients; nothing estimated.

**Proposed, not started**: reader feasibility study, 30 of 66 local patients (strata C 2, H 12, M 8 of 20, L 8 of 32; probabilities 1, 1, 0.40, 0.25),
readers blind to scores, report and labels; present / absent / uncertain plus assessability; two readers if available. 1,234 validation patients
reserved (not downloaded, scored or viewed) for a later evaluation. No calibration estimate promised.

**Blocked / needs decision**: sample size, readers, form fields (the clinician reviewer). Nothing committed.


## Repository housekeeping before first push of this branch (2026-11-24)

- **Names replaced by roles.** People are referred to by role ("the clinician reviewer", "the project lead") in code, configs and docs,
  because the GitHub repository is public. The five unpushed commits of this branch were rewritten for this; nothing already on GitHub changed.
- **No rule changed.** In 13 checklist configs (v1 to v3_10) only note / status / changelog text changed; every number and rule is identical
  (checked field by field). Their SHA-256 values are therefore different from the values recorded in earlier run folders.
  The run folders are unchanged and keep the hashes of the files as they were when the runs were made.
- **Grade files are looked up by case and date.** The reader's name is part of the export's file name, so launch files and
  `scripts/build_policy_pack.py` now use `*` in its place; exactly one file must match, otherwise the build stops.
- `docs/images/agent_pipeline_explained.png` was re-rendered from its SVG (fonts differ slightly from the earlier copy).
- Not in Git: the onboarding page under `Claude outputs/` (author byline), data, weights, runs, environments.

## Viewer v5: optional axial sliding-slab MIP (2026-11-24)

**Why**: the clinician reviewer said MIP helps him see nodules. Display mode only; no model input, prediction, measurement or checklist answer uses it.

**Implemented**: `scripts/viewer_v5.py` (build `checklist_viewer_v5_canvas2d_mip`; viewer v4 code unchanged, v5 adds to it by asserted insertions),
`--viewer v5` in `scripts/build_viewer_page.py` and `scripts/build_policy_pack.py`, `scripts/test_mip_viewer.py`, `tests/test_viewer_v5.py`,
`docs/VIEWER_V5_MIP_TEST_REPORT.md`. Pack `runs/M9_policy_v3_10_mip_20261124/` = pack M8 content with the v5 viewer (four graded development cases).
- Opens as before (single axial image, no outline). `Single slice` / `MIP`; slab 5 / 8 / 10 mm, 10 mm selected; one button back to the centre image.
- Slab = `max(1, round(requested mm / slice spacing))` images, spacing from the affine; real thickness and image range shown
  (0.75 mm spacing: 5.25 / 8.25 / 9.75 mm; 1.25 mm: 5.0 / 7.5 / 10.0 mm). Shortened at the scan edges, never shifted or padded.
- Maxima from CT values before windowing; one slab computed on demand; cache at most 24 MB, emptied on leaving MIP; no WebGL, no full-volume texture.
- Axial only. Outlines hidden in MIP, with an explanation on the page.

**Tested (technical checks only)**: synthetic 146 of 146 (known maxima, four slice spacings, both scan edges, six stored orientations, windows,
toggling back, outlines, cache limit); real pages 12 of 12 for each of the four cases in Edge and for valid_301 in Chrome, slabs equal to the
original NIfTI read with SimpleITK (0 HU difference), 600 scroll steps + 200 window changes per case without error or memory growth;
pack check 40 of 40; the v4 page test passes on a v5 page (36 of 36); a v4 page rebuilt before and after the change is byte-identical.

**Not tested / not known**: the reviewer's laptop; whether the mode is useful for reading. No clinician usability feedback exists yet.

**Next**: the user decides whether to send pack M9; questions for the reviewer are in the test report, part B.

## Viewer v5.1: MIP layout refinements and thickness convention (2026-11-25)

Display changes only; no model output, measurement or checklist answer changed. Build `checklist_viewer_v5_1_canvas2d_mip`,
pack `runs/M10_policy_v3_10_mip_20261125/` (pack M9, first layout, kept and not to be sent).

**Implemented**
- One-line MIP banner; the long explanation is behind `? MIP help` (floating panel, closed at first).
- Outline buttons not shown in MIP; their bar keeps its height.
- Page opens at Fit. MIP on / off leaves zoom, pan and the size of the scan area unchanged (bars keep their height in both modes).
- Window width and level shown next to the presets.

**Thickness convention, checked and documented** (`scripts/viewer_v5.py`, `docs/VIEWER_V5_MIP_TEST_REPORT.md`): stated thickness = images x spacing
(13 x 0.75 mm = 9.75 mm); first-to-last image centres = (images - 1) x spacing (9 mm). The page shows both. The projection was not changed.
Spacing (from the affine) is not acquisition slice thickness, which is not in the NIfTI affine nor in the CT-RATE metadata we hold; the page says "spacing".

**Tested (technical checks only)**: synthetic 194 of 194; real pages 16 of 16 for each of the four cases in Edge and for valid_301 in Chrome, on the
workstation's integrated AMD Radeon adapter (renderer confirmed); 105 real slabs including both scan edges equal to a direct maximum over the original
NIfTI (0 HU difference); 300 real wheel notches + 600 scroll steps + 200 window changes per case with no error, no memory growth, slab cache inside
24 MB; pack check 40 of 40; v4 page test on a v5.1 page 36 of 36; unit tests 247 passed.

**Not tested / not known**: the reviewer's laptop; whether the mode is useful for reading. No clinician usability feedback exists yet.

**Next**: the user decides whether to send pack M10. Questions for the reviewer: test report, part B.

## Viewer v5.2: short MIP label, outline bar removed in MIP (2026-11-25)

Display changes only. Build `checklist_viewer_v5_2_canvas2d_mip`, pack `runs/M11_policy_v3_10_mip_20261125/` (the one to show; M9 and M10 kept, not to be sent).

**Implemented**
- Label `MIP ≈10 mm · 13 slices · Images 183–195 · Centre 189`; spacing, exact coverage and centre distance of the current slab moved to `? MIP help`.
- Outline bar removed in MIP; "Outlines are not shown in MIP view" is a small notice on the label line. The scan area is taller in MIP.
- The reader's zoom is kept across that change of height (same screen pixels per image pixel, same image point at the centre) and restored exactly on return; from Fit the image stays at Fit.

**Tested (technical checks only)**: synthetic 194 of 194; real pages 17 of 17 for each of the four cases in Edge and for valid_301 in Chrome on the integrated
AMD Radeon adapter; 105 real slabs including both scan edges equal to a direct maximum over the original NIfTI (0 HU difference); scrolling and memory as
before (no error, no growth); pack check 40 of 40; v4 page test on a v5.2 page 36 of 36; unit tests 247 passed.

**Not tested / not known**: the reviewer's laptop; whether the mode is useful for reading. No clinician usability feedback exists yet.

**Next**: the user decides whether to send pack M11.

## NV-Reason-CT added as a third AI reader (shown only) (2026-11-25)

**Why now**: the licence was the open point. Model card and code repository both state OpenMDW-1.1 (use without restriction, no limits on
outputs, keep the licence text when redistributing). Public download, no token.

**Implemented**
- `environments/nvreason` (4.4 GB; torch 2.14.1+cu126, transformers 5.6.2 as pinned upstream, monai), `models/nvreason` (10 GB; weights SHA-256
  checked against the published value). Revision pinned `386b93e`; the two remote code files (874 lines) were read before first use.
- `configs/nvreason_ct_v1.json`, `scripts/run_nvreason.py`, `setup_nvreason.bat`, `run_nvreason_test.bat`, `scripts/fetch_public_file.py`
  (parallel, resumable download; one anonymous connection to the hub gave about 0.5 MB/s).
- The model reads the NIfTI itself as one 3D block (LPS from the affine, 2 mm, 192^3 = 384 mm cube, -1000..1000 HU). Only scans whose left / right
  was verified from anatomy are eligible. Per case the cube pictures and the millimetres outside the cube are saved.
- Asked in its documented way: one structured report, then "Is ... present in this CT? Answer only Yes or No." for each of the 18 reader
  conditions, then a right / left / both question (our wording) after a Yes. The checklist-shaped answer is assembled by our code;
  the model is never offered "cannot tell".
- `scripts/score_reader_checklists.py`: readers against CT-RATE report labels (17 labelled conditions).
- Readers do not vote in checklist_v3_10, so no checklist decision, measurement, threshold or historical page changed. No agent run or pack was rebuilt.

**Tested**
- Runs on Windows 11 / RTX 3090 (upstream lists Linux only): peak GPU memory 12.5 GB, 2.0-3.1 min per scan (report about 22 s, about 6 s per question).
  24 of 24 scans gave a valid answer, 0 failed. Unit tests 275 passed (28 new).
- test20 (20 development scans, `runs/P3_nvreason_20cases_20261125_191112`, scores `runs/E23_reader_label_scores_test20_20261125_201148`):

  | Reader | Found of labelled | Labelled of said present | Extra | Scans with every answer absent |
  |---|---|---|---|---|
  | MedGemma | 24 of 55 | 24 of 58 | 34 | 14 |
  | Hulu-Med | 20 of 53 (6 cannot tell) | 20 of 27 | 7 | 14 |
  | NV-Reason-CT | 43 of 55 | 43 of 64 | 21 | 5 |

  Lung nodule: NV-Reason-CT 8 of 10 labelled, the other two readers 0 of 10. Labels are report mentions, so an "extra" can be a real finding.
- Four clinician-graded scans (`runs/P3_nvreason_3cases_20261125_200014`, `runs/P3_nvreason_1cases_20261125_200819`), 72 graded conditions:
  same answer as the clinician reviewer on 60 (21 present, 39 absent); 9 of 30 present missed; 3 of 42 absent called present.
  Seven of the nine misses are on valid_832 (including pleural effusion, pericardial effusion and hiatal hernia).

**Limits**
- Development scans only; 20 and 4 scans; no confidence intervals; nothing here is a validation.
- The model was trained on CT-RATE training scans from the same hospital and learned that hospital's report wording; the labels it is compared
  with come from those reports. This favours it over the other two readers. We could not check its training list ourselves.
- Different question format from the other two readers (yes / no per condition against one JSON checklist), so the comparison is of the readers
  as we use them, not of the models under one prompt.
- Left / right answers were not scored (the grades and labels carry no side). 58 side answers: 32 both, 15 left, 11 right.
- Parts of the body outside the 384 mm cube are not seen: up to 52 mm in-plane and 21 mm head-to-foot on these scans (chest wall and table, not checked per case).
- ground_glass has no model label of its own; the question wording for it is ours.

**Next (user decides)**: show NV-Reason-CT on the viewer pages as a third reader (`--reader nvreason=<run>` in `agent_run_case.py`);
run it on a fresh, untouched CT-RATE set before any claim; score its left / right answers against the measured side.

## Viewer v6 (Axial 2.5 mm, MIP 10 mm) and NV-Reason-CT on the four pages: pack M12 (2026-11-25)

**Request (clinician reviewer, via the project lead)**: regular axial view 2.5 mm and axial MIP 10 mm, lung window; remove 5 and 8 mm; simple controls.
Project lead: show NV-Reason-CT on the four pages instead of MedGemma / Hulu-Med.

**Inspected first**: slice spacing 0.75 mm (301, 832, 988) and 1.25 mm (541) from the files. Slice thickness is not in the NIfTI header or the CT-RATE
metadata (ZSpacing only); the report text states 1.5 mm for 301, 541, 988 and nothing for 832. No 2.5 mm reconstruction exists (other reconstructions: 1.5 / 1.25 mm apart).

**Implemented**
- `scripts/viewer_v6.py`, build `checklist_viewer_v6_canvas2d_axial25_mip10`; `build_viewer_page.py --viewer v6`; `build_policy_pack.py --viewer v6`.
  Opens on "Axial 2.5 mm" (mean slab from HU before windowing, rounded to whole HU), Fit, lung window, no outline. "MIP 10 mm". "Details" holds exact
  figures and the native thin slices. Outlines on native slices only. Same slab rule as v5 (whole slices from spacing, centred, shortened at edges).
  0.75 mm: 3 slices = 2.25 mm and 13 slices = 9.75 mm; 1.25 mm: 2 slices = 2.5 mm and 8 slices = 10 mm.
- `--reader-run` in both builders: NV-Reason-CT's answer under each tool answer ("shown only, not counted") and its report in a closed card.
  Display only: the stored checklist files still say "no reader answer"; the page shows the reader's answer in that place.
- Pack `runs/M12_policy_v3_10_axial25_nvreason_20261125` (same settings as M11 otherwise). Not sent.
- `scripts/test_slab_viewer.py`, `tests/test_viewer_v6.py`, `docs/VIEWER_V6_TEST_REPORT.md`.

**Tested (technical checks only)**: synthetic 182 of 182; real pages 16 of 16 for each of the four cases in Edge and for valid_541 in Chrome on the
integrated AMD Radeon adapter; 90 real mean / MIP slabs including both scan edges equal a direct calculation over the original NIfTI (0 HU);
tool answers and embedded scan data identical to M11; M11 unchanged and a rebuilt M11 page byte-identical; unit tests 298 passed.

**Not tested / not known**: the reviewer's laptop; whether 2.25 mm is acceptable as "2.5 mm"; whether the AI reader line helps. No clinician feedback yet.
MedGemma and Hulu-Med are still called by the older batch files for new case sets; those were not changed.

**Next**: the project lead decides whether to send pack M12.

**Addition (2026-11-25): NV-Reason-CT yes / no score.** `run_nvreason.py` now stores `score_yes` per condition in `questions.json`: the share of "Yes" between
"Yes" and "No" for the first answer token (0 to 1, uncalibrated, not a probability of disease). Four graded scans re-asked (`runs/P3_nvreason_3cases_20261125_214608`,
`runs/P3_nvreason_1cases_20261125_215309`): answers identical to the first run. 68 of 72 scores are below 0.05 or above 0.95; 7 of the 9 findings the model
missed against the clinician reviewer have a score below 0.02, so the score does not flag most misses. Capital "Atelectasis" against lowercase on 24 scans: 0 answers changed
(`runs/test_logs/nvreason_caps_probe.json`). Pack M12 is unchanged and does not show the score.

## checklist_v3_11: NV-Reason-CT decides "present" for three conditions; pack M13 (2026-11-25)

**Decision (project lead)**: for medical material, hiatal hernia and bronchiectasis the tool should give the AI reader's answer instead of "cannot tell", with the reader's score as a tooltip.

**Implemented**
- `configs/checklist_v3_11.json` (= v3_10 plus `reader_decides_present: NV-Reason-CT` on those three conditions, block `trusted_reader`, decision rule 2e).
  Reader "present" -> "present" (basis "AI reader NV-Reason-CT"). Reader "absent" -> unchanged, "cannot tell": a reader's "absent" does not establish absence,
  because on the four graded scans it answered absent for bronchiectasis (valid_301) and hiatal hernia (valid_832) where the clinician reviewer marked present.
  No score limit: the reader's own Yes / No is used; the score (0 to 1, uncalibrated) is stored and shown.
- `lung_ct.agent.checklist.build_checklist` (v3_10 and earlier give identical answers); `rescore_with_inventory_rows.py --reader NAME=RUN`;
  `run_nvreason.py` stores `score_yes` in the assembled answer; page builder: tooltip with the score on every reader line and on reader-decided answers,
  reader line reads "used for this answer"; CSV rows carry `agent_decision_basis = AI reader NV-Reason-CT`.
- Reader runs with scores and reports: `runs/P3_nvreason_3cases_20261125_221636`, `runs/P3_nvreason_1cases_20261125_222432`.
  Re-score: `runs/E24_rescore_v3_11_20261125`. Pack `runs/M13_policy_v3_11_axial25_nvreason_20261125` (viewer v6). Not sent.

**Result on the four graded scans** (development data): 4 of 80 answers change against v3_10, all from "cannot tell" to "present": valid_301 hiatal hernia (score 0.90)
and medical material (1.00); valid_832 bronchiectasis (1.00) and medical material (1.00). All four match the clinician reviewer's grade. Still "cannot tell" although the
reviewer marked present: bronchiectasis on valid_301 (score 0.00) and hiatal hernia on valid_832 (0.00). No other answer changes; embedded scan data identical to M12.

**Tested**: page checks 16 of 16 on all four M13 pages and on an M12 page (Edge, integrated graphics); M12 files and the v3_10 checklist files unchanged; unit tests 311 passed (13 new).

**Limits**: a project decision, not a validation. The supporting numbers are tiny and from development scans (test20 against report labels, hit / miss / extra: medical material
1 / 0 / 1, hiatal hernia 1 / 0 / 1, bronchiectasis 2 / 1 / 0). The reader was trained on reports from the same hospital. The rule was chosen after seeing these four scans, so they
cannot be used to judge it. The other development scans were not re-scored. The score is almost always near 0 or 1 and does not flag the reader's misses.

**Next (project lead decides)**: send M13; test the rule on scans not used so far.

## Prompt test template; PT01: COLIPRI prompt wording (2026-11-26)

**Implemented**
- `docs/PROMPT_TEST_TEMPLATE.md`: reusable plan for testing prompt wording of any model (steps, rule "decide before you look", plan and result forms).
- First use: `docs/prompt_tests/PT01_colipri_prompts_PLAN.md` (frozen before any score), `configs/colipri_prompt_sets_v1.json`,
  `scripts/prompt_test_colipri.py` (each scan encoded once; `--reuse` re-analyses stored embeddings), `tests/test_prompt_test.py`.

**Result** (`runs/PT01_colipri_prompts_20261126_073924`, `docs/prompt_tests/PT01_colipri_prompts_RESULT.md`): 85 development scans, 70 patients, 18 CT-RATE labels.
Macro AUROC: A current short prompts 0.886; B four sentence forms 0.888 (difference +0.002, interval -0.007 to +0.015); C several names 0.886 (0.000, -0.006 to +0.006);
D report passages 0.869 (-0.017, -0.041 to +0.007; atelectasis 0.87 -> 0.64). No set met the pre-stated rule. **Set A stays; nothing changed.**
Set A scores equal the stored pipeline scores (largest difference 0.0001).

**Tested**: unit tests 322 passed (11 new).

**Limits**: development scans and report-mention labels, so a choice of method only, not a performance figure; five conditions have fewer than 10 positive scans;
set D used validation reports of other patients, not training reports.

**Open**: `configs/colipri_ct_v1.json` still quotes "external test AUROC about 0.77 (RAD-ChestCT)", a third-party figure for a fine-tuned variant; the paper's own
zero-shot figures on RAD-ChestCT are 0.60 to 0.67. Not changed (the note is copied into stored run files).

## Second segmentation: NV-Segment-CT, all supported classes; viewer v7; pack M14 (2026-11-26)

Looked at four NVIDIA models. NV-Reason-CT is already our third reader. NV-Reason-CXR reads chest X-rays only (not our data). NV-Generate-CTMR makes synthetic CT
(for training, outside scope; parked). NV-Segment-CTMR / NV-Segment-CT segment organs: the CT-only one was chosen (same lung classes, permissive weights licence).

**Implemented**
- `configs/nvsegment_ct_v1.json`, `scripts/run_nvsegment.py`, `setup_nvsegment.bat`, `run_nvsegment_test.bat`, `environments/nvsegment`, `models/nvsegment` (0.9 GB;
  nvidia/NV-Segment-CT @ afb5151, weights SHA-256 checked at load). The network is MONAI's `vista3d132`; the repository's Python files are not executed.
  Preprocessing and inference follow the vendor pipeline step for step. All supported classes: 117 ("segment everything", one label map) + 7 extra classes
  (class 'lung tumor', pancreatic / hepatic tumour, colon cancer, hepatic vessel, bone lesion, airway; one mask each). Output on the source grid, geometry checked before writing.
- `scripts/compare_nvsegment_totalseg.py`: agreement with TotalSegmentator (not accuracy).
- `scripts/viewer_v7.py` (build `checklist_viewer_v7_canvas2d_axial25_mip10_two_segmentations`): v6 plus the bar "Segmentation models" with TotalSegmentator and
  NV-Segment-CT buttons. `build_viewer_page.py --viewer v7 --second-seg RUN`, `build_policy_pack.py --viewer v7 --second-seg-run RUN`.
  `scripts/test_seg_viewer.py`, `tests/test_nvsegment.py`, `docs/VIEWER_V7_TEST_REPORT.md`. `test_slab_viewer.py --viewer-build` runs the v6 page checks on a v7 page.
- Runs: `runs/P2_nvsegment_3cases_20261126_123711` (301, 541, 988), `runs/P2_nvsegment_1cases_20261126_123758` (832); three earlier runs of the same scans with an
  earlier copy of the config are kept (identical mask files). Comparison `runs/E25_nvsegment_vs_totalseg_20261126_123834`.
  Pack `runs/M14_policy_v3_11_two_segmentations_20261126` = M13 content (policy checklist_v3_11, NV-Reason-CT lines) on viewer v7. Not sent.

**Result on the four graded scans** (development data; two models compared with each other, no clinician outline)
- Runs on Windows / RTX 3090: 11 to 26 s per scan, 8 to 14 GB peak. Every scan run twice gave identical files. 104 to 113 of 124 classes received voxels.
- Dice against TotalSegmentator, lowest / middle / highest: left upper lobe 0.91 / 0.97 / 0.98; left lower 0.89 / 0.95 / 0.98; right upper 0.93 / 0.95 / 0.97;
  right middle 0.78 / 0.87 / 0.96; right lower 0.86 / 0.97 / 0.98; both lungs 0.92 / 0.96 / 0.98; trachea 0.72 / 0.88 / 0.94; heart 0.77 / 0.96 / 0.97;
  aorta 0.92 / 0.93 / 0.95; liver 0.95 / 0.98 / 0.98. The lowest values are mostly valid_832 (heart 0.77, right middle lobe 0.79) and valid_301.
- Airways: Dice 0.40 to 0.77; volume 17 to 25 mL (NV) against 19 to 58 mL (TS). The classes are defined differently, so this does not rank them.
  Separate pieces TS / NV: 28 / 5, 42 / 22, 60 / 27, 69 / 69.
- Class 'lung tumor': valid_301 48.5 mL in 29 pieces (16.8 mL inside its own lobes), valid_832 7.8 mL in 23 pieces, 0 on valid_541 and valid_988.
  Dice with TotalSegmentator's lung_nodules mask 0.23, 0.07, 0, not defined. Other tumour classes also fire in every scan (colon cancer primaries up to 11.4 mL, bone lesion up to 10 mL,
  hepatic tumor up to 3.1 mL). Classes not expected in a chest scan received voxels in every scan (for example skull 89 mL and vertebra L3 69 mL on valid_832, left femur 12.8 mL on valid_301).

**Tested**: unit tests 335 passed (13 new). Viewer: synthetic 49 of 49; real pages 8 of 8 on all four (every button equals the original mask files pixel for pixel);
v6 page checks on the v7 pages: 16 of 16 on all four. Pack M13 unchanged.

**Limits**: four development scans; agreement between two models, not accuracy; nothing here was compared with the clinician reviewer's grades or with any outline drawn by a person.
The tumour classes and the out-of-chest classes show the second model marks things that are not there, so none of its classes should feed an answer without its own check.
The model card reports no accuracy figures and supports Linux only. Point-click segmentation (the model's interactive mode) was not tried.

**Next (project lead decides)**: send M14 or stay with M13; whether the NV airway mask should be tried in the airway-tree measurement (Phase 5) as a sensitivity check;
whether to ask the clinician reviewer which lobe outline looks better where the two differ (valid_832, valid_301).

## Both segmentation models against expert lung masks (2026-11-26)

**Implemented**: `scripts/score_segmenters_vs_reference.py` (core env). Reference: the expert left / right lung masks and infection masks of the COVID-19 CT set already on
disk (Phase 1); no download. NV-Segment-CT run `runs/P2_nvsegment_10cases_20261126_132013` (ten HU-valid scans; the ten display-scaled scans are skipped and listed).
TotalSegmentator masks are the Phase 2 ones in `data/derived`. Result folder `runs/E26_segmenters_vs_expert_lungs_20261126_132301`.

**Result** (10 scans, 20 lungs; whole lungs only)
- Dice, lowest / middle / highest: TotalSegmentator 0.975 / 0.982 / 0.987; NV-Segment-CT 0.966 / 0.977 / 0.984.
  Paired difference (NV minus TS) -0.006, 95 % interval -0.008 to -0.004 (patients resampled). TotalSegmentator higher on 9 lungs, within 0.005 on 11, NV higher on none.
- Volume error, middle / largest: TotalSegmentator 0.4 % / 1.2 %; NV-Segment-CT 1.2 % / 3.8 %.
- Share of the reference infected lung that the lung mask covers, lowest / middle: TotalSegmentator 94 % / 99.6 %; NV-Segment-CT 71 % / 96.5 %.
  NV-Segment-CT leaves out part of the diseased lung on coronacases_002 (71 % and 81 %) and coronacases_007 (73 % and 78 %).

**Tested**: unit tests 336 passed (1 new).

**Limits**: ten scans from one source, COVID-19 pneumonia only; whole lungs, not lobes, airways or any other class; whether this set was in either model's training data
is not established (it is not in the part of the VISTA3D dataset table that could be read; TotalSegmentator was trained on its own collection). The differences in Dice are small;
the difference in covering diseased lung is the practically relevant one for this project.

**Decision supported**: TotalSegmentator stays the segmentation the tool uses for the lungs.

**Open (project lead decides)**: a lobe-level check needs new data. The only public expert lobe set found is 50 LUNA16 scans (CC BY-NC-SA; a third-party list says it contains
errors); the scans must be taken from the LUNA16 archives, which are far larger than the 50 scans. Raw data is at 20 of 20 GB, so this needs a storage exception.
No public airway set was found that is both free of registration and outside NV-Segment-CT's training data (AeroPath is in it; ATM'22 and AIIB23 need registration).

**Closed (project lead, 2026-11-26)**: the segmentation-model comparison stops here. No lobe or airway data will be downloaded. TotalSegmentator stays the model the tool uses;
NV-Segment-CT stays display-only in viewer v7. No further runs of this comparison without a new request.

## Prompting experiments 1 to 4: NV-Reason-CT wording (PT02), cut-offs and trust rules (TR01) (2026-11-26)

**Implemented**: `configs/nvreason_prompt_sets_v1.json` (frozen), `scripts/prompt_test_nvreason.py` (nvreason env; asks, never reads a label),
`scripts/prompt_test_yesno.py` and `scripts/trust_rules.py` (core env; scoring). Plans frozen before any figure was computed and results in `docs/prompt_tests/`
(PT02_nvreason_prompts_*, TR01_cutoffs_and_trust_rules_*). The scan is now processed once and reused for every question (1.2 s instead of 5.2 s per question);
on 5 scans, 15 answers were also produced the ordinary way and were identical.
Runs: `runs/PT02_nvreason_prompts_20261126_134837` (83 scans, 7 ways of asking, 10,541 answers, 0 failed, 0 unreadable), `runs/TR01_trust_rules_20261126_190105`.

**Result** (development scans, reference = CT-RATE report labels)
- Experiment 1, way of asking: no set met the pre-stated rule; the current wording stays. Macro balanced accuracy: current 0.814, vote of three wordings 0.812,
  reasoning on 0.814, definition in the question 0.810, report first 0.814, location first 0.828 (interval of the difference -0.019 to +0.052; 87 more "extra" answers,
  three conditions more than 0.05 worse).
- Experiment 2, three wordings: same answer on 98.4 % of 1,411 scan-conditions. Disagreement marks 8 of 156 errors; a Yes-score between 0.05 and 0.95 marks 29 of 156.
  Both meet the pre-stated test for a useful flag, but 127 of 156 errors are given with a score near 0 or 1.
- Experiment 3, trust rules: of 52 source-condition pairs none may decide "present"; one may decide "absent" (COLIPRI with cut-off, consolidation). No trust table adopted.
  The three conditions where checklist_v3_11 lets NV-Reason-CT decide "present" do not meet the bar against report labels
  (medical material 5 of 9 calls labelled, hiatal hernia 4 of 8, bronchiectasis 7 of 10).
- Experiment 4, COLIPRI cut-offs: fitted per condition with each patient left out; they range from 0.009 to 0.993 and several are unstable. Proposals only.

**Tested**: unit tests 350 passed (14 new, `tests/test_prompt_test_yesno.py`).

**Limits**: development scans; report labels count a correct finding the report did not mention as an error; 10 to 20 positive scans for most conditions, five conditions
with fewer than 10; the model was trained on reports from the same hospital. MedGemma and Hulu-Med (20 scans) were not judged. The tool's answers are from a 58-scan
checklist_v3_9 run. Nothing was applied to any checklist, page or pack.

**Open (project lead decides)**: whether checklist_v3_11's three reader-decides conditions should be looked at again (for example by the clinician reviewer on the scans
where the reader says present and the report does not); whether to show a "less certain" mark from the Yes-score on pages; a confirmation run on untouched scans would
need a data-budget exception.

## Data budget removed (project lead, 2026-11-26)

The 20 GB new-data and 30 GB model-weight limits of the initial scope no longer apply (D: 6.5 TB free of 7.3 TB; data/raw 20 GB, models 74 GB at the time).
Still in force: D: only, never C:; download size and source recorded per run; CT-RATE is not mirrored in full; licence and access terms are checked before any download.
Earlier STATUS entries and the frozen plans that mention the budget are left as written.

## TR02: calibration on 400 new CT-RATE patients - prepared, download waiting for the token (2026-11-26)

**Implemented**: plan frozen before any new scan was downloaded (`docs/prompt_tests/TR02_calibration_400_PLAN.md`); `scripts/ctrate_fetch_calibration.py`
(label-blind random choice of 400 of the 1,234 validation patients not used so far, seed 20261126; first 200 = fit half, next 200 = confirm half; data in
`D:/datasets/ct_rate/calib400`); `scripts/segment_orientation_organs.py` (organ masks for the left/right check only); `--seg-root` / `--write-scan-list` in
`anatomy_contrast_check.py`; `--scan-list` in `prompt_test_nvreason.py`; `--scan-list --scores-only` in `prompt_test_colipri.py`; `scripts/trust_rules_confirm.py`;
`fetch_ctrate_calibration.bat` (all five steps, unattended after the token).

**Tested**: unit tests 357 passed (7 new, `tests/test_tr02_calibration.py`). Chain test on 3 existing scans (`runs/TR02_chaintest_*`): organ masks, check
(valid_64_a_3 left out again, as in its earlier QC), NV-Reason-CT answers identical to PT02 (36 of 36), COLIPRI scores identical to PT01 (largest difference 0.000001).
Selection dry run: 400 patients, none already used; 353 standard-kernel reconstructions, 47 not (kept, recorded).

**Run (2026-11-26 20:04 to 2026-11-27 05:35)**: 400 downloaded (89.6 GB, 0 failed), 400 HU-valid, 395 orientation-verified (5 left out), 198 fit / 197 confirm.
NV-Reason-CT 0 unreadable, 60 equivalence checks identical; COLIPRI 0 failed. Scoring `runs/TR02_calibration_20261127_053515`; result `docs/prompt_tests/TR02_calibration_400_RESULT.md`.

**Result**: no source may decide "present" for any condition. Confirmed in both halves: COLIPRI "absent" for consolidation; COLIPRI and NV-Reason-CT "absent" for
coronary artery wall calcification. TR01's cut-offs and its consolidation entry do not hold on the new scans (withdrawn). Several COLIPRI cut-offs lose much on the confirm
half (septal thickening 0.80 to 0.49, medical material 0.87 to 0.66). For checklist_v3_11's three reader-decides conditions the report agrees with 66 % (medical material),
59 % (hiatal hernia) and 40 % (bronchiectasis) of NV-Reason-CT's "present" calls.

**Open (project lead decides)**: whether to propose a checklist version with the three "absent" entries; whether checklist_v3_11's reader-decides rule (bronchiectasis in
particular) should be reviewed by the clinician reviewer; the confirm half is used up.

## Model against report for every condition; review sample for two readers (2026-11-27)

**Implemented**: `scripts/model_vs_report_table.py` (all 18 conditions, both models, on the 395 calibration scans; output
`runs/TR02_calibration_20261127_053515/model_vs_report_all_conditions.{md,csv}`, descriptive). `scripts/review_sample_model_vs_report.py`: random sample, seed 20261127,
`runs/RS01_review_sample_20261127` (seed 20261127, `--all-conditions`): 46 scans, every page asks all 18 conditions. Scans were added until each labelled
condition had at least 3 scans where the model says present and the report is silent (8 for bronchiectasis, hiatal hernia, medical material), 2 where the report
mentions it and the model does not, and 1 where both say present; result: 118 "model present, report silent" items (3 to 12 per condition).
`scripts/build_reading_pages.py`: one blinded page per scan (viewer v6 CT display, no outlines; Present / Absent / Cannot tell + optional location per condition;
autosave per reader; CSV download). No model answer, score, report or sample cell on the pages; reader names given on the command line only.
Pages: `runs/RS01_review_sample_20261127/pages` (46 pages, 4.8 GB, index.html). `sample.json` holds the cells and is for the analyst only.

**Tested**: unit tests 358 passed. `scripts/test_reading_page.py` (Edge, file://): 11 of 11 on pages 1, 7 (largest, 128 MB) and 46. Technical checks only.

**Next**: the project lead sends the pages to the two readers; reading CSVs come back; analysis per condition (model-only items confirmed by each reader, reader agreement).

**Withdrawn (2026-11-27)**: the RS01 reading pages v1 (`pages_v1_WITHDRAWN_do_not_send`) used a one-off layout. The project lead asked for the standard page;
it is now written down in `docs/REVIEW_PAGE_FORMAT.md` (from pack M14). RS01 pages will be rebuilt with `build_viewer_page.py`, which needs a full agent run
(segmentation, measurements, checklist, AI reader) on each of the 46 scans.

## RS01 review pages in the standard format: pack M15 (first 10 scans) and the full pipeline script (2026-11-27)

**Implemented**: `docs/REVIEW_PAGE_FORMAT.md` (the fixed page format, taken from pack M14). `run_full_pipeline.sh` (all ten steps behind pack M14 on the scans of
one verified QC run; `FROM_STEP` restarts part-way). `build_policy_pack.py --cases-file` (new cases without earlier grades). `nodule_evidence_rows.py` now
adds E97 / E98 when the dense-lung inventory was not computed for a scan. The 46 RS01 scans are hard links in `data/raw/ct_rate/volumes` to
`D:/datasets/ct_rate/calib400/volumes` (no copy) and are in `data/raw/ct_rate/cases.json` as set `rs01` (backup of the earlier file in `D:/datasets/ct_rate`).
QC runs `runs/P1_qc_verified_rs01_{first10,rest36,all46}_20261127_072059`.

**Pack M15** `runs/M15_policy_v3_11_rs01_first10` (10 pages, 1.3 GB, open, CT-RATE label column, AI reader, second segmentation). All steps 10 of 10 ok;
valid_659_a_1: dense-lung inventory not computed (image too grainy for the consolidation range), listed, other rows present. Pages 46 to 238 MB
(larger than M14's 67 to 127 MB because some scans are long).

**Tested**: unit tests 358 passed. Page checks (Edge): viewer v6 checks 16 of 16 on valid_849 (largest) and valid_477 (smallest); segmentation checks 8 of 8 on
valid_477, 7 of 8 on valid_849. The failed check samples the middle axial image of the scan; valid_849 is 1,047 images long and that image is below the right lung,
where the second model labels 4 stray voxels as right lung near the midline. On a mid-lung image (349) the labels lie on the correct sides in the file, and the
pixel-by-pixel check that every button draws exactly the file's classes passed. Not a display error; the check's choice of image does not suit long scans.

**Running**: the same pipeline on the other 36 scans (pack M16).

## checklist_v3_12: two sources for "present", no "suggested by segmentation" (project lead, 2026-11-27)

**Implemented**: `configs/checklist_v3_12.json`; `_two_source` in `src/lung_ct/agent/checklist.py` (active only with `two_source_rule`); page and pack wording
for v3_12 in `build_viewer_page.py` / `build_policy_pack.py`. Rule `two_source_v1_tr02`:
- A positive answer from a predicted mask alone (pleural fluid >= 20 mL outlined, nodule candidates) is "cannot tell"; the statement lists what was outlined.
- Pleural effusion, aortic and coronary calcification: "present" when NV-Reason-CT says present AND COLIPRI is above the TR02 fit-half cut-off
  (0.8197 / 0.0620 / 0.1346) or, for pleural effusion, >= 20 mL fluid is outlined. Only where the answer would otherwise be "cannot tell".
- Medical material, hiatal hernia, bronchiectasis: NV-Reason-CT alone no longer gives "present".

**Evidence for the choice** (TR02, report labels, confirm half; share of "present" calls the report mentions): NV-Reason-CT alone vs with COLIPRI above the
cut-off: pleural effusion 73 % -> 91 %, aortic 89 % -> 88 %, coronary 88 % -> 90 %. For 12 of 17 conditions no COLIPRI cut-off on top of NV-Reason-CT reached
90 % on the fit half; for the others the gain was small or bought by finding fewer. Report labels cannot separate shared model errors from findings the
report left out: the readers' RS01 grades are the reference for any further condition.

**Effect on the first 10 RS01 scans** (v3_11 -> v3_12): 9 nodule "suggested" -> cannot tell; 2 pleural "suggested" -> present (two sources); 6 coronary and
1 aortic cannot tell -> present (two sources); 10 reader-only "present" (medical material, hiatal hernia, bronchiectasis) -> cannot tell.

**Tested**: unit tests 367 passed (9 new, `tests/test_v3_12.py`).

**Next**: pack M17 (`runs/M17_policy_v3_12_rs01`) with v3_12 for all 46 RS01 scans once pack M16's pipeline has finished; M15 / M16 (v3_11) are kept, not to be sent.
