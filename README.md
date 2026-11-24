# Lung CT Research

Local, reproducible research pipeline for public 3D chest CT: ingestion, geometry/HU QC,
lung segmentation, volume quantification, CT-capable LVLM inference, reviewable reports.
Research use only - no diagnostic or clinical validation claims. See `PROJECT_GUIDELINES.md` and `docs/STATUS.md`.

**Project direction:** [Project plan and agent pipeline](docs/PROJECT_PLAN.md) · [Paper novelty (candidate contributions)](docs/PAPER_NOVELTY.md) · [Model research report](docs/research/Chest%20CT%20model%20candidates.md)

## Setup

Requires Python 3.11. Data, model weights and run outputs (`data/`, `models/`, `runs/`) are
not in the repository. Download them with the commands below.

```bash
git clone https://github.com/vaniinfo/lung_ct.git
cd lung_ct
```

### Platform support

| Stage | Windows + NVIDIA GPU | macOS |
|---|---|---|
| Core: download, QC, evaluation, reports, airway (Phase 5), review package, tests | Tested | Should work (no GPU or torch needed), **not yet tested** |
| Phase 2 TotalSegmentator (`segment_totalseg.py`) | Tested (RTX 3090, CUDA) | **Not supported as written**: the script requires a CUDA GPU |
| Phase 3 MedGemma inference (`run_medgemma.py`) | Tested (RTX 3090, CUDA) | **Not supported as written**: only `--dry-run` (builds inputs, loads no model) avoids CUDA |

Running the GPU stages on a Mac (CPU or Apple MPS) would need code changes and separate
validation. Without that, results from a Mac should not be compared with the CUDA runs.

### Core environment (`.venv`)

Windows (Git Bash):

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e ".[test]"
```

macOS (Terminal, zsh or bash):

```bash
python3.11 -m venv .venv          # e.g. from `brew install python@3.11`
source .venv/bin/activate
pip install -e ".[test]"
```

### GPU environments (Windows + NVIDIA only)

Each GPU stage has its own environment, run as separate processes (see `PROJECT_GUIDELINES.md`).
The versions below are the ones installed in the tested environments. The commands are
reconstructed from those versions and use [uv](https://docs.astral.sh/uv/).

```bash
# segmentation: TotalSegmentator + CUDA 12.4 PyTorch
uv venv environments/segmentation --python 3.11
uv pip install --python environments/segmentation/Scripts/python.exe \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python environments/segmentation/Scripts/python.exe totalsegmentator==2.18.0

# vlm: MedGemma 1.5 4B + CUDA 12.6 PyTorch
uv venv environments/vlm --python 3.11
uv pip install --python environments/vlm/Scripts/python.exe \
    torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python environments/vlm/Scripts/python.exe \
    transformers==5.22.2 accelerate==1.15.0 pillow nibabel numpy pyyaml

# check CUDA is visible
environments/vlm/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect True
```

MedGemma is a gated model. Accept its terms on Hugging Face and keep the token in an
environment variable (`HF_TOKEN`), never in files.

On macOS the environment's Python is at `environments/<name>/bin/python` rather than
`environments/<name>/Scripts/python.exe`. The same applies to `.venv`.

## Tests

```bash
python -m pytest -q
```

## Datasets used

All CT images in this project come from **one** public dataset. No other CT database has been
downloaded or analysed.

| Dataset | Source | Contents | Licence |
|---|---|---|---|
| COVID-19 CT Lung and Infection Segmentation (COVID-19-CT-Seg) | Zenodo record [3757476](https://zenodo.org/records/3757476); benchmark: [JunMa11/COVID-19-CT-Seg-Benchmark](https://github.com/JunMa11/COVID-19-CT-Seg-Benchmark) | 20 COVID-19 chest CT volumes (NIfTI) with expert left/right lung masks and infection masks (~1.1 GB) | CC BY 4.0 (per Zenodo record, checked 2026-11-13) |

**Example results:** [docs/RESULTS_COVID19_CT_SEG.md](docs/RESULTS_COVID19_CT_SEG.md) shows images
and measurements from this dataset: lung and infection masks, predicted lobes, airway tree,
and lung volumes, Dice and airway metrics for all 10 HU cases.

The 20 cases originally come from two public sources, and the difference matters for analysis:

- `coronacases_001`-`010` (10 cases): originally from coronacases.org; intensities are in HU.
- `radiopaedia_*` (10 cases): originally from Radiopaedia; 8-bit display-scaled, **not HU**.
  These are flagged `hu_valid=False` and excluded from HU-dependent analysis.

The archives are verified against pinned Zenodo MD5 checksums by `scripts/download_covid_ct.py`.
The lung and infection masks are expert **reference** annotations and are kept separate from
pipeline predictions. Infection masks are not fibrosis or GGO labels.

Not used: CT-RATE (planned E4 comparison with paired reports, not run). No airway reference
dataset is used; the airway masks are TotalSegmentator predictions.

Please cite: Ma J. et al., "Toward data-efficient learning: A benchmark for COVID-19 CT lung and
infection segmentation", *Medical Physics* 48(3):1197-1210, 2021; and the dataset itself,
doi:[10.5281/zenodo.3757476](https://doi.org/10.5281/zenodo.3757476).

### Candidate datasets (not yet used)

**None of these datasets has been downloaded or analysed.** They are options for later work.
Details were checked against official pages on 2026-11-13. Recheck the licence and access terms
before downloading anything. Any download must fit the data budget in `PROJECT_GUIDELINES.md`
(at most 20 GB of new data), so large sets would be subsets only.

| Dataset | What it would add | Size / access | Licence | Caveats |
|---|---|---|---|---|
| [ATM'22](https://atm22.grand-challenge.org/) (Airway Tree Modeling) | Airway reference labels by 3 radiologists: 300 training + 50 validation CTs (150 test cases withheld). Fills the Phase 5 gap, since we have no airway reference. | Open on Zenodo, no registration: [TrainBatch1](https://zenodo.org/records/7949582) 32.3 GB, [TrainBatch2](https://zenodo.org/records/7949571) 24.7 GB. The challenge site's dataset page needs a login. | CC BY 4.0 (labels); source images keep their own terms | Many cases come from LIDC-IDRI (CaseInfo CSV maps them). EXACT'09 images must be fetched separately. Discard `ATM_164` (misaligned label). |
| [AeroPath](https://zenodo.org/records/10069289) | 27 CTA scans with airway and lung masks, with pathology (tumours, emphysema, sarcoidosis). A small first airway validation set that fits the budget. | Zenodo / [Hugging Face](https://huggingface.co/datasets/andreped/AeroPath), NIfTI | CC BY 4.0 (dataset); MIT (code) | Contrast-enhanced CTA, not the non-contrast CT we use now |
| [AIIB23](https://codalab.lisn.upsaclay.fr/competitions/13238) | Airway labels in fibrotic interstitial lung disease (ILD), plus a survival task. Fits the later ILD work. | Download links are emailed after challenge registration; test set not public | Not stated on the pages checked | Published case counts and splits disagree across sources |
| [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | CTs paired with radiology reports, plus 18 abnormality labels (including bronchiectasis and interlobular septal thickening). Planned E4 experiment. | Gated on Hugging Face; 25,692 volumes from 21,304 patients, about 21.3 TB in total | CC BY-NC-SA 4.0; no redistribution | Small subset only (project guidelines: do not mirror in full). Labels come from reports, not image annotations. |
| [CT Images in COVID-19](https://www.cancerimagingarchive.net/collection/ct-images-in-covid-19/) (TCIA) + [COVID-19-20](https://covid-segmentation.grand-challenge.org/Data/) lesion masks | Many more COVID CTs (632 subjects): 199 training CTs with lesion masks. Would extend our 10 HU cases. | 14.71 GB NIfTI via Aspera | CC BY 4.0 (images); CC0 (challenge masks) | Masks label COVID lesions, not GGO or fibrosis. Check HU scaling per case. |
| [MosMedData COVID19_1110](https://mosmed.ai/datasets/covid19_1110) | 1,110 COVID CTs with severity grades CT-0 to CT-4; 50 with GGO/consolidation masks | Public download | CC BY-NC-ND 3.0 | No redistribution of modified data. Severity classes are heavily unbalanced (only 2 CT-4). |
| [LCTSC 2017](https://www.cancerimagingarchive.net/collection/lctsc/) (TCIA) | 60 CTs with expert left/right lung contours. A second lung-segmentation reference. | 5.15 GB DICOM-RT via TCIA Data Retriever | CC BY 3.0 | Radiotherapy-planning CTs (oncology patients) |
| [MSD Task06 Lung](https://registry.opendata.aws/msd/) | 63 labelled CTs with non-small-cell lung tumour masks | About 6.2 GB tar | CC BY-SA 4.0 | Share-alike applies to derived works. Tumour labels only. |
| [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) (TCIA) | 1,010 patients with nodule annotations from 4 radiologists. Wide range of scanners. | 133 GB via TCIA Data Retriever | CC BY 3.0 | Subset only. Overlaps with LUNA16 and ATM'22. |
| [LUNA16](https://luna16.grand-challenge.org/Data/) | 888 LIDC-IDRI CTs (slices 2.5 mm thick or thinner) with nodule reference. Includes lung masks. | Download from grand-challenge | CC BY 4.0 | The lung masks are automatic and **not** a reference standard. Subset of LIDC-IDRI. |
| [VESSEL12](https://vessel12.grand-challenge.org/) | 20 CTs with lung vessel annotations and lung masks | Registration on grand-challenge | Not stated on the official page | Only relevant if vessel work comes into scope |
| [OSIC Pulmonary Fibrosis Progression](https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression/data) (Kaggle) | 176 ILD patients: baseline CT plus FVC (lung function) follow-up | About 22 GB DICOM; you must accept the competition rules | Competition rules (could not be read without login) | Confirm the rules allow research use outside the competition |
| [LTRC](https://www.nhlbi.nih.gov/science/lung-tissue-research-consortium-ltrc) | ILD and COPD CTs with clinical and pathology data | Application through NHLBI BioLINCC | Data use agreement | Not public; needs an approved application |
| [COPDGene](https://www.copdgene.org/) | Large COPD cohort, inspiratory and expiratory CT | Ancillary study proposal + data use agreement | Data use agreement | Not public; needs an approved application |

**Do not use for evaluating TotalSegmentator:** the [TotalSegmentator CT dataset](https://zenodo.org/records/10047292)
(1,228 CTs, 117 structures, CC BY 4.0) is TotalSegmentator's own training data, so scoring
TotalSegmentator on it would overstate accuracy.

**Not useful as a reference:** EXACT'09 (40 airway CTs) never released its reference
annotations; evaluation is done by the organisers.

**Bronchiectasis:** no public CT dataset with voxel-level bronchiectasis annotations was found.
The closest options are the CT-RATE report-derived bronchiectasis label, PatchChestCT
(patch-level labels on CT-RATE scans) and BronAtlas (segmental bronchi labels including an
"abnormal bronchus" class). Their licences have not been checked yet.

## Phase 1: Data

**In plain words:** download 20 public chest CT scans, then check each one is usable: the image size and orientation are correct, left and right are not swapped, and the pixel values are real CT density units (HU) rather than a picture brightened for display.

```bash
python scripts/download_covid_ct.py      # COVID-19-CT-Seg (Zenodo 3757476) -> data/raw/covid_ct/
python scripts/process_first_case.py     # E0: QC + reference volumes for coronacases_001 -> runs/E0_*
python scripts/batch_qc.py --n 5         # batch QC (also --n 20, or --cases <id> ...) -> runs/P1_qc_*/
```

`batch_qc.py` writes `run_manifest.json` (provenance, per-case geometry/HU/laterality/volumes)
and `summary.csv`. Volumes flagged `hu_valid=False` (the 10 `radiopaedia_*` cases are 8-bit
display-scaled) must be excluded from any HU-dependent analysis.

## Phase 2: Segmentation (TotalSegmentator)

**In plain words:** a pretrained AI tool outlines the lungs, lung lobes and airways on each scan. We then compare its lung outlines with outlines drawn by experts to see how closely they match, and measure lung volumes in millilitres.

Segmentation runs in its own environment; evaluation runs in the core `.venv`.
Weights live in `models/totalsegmentator` (`TOTALSEG_HOME_DIR`, usage statistics disabled).

```bash
# segmentation env: masks + provenance -> data/derived/<case>/segmentation, runs/P2_seg_*
environments/segmentation/Scripts/python.exe scripts/segment_totalseg.py     --qc-run runs/P1_qc_20cases_<ts> --n 10          # HU-valid cases only by default

# core env: Dice / volume error vs expert masks, QC overlays -> runs/P2_eval_*
python scripts/evaluate_segmentation.py --seg-run runs/P2_seg_10cases_<ts>
```

## Phase 3: MedGemma 1.5 4B CT inference (vlm env)

**In plain words:** a medical AI model that can read images and write text looks at slices from each scan and writes a short structured description. We save exactly what it was shown and exactly what it said, so a doctor can check it later.

Requires accepting the model terms at https://huggingface.co/google/medgemma-1.5-4b-it with
the Hugging Face account whose token is used (`HF_TOKEN` env var). Weights are pinned
(revision in `configs/medgemma_ct_v1.json`) and cached in `models/medgemma/hub`.

The input representation reproduces the official notebook (pinned copy in `docs/upstream/`):
up to 85 uniformly sampled axial slices, RGB = HU windows (-1024,1024)/(-135,215)/(0,80),
JPEG, interleaved with `SLICE n` labels. Slices are reoriented from the affine to
radiological layout and ordered superior->inferior. Only HU-valid cases are eligible.
Prompt, preprocessing and generation settings are frozen in `configs/medgemma_ct_v1.json`.

```bash
# build + save the model-visible images and prompt only (no model needed)
environments/vlm/Scripts/python.exe scripts/run_medgemma.py --qc-run runs/P1_qc_20cases_<ts> --n 10 --dry-run
# inference: 1 case first, then 5
environments/vlm/Scripts/python.exe scripts/run_medgemma.py --qc-run runs/P1_qc_20cases_<ts> --n 1
```

Per case: `model_input/` (exact JPEGs), `model_input.json`, `vlm_raw.txt` (saved before
parsing), `vlm_result.json`, `vlm_parsed.json` (schema-valid only). Run: `run_manifest.json`,
`batch_results.csv`, `failures.csv`. Unparseable output is a recorded failure, never repaired.

## Phase 4: Reviewable outputs (core env)

**In plain words:** gather everything from the earlier phases into one easy-to-read report per patient (pictures, numbers and the AI's text side by side), plus a one-page summary of how the whole batch went.

Assembles finished runs into per-case standalone HTML reports (all images embedded, no
external assets), batch CSVs, a clinician reviewer sheet and a one-page feasibility summary.
Reference (expert masks), predicted (TotalSegmentator) and model observation (MedGemma) are
kept in separate sections. Every QC case gets a report; exclusions/failures are listed.

```bash
python scripts/build_reports.py --qc-run runs/P1_qc_20cases_<ts>     --seg-run runs/P2_seg_10cases_<ts> --eval-run runs/P2_eval_10cases_<ts>     --vlm-run runs/P3_vlm_10cases_<ts>
```

Output `runs/P4_report_*`: `cases/<id>/case_report.html` (+ qc.json, metrics.json, vlm_*,
model_input_sheet.png, overlays/), `batch_results.csv`, `failures.csv`,
`reviewer_template.csv`, `feasibility_summary.md`, `run_manifest.json`.

## Phase 5: Lung volumes, airway centrelines, branch lengths, tortuosity (core env)

**In plain words:** turn the outlined airways into a "tree" of thin centre lines, like a map of the branching tubes in the lungs. Then count the branches, measure how long each one is in millimetres, and how curvy it is.

**How [AirQuant](https://airquant.readthedocs.io/en/latest/) helps:** AirQuant is an open-source
research tool ([GitHub](https://github.com/ashkanpakzad/AirQuant)) built by other researchers
to measure airways in lung CT scans. Think of it as a published "recipe book": it shows one
careful, peer-reviewed way to turn an airway outline into a branch tree and measure it. We
read it to learn good ideas, such as how to define a branch and how to number branch levels
(generations). We do **not** run AirQuant, copy its code or need MATLAB. We wrote our own
version in Python, so our numbers are ours and should not be described as AirQuant results.

Our own Python implementation ("lung_ct airway tree v2": nibabel/SimpleITK geometry,
scikit-image 3D skeletonisation, Skan branch graph). It is not AirQuant and makes no claim
of equivalence; AirQuant is a methodological reference only. MATLAB is not required.
Unvalidated: no airway reference exists in this dataset, and the input airway mask is a
TotalSegmentator prediction.

Definitions (branch, pruning, contraction, root, generation, side, branch count, smoothed
length, tortuosity = length / endpoint distance) are in `src/lung_ct/airway/tree.py`.
Diameter, tapering and wall thickness are not computed.

```bash
python scripts/airway_tree.py --seg-run runs/P2_seg_10cases_<ts> \
    --eval-run runs/P2_eval_10cases_<ts> [--cases coronacases_001]
```

Parameter sensitivity (one-at-a-time grid, predeclared stability ratings):

```bash
python scripts/airway_sensitivity.py --seg-run runs/P2_seg_10cases_<ts>
```

Full reports including airway section, stability ratings and MedGemma review flags:

```bash
python scripts/build_reports.py --qc-run runs/P1_qc_20cases_<ts> \
    --seg-run runs/P2_seg_10cases_<ts> --eval-run runs/P2_eval_10cases_<ts> \
    --vlm-run runs/P3_vlm_10cases_<ts> --airway-run runs/P5_airway_10cases_<ts> \
    --sensitivity-run runs/P5_sensitivity_10cases_<ts> \
    [--vlm-compare-run runs/P3_vlm_5cases_<ts_alt_processor>]
```

Output `runs/P5_airway_*`: per case `airway_branches.csv`, `airway_summary.json`,
`airway_tree_qc.png` (centreline on CT coronal minIP slab and on the mask MIP); run
`summary.csv` (lung volumes + tree metrics), `failures.csv`, `run_manifest.json`.

## Clinician review package

**In plain words:** package the scans and results so a doctor can score them independently, first without seeing the AI's answers, then comparing them, so we get a fair human check of the AI.

### What the doctor does

The package is a set of folders sent to the doctor **one round at a time**. Everything opens
offline in a normal web browser (Chrome or Edge). Nothing has to be installed and nothing goes
online. The answers go into spreadsheets (CSV files that open in Excel).

| Round | Folder sent | Size | What the doctor does | What comes back |
|---|---|---|---|---|
| 1. Blind review | `handoff_1_blinded/` | 1.2 GB | Opens `view1_independent/index.html` and scrolls through all 10 scans, labelled R01-R10 in random order. **No AI answers, expert annotations or case names are shown.** For each scan the doctor records: can it be assessed, are opacities present, which side, where, and any other findings. | `scoring_view1_independent_<reviewer_id>_<YYYYMMDD>.csv` |
| 2. Comparison | `handoff_2_comparison/` (put next to the round 1 folder) | 194 MB | Now sees, for each case, the AI's exact question and answer, the expert infection annotation over the scan, the automated flags, and the exact 85 images the AI was shown. Marks where they agree with the AI and the type of any error (wrong finding, wrong side, claim not supported by the images, missed finding, wrong description of its own input). Round 1 answers are not edited. If the comparison changes their mind, they note it in `view1_revisions.csv`. | `scoring_view2_comparison.csv` (+ `view1_revisions.csv` if used) |
| 3. Airway check | `handoff_3_airway/` | 1.5 MB | Looks at 4 of the 10 cases (opens `central_airway_review.html`) and checks that our trachea and main-bronchus length measurements start and end at the right anatomical points, and that the centre line stays inside the airway. Answers yes / no / unsure. | `central_airway_review.csv` |

**Why rounds:** the doctor's first opinion must not be influenced by the AI. Round 2 is sent
only after the round 1 file comes back. Before that, the analyst records the file's SHA-256
fingerprint, which proves the round 1 answers were not changed afterwards.

**Sending the folders:** zip each folder and share it with a file-sharing link (e.g. Google
Drive, OneDrive, Dropbox) or a USB drive. Round 1 is too large for email. The scans are
public, anonymous research data, but choosing how to share them is the project owner's
decision. **Never send `analyst_only/`**, because it contains the key that maps R01-R10 back
to the real cases.

**Decisions still open before sending** (the doctor is told to use usual clinical judgement
and explain it in the comments until these are settled):
- What counts as upper, middle or lower zone, and central or peripheral, when describing where
  findings are.
- For round 2: whether the AI is judged against the expert infection annotation or against
  the doctor's own round 1 answers.

Each folder includes its own README with these instructions for the doctor.

### Building the package

Blinded full-CT view, comparison view, blank scoring sheets, model-input contact sheets and a
central-airway landmark sheet for all HU-valid cases (no model inference):

```bash
python scripts/build_review_package.py --report-run runs/P4_report_20cases_<ts>     --qc-run runs/P1_qc_20cases_<ts> --seg-run runs/P2_seg_10cases_<ts>     --airway-run runs/P5_airway_10cases_<ts> --sensitivity-run runs/P5_sensitivity_10cases_<ts>
```

Then apply the documentation/layout revision (staged handoffs, View 1 lock, prompt in View 2):

```bash
python scripts/revise_review_package.py --package runs/P6_clinician_review_v1_<ts>
```

Output `runs/P6_clinician_review_v1_*/`: `PACKAGE_LAYOUT.md`, `handoff_1_blinded/`,
`handoff_2_comparison/`, `handoff_3_airway/`, `analyst_only/` (never sent to reviewers). Automated flags
use the frozen exploratory rules in `configs/review_flag_rules_v1.json`.

Data (`data/`), weights (`models/`) and run artifacts (`runs/`) are git-ignored.

## Location-first checklist viewer and grading form v2 (checklist_v3_4)

Plain summary: reviewers describe findings and say where they are; they do not quote millilitres or
percentages. The page therefore shows the lobe names first and keeps every number in a measurement
panel. This is a change in presentation. It does not validate, or change, any cut-off: `checklist_v3_4`
has exactly the decision rules and measurements of `checklist_v3_3`.

What is new:

- **Per-lobe distribution rows** `E83` (ground-glass-range region), `E89` (consolidation-range region),
  `E91` (dense lung next to pleural fluid): mL, % of that lobe and % of the selected region for every
  lobe, plus the share of the region within 5 mm of a predicted fissure. Nothing is dropped.
- **Provisional lobe-naming rule** (`location_reporting` in `configs/checklist_v3_4.json`,
  `location_rule_v1_provisional`): a lobe is named when it holds at least 10% of the selected region.
  Fixed in advance, not fitted, not validated. Do not tune it on fresh evaluation cases.
- **Wording**: the region is an "HU-selected region"; for a "cannot tell" answer it is a "candidate
  region, not a confirmed finding". Lobe names only (TotalSegmentator outlines); thirds of lung height are
  geometric and are never called segments. "Lobe assignment uncertain" is flagged near fissures, outside
  lobe outlines, on poorly inflated lungs and when pleural effusion is present.
- **Viewer v2** (`--viewer v2`): location first, a "Numbers" box per answer (opens on click) and a
  measurement panel on the page; every percentage names its denominator.
- **Grading form `grades_v2`**: finding (present / absent / can't tell) and location (lobes,
  "Indeterminate", or "Not assessed" = default; pleural effusion by side) are separate fields.
  Grades saved with form v1 still load; their location is `not_recorded_form_v1`.
- **Staged page** (`--blinded`): step 1 shows the scan and the form only; the tool's answers,
  locations, measurements and outlines appear after "Save and lock step 1". Later changes are stored
  next to the locked answers with a timestamp. No radiology report in staged pages. The hidden content
  is inside the HTML file: this is a reading procedure, not a security barrier.

```bash
# 1. add the per-lobe rows to finished runs; every existing row is recomputed and must match exactly
python scripts/add_lobe_distribution.py --pages-file runs/test_logs/pages_test20b_v3.txt \
    runs/test_logs/pages_test20c_v3_1.txt runs/test_logs/pages_test20soft_v3.txt
# 2. apply checklist_v3_4 (writes checklist_checklist_v3_4.json next to the others)
python scripts/rescore_checklist.py --pages-file runs/test_logs/pages_test20c_v3_1.txt --config configs/checklist_v3_4.json
# 3. prove that no measurement and no decision changed
python scripts/verify_location_update.py --pages-file runs/test_logs/pages_test20b_v3.txt \
    runs/test_logs/pages_test20c_v3_1.txt runs/test_logs/pages_test20soft_v3.txt \
    --snapshot runs/test_logs/snapshot_before_v3_4 --lobe-verification runs/E12_lobe_distribution_<ts>
# 4. pages: location-first, and staged
python scripts/build_viewer_page.py --run runs/A1_agent_<case>_<ts> --checklist-file checklist_checklist_v3_4.json \
    --rule-based --viewer v2 --out runs/M2_location_first_20261122/viewer_<case>.html
python scripts/build_viewer_page.py --run runs/A1_agent_<case>_<ts> --checklist-file checklist_checklist_v3_4.json \
    --rule-based --viewer v2 --blinded --out runs/M2_location_first_20261122/staged_<case>.html
# 5. collect returned grades (form v1 and v2 files can be mixed)
python scripts/collect_grades.py --folder runs/grades_received
```

`build_location_first_viewers.bat` runs step 4 for the four meeting cases. Without `--viewer v2` the
page is built exactly as before (viewer v1, form v1), so `build_meeting_viewers.bat` still reproduces the
first pack. The finding-mask audit review page (`scripts/build_blinded_viewer.py`, `blinded_viewer_v2`,
record `audit_review_v2`, protocol draft `configs/audit_review_protocol_v0_1_draft.json`) also asks in
stage 1, before the lobe outlines are shown, whether the finding is there and where the reader sees it.

Other additions on the same working tree: `scripts/segment_extra_task.py` (one extra TotalSegmentator
task without touching existing outlines) and `scripts/nodule_candidates.py` (exploratory table of
`lung_nodules` outlines; no checklist rule uses it).

### Viewer v3: same page, lighter loader (`--viewer v3`)

One standalone HTML for file://, as before. The CT is stored losslessly in less space and verified in the
browser (checksum and SHA-256), payloads are decoded step by step, only one overlay is loaded at a time,
and loading / error states are shown (`scripts/viewer_v3.py`). Layout, answers, grading form and CSV are
those of viewer v2.

```bash
python scripts/build_viewer_page.py --run runs/A1_agent_<case>_<ts> --checklist-file checklist_checklist_v3_4.json \
    --rule-based --viewer v3 --out runs/M2_location_first_20261122/viewer_<case>_v3.html
# open the unmodified file from file:// in Chrome or Edge and check it (start-up state, switching, grading, memory)
python scripts/test_viewer_page.py --page runs/M2_location_first_20261122/viewer_<case>_v3.html --browser chrome
python scripts/test_viewer_page.py --page runs/M2_location_first_20261122/viewer_<case>_v3.html --browser edge --gpu
```

Tested on the development workstation only; see `docs/STATUS.md` for the before / after numbers and limits.

### Viewer v4: 2D canvas page for laptops with integrated graphics (`--viewer v4`)

No WebGL and no viewer library: one slice is drawn at a time on an ordinary canvas. Voxels are reordered to RAS+
from the affine at build time (no resampling); oblique scans are refused. Layout, answers, grading form and CSV
are those of viewer v2. Test report: `docs/VIEWER_V4_TEST_REPORT.md`. Not yet tried on the reviewer's laptop.

```bash
python scripts/build_viewer_page.py --run runs/A1_agent_<case>_<ts> --checklist-file checklist_checklist_v3_4.json \
    --rule-based --viewer v4 --out runs/M3_canvas_viewer_20261122/viewer_<case>_v4.html      # add --blinded for the staged page
python scripts/test_canvas_viewer.py --synthetic --browser edge --gpu-mode igpu               # geometry on a phantom, 48 orientations
python scripts/test_canvas_viewer.py --page runs/M3_canvas_viewer_20261122/viewer_<case>_v4.html --browser chrome --gpu-mode igpu
```

`build_canvas_viewers.bat` builds the four meeting cases. Viewer v3 (WebGL) is not a confirmed fix for
integrated graphics.

## Checklist v3_5, v3_6 and the PleThora pleural-effusion check

Since 2026-11-23 new agent runs use `configs/checklist_v3_5.json`: a classifier score is shown as supporting
evidence but can no longer make an answer "not suggested". `configs/checklist_v3_4.json` is kept for comparison
(`--checklist-config configs/checklist_v3_4.json`). To apply v3_5 to finished runs without recomputing anything:

```bash
python scripts/rescore_checklist.py --pages-file runs/test_logs/pages_58runs_with_v3_4.txt --config configs/checklist_v3_5.json
```

External check of the pleural-effusion outline against expert outlines (PleThora, public TCIA data, about 4.5 GB
on disk; results in `runs/E14_plethora_pleural_effusion_<timestamp>/REPORT.md`). Put the PleThora zips and the
spreadsheet from the TCIA page into `data/raw/plethora/download/` and unzip them into `effusions/` and `thorax/`, then:

```bash
python -I scripts/plethora_fetch.py                                           # core env: 78 effusion + 78 no-effusion scans
environments/segmentation/Scripts/python.exe scripts/plethora_segment.py     # GPU, about 30-40 s per scan
python -I scripts/plethora_compare.py                                        # core env: results.csv, summary.json
```

This is development evidence from lung-cancer planning scans; it does not validate any limit.

### checklist_v3_6 (default for new runs since 2026-11-23)

`configs/checklist_v3_6.json` = v3_5 plus two things, with every limit unchanged:

- **Dense-lung candidates are kept and tagged, not removed.** Dense lung next to predicted pleural fluid or at the back
  of the lung stays in the evidence (rows E92-E94) with a spatial tag. It is not called compressed lung, and
  atelectasis is no longer answered from it. Background: `docs/DENSE_LUNG_EVIDENCE.md`.
- **Pleural effusion wording.** "No fluid outlined" (empty mask) and "low volume outlined" (under 5 mL) are worded
  differently, neither as excluding an effusion; volumes are called estimates from predicted masks.

Replay on finished runs (adds `checklist_checklist_v3_6.json` to each run, rewrites nothing; about 3 min per scan,
use `--part i/n` for parallel workers and a final call without `--part` for the summary):

```bash
python scripts/dense_lung_inventory_replay.py --pages-file runs/test_logs/pages_58runs_with_v3_4.txt --out runs/E15_dense_lung_inventory_replay_<date>
```

`--checklist-config configs/checklist_v3_6.json`, `checklist_v3_5.json` or `checklist_v3_4.json` reproduce the earlier behaviour.

### checklist_v3_7 (default for new runs since 2026-11-23) and the empty-mask root-cause check (E16)

`configs/checklist_v3_7.json` = v3_6 plus one correction, with every limit unchanged: a predicted pleural-fluid mask
below the reporting limit (empty, or under 5 mL) gives "cannot tell", never "absent". The answer of the unchanged
20 mL / 5 mL rule is stored as `historical_rule_answer`. All numeric limits are provisional reporting limits of a
development tool, not validated diagnostic boundaries.

```bash
# re-score finished runs (adds checklist_checklist_v3_7.json, rewrites nothing)
python scripts/rescore_with_inventory_rows.py --pages-file runs/test_logs/pages_58runs_with_v3_4.txt \
    --rows runs/E15_dense_lung_inventory_replay_20261123 --config configs/checklist_v3_7.json \
    --compare configs/checklist_v3_6.json --out runs/E16_empty_mask_root_cause_20261123/rescore_v3_7.json
```

Why the model returns an empty mask on some effusion scans (`runs/E16_empty_mask_root_cause_20261123/REPORT.md`):

```bash
OUT=runs/E16_empty_mask_root_cause_20261123
IDS="LUNG1-013 LUNG1-113 LUNG1-170 LUNG1-195 LUNG1-249 LUNG1-299 LUNG1-016 LUNG1-285 LUNG1-301 LUNG1-051"
python -I scripts/plethora_source_audit.py --out $OUT/source_audit.json $IDS                      # DICOM against NIfTI
environments/segmentation/Scripts/python.exe scripts/plethora_empty_mask_trace.py $IDS             # GPU, about 1 min per scan
python scripts/plethora_empty_mask_report.py --out $OUT $IDS                                       # table and overlays
environments/segmentation/Scripts/python.exe scripts/plethora_counterfactual_density.py --out $OUT/counterfactual_density.json $IDS
python -I scripts/ts_effusion_training_labels.py --out $OUT/training_labels                        # labels only, about 90 MB
python -I scripts/ts_effusion_training_density.py --attrs $OUT/training_labels/training_label_attributes.csv --out $OUT/training_labels
```

Diagnostic only: no model is trained and no filter or limit is added.

### checklist_v3_8 (default for new runs since 2026-11-23), exact stage audit, comparison plan

`configs/checklist_v3_8.json` = v3_7 plus one wording change: a positive pleural-effusion answer that rests only on
the predicted mask is headlined "Pleural effusion suggested by segmentation". The answer of the unchanged 20 mL /
5 mL rule is stored separately as `provisional_threshold_result`. `--checklist-config configs/checklist_v3_7.json`
reproduces the earlier behaviour.

```bash
OUT=runs/E16_empty_mask_root_cause_20261123
python scripts/rescore_with_inventory_rows.py --pages-file runs/test_logs/pages_58runs_with_v3_4.txt \
    --rows runs/E15_dense_lung_inventory_replay_20261123 --config configs/checklist_v3_8.json \
    --compare configs/checklist_v3_7.json --out $OUT/rescore_v3_8.json
# exact voxel counts, grids and resampling calls at every stage (needs the trace above; GPU, about 30 s per scan)
environments/segmentation/Scripts/python.exe scripts/plethora_stage_audit.py --out $OUT/stage_audit $IDS
python scripts/plethora_alignment_check.py --out $OUT/alignment_by_reviewer_file.csv   # every reviewer file, from image content
```

The PleThora reference is the `first_reviewer` file (see the E16 report, section 3). Plan for comparing the current
model with one alternative, not started: `docs/PLEURAL_MODEL_COMPARISON_PLAN.md`.

### Export, scoring and comparison preparation (E17, 2026-11-23)

The v3_8 headline survives export and scoring: `lung_ct.agent.checklist.assessment_category()` gives the current
assessment (`present`, `suggested`, `absent`, `cannot tell`, `cannot assess`). "Suggested" (a positive answer from a
predicted mask only) is never counted as a confirmed finding, and the provisional threshold result is never scored.

```bash
PREP=runs/E17_pleural_model_comparison_prep_20261123
python scripts/export_checklist_csv.py --pages-file runs/test_logs/pages_58runs_with_v3_4.txt     --checklist-file checklist_checklist_v3_8.json --out runs/E16_empty_mask_root_cause_20261123/checklist_v3_8_export.csv
python scripts/score_checklist.py --pages-file runs/test_logs/pages_58runs_with_v3_4.txt     --checklist-file checklist_checklist_v3_8.json --label v3_8_dev58
python -I scripts/pleural_armB_inspect.py --out $PREP                # alternative model: file list and plans, no weights
python scripts/pleural_compare_metrics.py --manifest $PREP/pilot_manifest_armA.json --out $PREP/pilot   # current model, 14 pilot scans
# alternative model (GPU, separate environment environments/pleural_v1; a version separation, not a security sandbox)
environments/pleural_v1/Scripts/python.exe scripts/pleural_armB_run.py --manifest $PREP/pilot_manifest_armB.json --out $PREP/pilot
environments/pleural_v1/Scripts/python.exe scripts/pleural_armB_run.py --manifest $PREP/full/manifest_armB.json --out $PREP/full
# LUNG1-104 only: save and score the mask although its "lung" label is 18 L (flagged, not hidden)
environments/pleural_v1/Scripts/python.exe scripts/pleural_armB_run.py --manifest $PREP/full/rerun_LUNG1-104/manifest_armB.json --out $PREP/full/rerun_LUNG1-104 --lung-guard warn
python scripts/pleural_compare_metrics.py --manifest $PREP/full/manifest_armA.json --out $PREP/full
python scripts/pleural_compare_metrics.py --manifest $PREP/full/manifest_armB.json --out $PREP/full
python scripts/pleural_armB_qc.py --manifest $PREP/full/manifest_armB.json --armA-manifest $PREP/full/manifest_armA.json --run-log $PREP/full/run_log_B_task307_pleuralfinal.json --out $PREP/full
python scripts/pleural_compare_report.py --a $PREP/full/metrics_A_totalsegmentator_315.csv --b $PREP/full/metrics_B_task307_pleuralfinal.csv --manifest $PREP/full/manifest_armB.json --out $PREP/full
python scripts/pleural_armB_outside_chest.py --manifest $PREP/full/manifest_armB.json --armA-manifest $PREP/full/manifest_armA.json --out $PREP/full   # descriptive only
python scripts/pleural_armB_class_review.py --manifest $PREP/full/manifest_armB.json --armA-manifest $PREP/full/manifest_armA.json --qc $PREP/full/qc_armB.csv --out $PREP/full   # per-scan review of class-check warnings
python scripts/pleural_compare_freeze.py --run $PREP    # freeze record: hashes, model identity, environments, count verification
```

Result (development data, 213 scans; `runs/E17_pleural_model_comparison_prep_20261123/REPORT.md`; plain summary in
`docs/PLEURAL_EFFUSION_LEARNINGS.md`): the comparison of the two pipelines as implemented here does not support
replacing the current pipeline. Against the PleThora expert outlines the median Dice is 0.376 (current) and 0.414
(alternative); 6 scans improved, 4 worsened, 67 were similar. The alternative recovers none of the six empty-mask
misses, and on long scans it marks tissue outside the chest. This does not show the models are equivalent and does not
validate the current model. The comparison is closed; no further model runs without a new experiment brief.

