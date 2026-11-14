# Lung CT Research

Local, reproducible research pipeline for public 3D chest CT: ingestion, geometry/HU QC,
lung segmentation, volume quantification, CT-capable LVLM inference, reviewable reports.
Research use only - no diagnostic or clinical validation claims. See `PROJECT_GUIDELINES.md` and `docs/STATUS.md`.

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
