# PT01: COLIPRI - prompt wording for the finding score - RESULT (2026-11-26)

Plan: [PT01_colipri_prompts_PLAN.md](PT01_colipri_prompts_PLAN.md); plan unchanged since freezing: yes.

- Run folder: `runs/PT01_colipri_prompts_20261126_073924`
- Prompt sets: `configs/colipri_prompt_sets_v1.json`, SHA-256 `0791b1361fdcc5f2...` (full value in the run manifest)
- Scans run / failed: 85 / 0. Patients: 70.
- Check of the set-up: set A scores equal the classifier scores stored by earlier pipeline runs (largest difference 0.0001, rounding).

## Result

**No prompt set met the pre-stated rule. Set A, the current short prompts, stays.**

| Prompt set | Idea | Macro AUROC | Difference to A | 95 % interval |
|---|---|---|---|---|
| A (baseline) | "X present" / "No X present" | 0.886 | | |
| B | Four sentence forms averaged | 0.888 | +0.002 | -0.007 to +0.015 |
| C | Several names averaged | 0.886 | 0.000 | -0.006 to +0.006 |
| D | 50 + 50 real report passages ("native") | 0.869 | -0.017 | -0.041 to +0.007 |

AUROC: 0.5 = coin flip, 1 = perfect ranking. Average precision points the same way (A 0.670, B 0.663, C 0.665, D 0.611).

| Condition | Positive scans | A | B | C | D | Note |
|---|---|---|---|---|---|---|
| Lung opacity | 16 | 0.84 | 0.84 | 0.84 | 0.87 | |
| Consolidation | 17 | 0.97 | 0.97 | 0.97 | 0.98 | |
| Atelectasis | 15 | 0.87 | 0.87 | 0.86 | 0.64 | D much worse |
| Lung nodule | 34 | 0.69 | 0.68 | 0.69 | 0.69 | weakest condition in every set |
| Emphysema | 16 | 0.83 | 0.85 | 0.83 | 0.83 | |
| Bronchiectasis | 14 | 0.90 | 0.88 | 0.88 | 0.88 | |
| Interlobular septal thickening | 14 | 0.95 | 0.96 | 0.95 | 0.94 | |
| Peribronchial thickening | 5 | 0.89 | 0.88 | 0.90 | 0.86 | too few to interpret |
| Mosaic attenuation | 12 | 0.89 | 0.90 | 0.89 | 0.87 | |
| Fibrotic sequela | 23 | 0.78 | 0.81 | 0.82 | 0.80 | |
| Pleural effusion | 12 | 0.99 | 0.99 | 0.99 | 0.97 | |
| Pericardial effusion | 13 | 0.94 | 0.96 | 0.94 | 0.94 | |
| Cardiomegaly | 12 | 0.96 | 0.96 | 0.95 | 0.93 | |
| Lymphadenopathy | 18 | 0.87 | 0.88 | 0.87 | 0.92 | |
| Arterial wall calcification | 9 | 0.99 | 0.99 | 0.98 | 0.96 | too few to interpret |
| Coronary artery wall calcification | 8 | 0.93 | 0.93 | 0.92 | 0.95 | too few to interpret |
| Hiatal hernia | 4 | 0.83 | 0.84 | 0.81 | 0.72 | too few to interpret |
| Medical material | 6 | 0.83 | 0.84 | 0.85 | 0.90 | too few to interpret |

## Rule for "better": not met

- B and C: the interval of the difference includes zero. Rewording and synonyms changed the ranking very little.
- D: the interval includes zero and one condition with 10 or more positive scans (atelectasis) dropped by more than 0.05 (0.87 to 0.64).

## Decision

Set A stays as the prompt set for the classifier score. No confirmation run is needed, because nothing is being changed.

## What this does and does not show

- It shows that, on these 85 scans, the three alternatives do not rank better than the short prompts. In particular the "native"
  prompts, which were better in the COLIPRI paper, were not better here.
- It does not show how good COLIPRI is. The 0.886 is on development scans chosen for this project, against report-mention labels;
  it is not comparable with the 0.70 the paper reports for short prompts on the full CT-RATE test set.
- It says nothing reliable about the five conditions with fewer than 10 positive scans, including hiatal hernia and medical material.
- Set D differs from the paper in one way: its passages come from the validation reports of other patients (the training reports
  are not held locally). 23 of 1,800 passages were cut at the text encoder's 512-token limit.

## What was not tested

Other sentence templates, prompts in a longer report style written by hand, more or fewer than 50 passages, a trained probe on the embeddings,
and any scan from another hospital.

## Scan sets that are development data from now on

All 85 local CT-RATE validation scans were already development data and remain so. No untouched scan was used.
