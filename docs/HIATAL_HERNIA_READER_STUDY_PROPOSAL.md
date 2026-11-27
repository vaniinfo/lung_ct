# Hiatal hernia: reader-labelled feasibility study (proposal)

Date: 2026-11-24. Status: **proposal, not started.** No page has been built and no reader has seen the sample.
Research use only. Background: `runs/E21_ctclip_pilot_20261124/REPORT.md`.

## 1. Why

We cannot say how well any model finds hiatal hernia, because we have almost no trustworthy labels: three patients
whose report mentions it and four scans graded by one reader on pages that showed the tool's answers. A report that does
not mention a hernia does not show it is absent. This study asks a smaller question first.

## 2. Questions this study can answer

1. Can readers label hiatal hernia on these chest CTs reliably (how often assessable, how often uncertain)?
2. How often do two readers agree?
3. How many confirmed-present and confirmed-absent patients does a sample of this kind yield?
4. Do report labels miss hernias that readers see?

It will **not** give model sensitivity, specificity or calibration. Whether those can be estimated later depends on the
numbers found in question 3 (section 9).

## 3. Sample (draft table exists; not issued)

Unit: **patient**, one reconstruction each. Draft: `runs/E22_hh_reader_feasibility_DRAFT/` (seed 20261124).

| Stratum | Who | Patients in frame | Selected | Sampling probability |
|---|---|---|---|---|
| C | report mentions hiatal hernia | 2 | 2 | 1 |
| H | 12 highest composite model scores among the rest | 12 | 12 | 1 |
| M | next 20 | 20 | 8 | 0.40 |
| L | all others (low scores) | 32 | 8 | 0.25 |
| Total | | 66 | **30** | |

- **Frame:** the 70 local CT-RATE patients with stored scores, minus the four already graded on open pages
  (valid_301, 541, 832, 988). Those four cannot be read blind again by the same reader.
- **Composite score:** mean of the patient's percentile ranks in COLIPRI, CT-CLIP zero-shot, VocabFine and ClassFine.
  It is used only to spread the sample across the score range. It is not a rule and not a threshold.
- **Low-score patients are included on purpose** (8 of 32), so that hernias the models miss can be found.
- **The sample is enriched** for high scores. Every estimate for the frame must weight a patient by
  1 / sampling probability (1, 1, 2.5, 4). The table of strata and probabilities is for the analyst only.
- No new scan is downloaded and no model is run for this study.

## 4. What readers record, per patient

| Field | Choices |
|---|---|
| Assessability | assessable / limited / not assessable, with reason (hiatus not in the scan, motion, other) |
| Hiatal hernia | present / absent / uncertain |
| If present: type (optional) | sliding / paraesophageal / cannot tell |
| If present: size (optional) | small / moderate / large (reader's own judgement; not a measurement) |
| Confidence | sure / fairly sure / unsure |
| Comment | free text |

"Uncertain" and "not assessable" are valid answers and are kept as such. They are never recoded as absent.

## 5. Blinding and reading procedure

- Readers see the CT only: axial, coronal and sagittal views, soft-tissue and lung windows, no outlines.
- Hidden from readers: all model scores, the stratum, the radiology report, the report label, the other reader's answers.
- Cases are shown in a shuffled order that carries no information about the stratum (`reading_list_FOR_READERS.csv`).
- Each reader's first answer is locked and time-stamped before anything else is shown. The existing staged-page
  mechanism (step 1 locked before reveal) is reused; nothing is revealed in this study.
- **Two readers** read all 30 independently. Disagreements (including present versus uncertain) are resolved in a
  consensus session, or by a third reader if available. Both the independent answers and the resolved answer are stored.
- If only one reader is available: the study still answers questions 1, 3 and 4; question 2 is dropped and stated as such.

## 6. Analysis (descriptive; fixed before reading starts)

- Counts of assessable / limited / not assessable; present / absent / uncertain, per reader and after resolution.
- Agreement between readers: raw percentage and a 3 x 3 table. Kappa is reported with its interval and the warning
  that 30 patients give a wide one.
- Reader result against report label, as a table. No accuracy figure for the report.
- Model scores by resolved reader status, as a table and a plot, with the sampling weights. No threshold, no
  sensitivity, specificity or calibration.
- Whether model scores still follow the shared score component among reader-confirmed patients.
- Everything is reported per patient. The four open-page patients are shown separately and never pooled.

## 7. What it costs

| Item | Estimate |
|---|---|
| Reader time | 30 patients per reader; reading time per case to be measured on the first 5 (not assumed here) |
| Our preparation | 30 blinded pages from scans already on disk; grading form with the fields above; import check |
| Data and models | none new |

## 8. Fresh patients reserved for later evaluation

- The CT-RATE validation split has 1,304 patients (3,039 volumes); the report-derived hiatal hernia label is set for
  193 of them. 70 patients are local.
- **Reserved:** the 1,234 validation patients that are not local. None of their scans is downloaded, scored or viewed
  until an evaluation protocol is frozen (models, analysis, sample size, sampling probabilities).
- A later evaluation draws from the reserved patients only, by patient, with a recorded seed, and reads them blind.
- The 70 local patients, including the 36 not selected here, stay development data. They are never used as a test set.
- Known limit of the reserve: it is the same public split the model authors report on, so it is fresh for us but not
  guaranteed unseen in model selection. A later protocol should say so, or use a different source if one is available.
- A later download needs a storage and data-budget decision; the raw-data budget is currently used up.

## 9. Calibration and performance: what we do not promise

No calibration estimate is planned. It will be considered only if the reader study and any later sample together give
enough reader-confirmed positive and negative patients, drawn with known probabilities, to support it. The feasibility
study reports those counts so that the decision can be made on numbers. As an order of magnitude only: the three local
report-labelled patients among 70 suggest that confirmed positives will be few unless sampling is enriched further.

## 10. Rules carried over

- COLIPRI stays in the pipeline; no CT-CLIP variant is adopted for hiatal hernia.
- No hiatal-hernia threshold; no score tuning; no within-patient ranking rule.
- No further model runs aimed at the four familiar cases.
- CT images, masks and reports stay on this machine.

## 11. Decisions needed

1. Approve the sample of 30 patients as drafted, or change its size.
2. Name the readers (one or two) and how disagreements are resolved.
3. Confirm the fields in section 4 with the clinician reviewer before the form is built.
