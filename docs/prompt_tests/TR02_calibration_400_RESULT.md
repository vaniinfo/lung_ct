# TR02: cut-offs and trust rules on 400 new CT-RATE scans - RESULT (2026-11-27)

Plan: [TR02_calibration_400_PLAN.md](TR02_calibration_400_PLAN.md); plan unchanged since freezing: yes.

- Data: `D:/datasets/ct_rate/calib400` (400 scans downloaded, 0 failed, 89.6 GB including the original downloads).
- Checks: 400 HU-valid; left / right and orientation verified on 395, 5 left out (3 inconsistent organ positions, 2 front/back or top/bottom inconsistent).
- Halves: 198 fit scans, 197 confirm scans; no patient in both.
- Runs: `runs/P1_qc_ctrate_calib400_20261126_214335` (download / HU), `runs/P1_qc_verified_P1_qc_ctrate_calib400_20261126_214335_010411` (left / right),
  `runs/TR02_nvreason_calib400_20261127_020852` (NV-Reason-CT set A, 0 unreadable; 60 answers also asked the ordinary way, all identical),
  `runs/TR02_colipri_calib400_20261127_051541` (COLIPRI set A, 0 failed), `runs/TR02_calibration_20261127_053515` (scoring; full tables in `summary.md`).
- Reference: CT-RATE report labels. Nothing is applied.

## Result 1: what met the bars in both halves

**No source may decide "present" for any condition.** Three "absent" entries are confirmed:

| Condition | Source | Found, fit / confirm | "Absent" answers right, fit / confirm |
|---|---|---|---|
| Consolidation | COLIPRI (cut-off) | 91 % / 95 % | 98 % / 98 % |
| Coronary artery wall calcification | COLIPRI (cut-off) | 92 % / 90 % | 97 % / 96 % |
| Coronary artery wall calcification | NV-Reason-CT | 94 % / 90 % | 98 % / 97 % |

Met the bars in one half only (not confirmed): NV-Reason-CT "absent" for arterial calcification, cardiomegaly, consolidation, medical material;
COLIPRI "absent" for arterial calcification and cardiomegaly; COLIPRI "present" for pleural effusion (15 of 15 right in the fit half, 10 of 13 in the confirm half).

## Result 2: TR01's proposals on scans it never saw

- COLIPRI "absent" for consolidation with TR01's cut-off (0.853): found 86 % (lower end 78 %): **does not meet the bar**.
  With the cut-off refitted on 198 scans (0.558) it does, in both halves. The 83-scan cut-off was too high.
- TR01's cut-offs on all new scans give balanced accuracy close to the newly fitted ones for most conditions, but very different values for
  cardiomegaly (0.087 vs 0.716), lung opacity (0.19 vs 0.99), mosaic attenuation (0.20 vs 0.89) and fibrotic sequela (0.35 vs 0.98): those 83-scan cut-offs were unreliable.

## Result 3: COLIPRI cut-offs, fit half against confirm half

The cut-offs barely move between folds of the fit half, but several lose a lot on the confirm half:

| Condition | Balanced accuracy, fit (left-out) | Confirm (frozen cut-off) |
|---|---|---|
| Interlobular septal thickening | 0.80 | 0.49 |
| Medical material | 0.87 | 0.66 |
| Pericardial effusion | 0.84 | 0.73 |
| Atelectasis | 0.76 | 0.67 |
| Cardiomegaly | 0.92 | 0.84 |
| Pleural effusion | 0.94 | 0.85 |

Held up or improved: arterial calcification 0.85 / 0.89, coronary calcification 0.89 / 0.90, consolidation 0.82 / 0.85, peribronchial thickening 0.77 / 0.75.
Weakest everywhere: lung nodule (0.61 / 0.64) and fibrotic sequela (0.60 / 0.64).

## Result 4: the three checklist_v3_11 conditions (all 395 scans)

| Condition | NV-Reason-CT "present" calls | Report agrees | Lower end |
|---|---|---|---|
| Medical material | 41 | 27 (66 %) | 51 % |
| Hiatal hernia | 59 | 35 (59 %) | 46 % |
| Bronchiectasis | 50 | 20 (40 %) | 27 % |

On 395 scans the report mentions the finding for 40 % to 66 % of the model's "present" calls; for bronchiectasis fewer than half.

## Decision

- No trust table is adopted by this test. The three confirmed "absent" entries are a proposal for a new checklist version with its own review.
- The TR01 cut-offs and TR01's one entry are withdrawn; if cut-offs are used, they are the fit-half values of this test.
- The confirm half is now used and cannot confirm anything again.

## What this does and does not show

- It shows which entries met pre-stated bars on two separate halves of about 200 patients from one hospital, against report labels.
- It does not validate any source. Report labels count a correct finding the report did not mention as an error: this weighs most on the "present" bar
  and on findings reports often leave out (medical material, hiatal hernia, small calcifications). Both models were trained on reports from this hospital.
- "Absent may decide" means: when the source says absent, the report almost never mentions the finding, and the source catches at least 90 % of the scans whose report does.
- 5 of 400 scans failed the orientation check and were not answered; that is reported, not hidden.

## Deviations from the plan

None.
