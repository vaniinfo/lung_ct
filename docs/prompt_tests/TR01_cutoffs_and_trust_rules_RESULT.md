# TR01: COLIPRI cut-offs and trust rules across three answer sources - RESULT (2026-11-26)

Plan: [TR01_cutoffs_and_trust_rules_PLAN.md](TR01_cutoffs_and_trust_rules_PLAN.md); plan unchanged since freezing: yes.

- Run folder: `runs/TR01_trust_rules_20261126_190105` (`summary.md`, `summary.json`, `cutoffs.csv`, `source_table.csv`)
- Inputs: NV-Reason-CT set A of `runs/PT02_nvreason_prompts_20261126_134837` (83 scans); COLIPRI scores of
  `runs/PT01_colipri_prompts_20261126_073924` (85 scans); the tool's own answers in `checklist_checklist_v3_9.json` of the latest agent run (58 scans).
- Reference: CT-RATE report labels. All scans are development data. Nothing is applied by this test.

## Result, part 2 (trust rules)

**Almost no source meets the bars. Of 52 source-condition pairs, none may decide "present" and one may decide "absent".**

| Bar | Sources that meet it |
|---|---|
| "Present" may decide (10 or more present calls, 90 % right, lower end above 75 %) | none |
| "Absent" may decide (10 or more positive scans, 90 % found with lower end above 75 %, 90 % of absent answers right) | COLIPRI with cut-off, for consolidation only (found 16 of 17) |

Closest to the "present" bar:

| Condition | Source | Present calls | Right | Why it fails |
|---|---|---|---|---|
| Pleural effusion | NV-Reason-CT | 9 | 9 (100 %) | one call short of 10 |
| Pleural effusion | COLIPRI (cut-off) | 10 | 9 (90 %) | lower end 67 % |
| Pleural effusion | Tool | 6 | 6 (100 %) | too few calls |
| Consolidation | NV-Reason-CT | 15 | 13 (87 %) | below 90 % |
| Consolidation | Tool | 5 | 5 (100 %) | too few calls |
| Mosaic attenuation | NV-Reason-CT | 13 | 11 (85 %) | below 90 % |
| Pericardial effusion | COLIPRI (cut-off) | 12 | 10 (83 %) | below 90 % |

The full table (every source and condition) is in `summary.md` of the run folder.

**Proposed trust table:** COLIPRI's "absent" could decide for consolidation; everything else stays as today.
On the 986 tool answers this would change 2 (both to "absent", both equal to the report label). The table was chosen on
these same scans, so that count describes the proposal and is not a measure of its quality.

## The three conditions where checklist_v3_11 lets NV-Reason-CT decide "present"

| Condition | Positive scans | Present calls | Labelled present | Meets the bar |
|---|---|---|---|---|
| Medical material | 6 | 9 | 5 (56 %) | no |
| Hiatal hernia | 4 | 8 | 4 (50 %) | no |
| Bronchiectasis | 14 | 10 | 7 (70 %) | no |

Against report labels, about half of these "present" answers are on scans whose report does not mention the finding.
That can mean the model is wrong or that the report left the finding out; the labels cannot tell these apart.
checklist_v3_11 was based on the clinician reviewer's grades on four scans, which is a different and much smaller reference.
This table is a reason to have that policy looked at again, not a verdict on it.

## Result, part 1 (COLIPRI cut-offs)

Each patient was answered with a cut-off fitted without that patient; the four clinician-graded scans never moved a cut-off.

| Condition | Positive scans | Proposed cut-off | Range over folds | Found | Missed | Extra | Balanced accuracy |
|---|---|---|---|---|---|---|---|
| Consolidation | 17 | 0.853 | 0.573 to 0.876 | 16 | 1 | 6 | 0.93 |
| Interlobular septal thickening | 14 | 0.986 | 0.977 to 0.987 | 12 | 2 | 5 | 0.89 |
| Pericardial effusion | 13 | 0.872 | 0.783 to 0.885 | 10 | 3 | 2 | 0.87 |
| Pleural effusion | 12 | 0.270 | 0.252 to 0.683 | 9 | 3 | 1 | 0.87 |
| Cardiomegaly | 12 | 0.087 | 0.078 to 0.295 | 10 | 2 | 12 | 0.83 |
| Atelectasis | 15 | 0.590 | 0.556 to 0.718 | 12 | 3 | 14 | 0.80 |
| Lung opacity | 16 | 0.192 | 0.179 to 0.994 | 12 | 4 | 16 | 0.76 |
| Lymphadenopathy | 18 | 0.009 | 0.008 to 0.059 | 14 | 4 | 20 | 0.74 |
| Mosaic attenuation | 12 | 0.205 | 0.186 to 0.884 | 10 | 2 | 26 | 0.74 |
| Fibrotic sequela | 23 | 0.354 | 0.353 to 0.754 | 18 | 5 | 19 | 0.74 |
| Bronchiectasis | 14 | 0.993 | 0.988 to 0.999 | 9 | 5 | 13 | 0.73 |
| Emphysema | 16 | 0.993 | 0.970 to 0.998 | 8 | 8 | 10 | 0.68 |
| Lung nodule | 34 | 0.970 | 0.970 to 0.972 | 20 | 14 | 14 | 0.66 |

Five conditions with fewer than 10 positive scans (arterial and coronary calcification, hiatal hernia, medical material,
peribronchial thickening) are in `cutoffs.csv`, marked "too few to judge".

- The cut-offs differ widely between conditions (0.009 to 0.993), so one common cut-off such as 0.5 would not work.
- Several cut-offs are unstable: for lung opacity, mosaic attenuation, fibrotic sequela and pleural effusion the value moves
  a long way when one patient is left out.
- Next to NV-Reason-CT's set A answers (83 of the same scans), COLIPRI with a cut-off is not clearly better or worse: higher balanced
  accuracy for consolidation and septal thickening, lower for mosaic attenuation, lung nodule and emphysema, close elsewhere.
  It gives more "extra" answers for most conditions.

## Against the clinician reviewer's grades (4 scans, descriptive only)

| Source | Graded answers | Same | Present found | Present missed | Absent called present | Neither |
|---|---|---|---|---|---|---|
| NV-Reason-CT | 68 | 57 | 20 | 9 | 2 | 0 |
| COLIPRI (cut-off) | 68 | 48 | 18 | 11 | 9 | 0 |
| Tool (checklist_v3_9) | 68 | 12 | 4 | 25 | 0 | 56 |

The tool answers "cannot tell" on most conditions ("Neither"), which is why it agrees on few and never calls an absent finding present.

## Decision

No trust table is adopted. The bars were set before looking and the sources do not reach them on these scans.
The cut-offs are proposals for display or later testing, not validated thresholds.

## What this does and does not show

- It shows that on about 60 to 85 development scans, with report labels as reference, no source is right often enough for its
  "present" to decide alone, and only one source-condition pair is for "absent".
- It does not show that the sources are useless: several rank findings well (COLIPRI AUROC 0.89 in PT01), and report labels
  count a correct finding the report did not mention as an error. With about 10 to 20 positive scans per condition the lower
  end of the interval is hard to get above 75 % even for a good source.
- MedGemma and Hulu-Med answered only 20 scans and were not judged.
- The tool's answers come from a 58-scan agent run with checklist_v3_9; later checklist versions were not re-run on all scans.

## Deviations from the plan

None. The one change to the "absent" bar was made before freezing and is recorded in the plan.
