# TR01: COLIPRI cut-offs and trust rules across three answer sources - PLAN (frozen 2026-11-26, before any of these figures was computed)

Same principle as [PROMPT_TEST_TEMPLATE.md](../PROMPT_TEST_TEMPLATE.md): everything is decided before looking.
This is not a prompt test; it uses answers that already exist or that PT02 produces.

## 1. Questions

- **Part 1 (cut-offs):** if the COLIPRI score had to give a yes / no answer, which cut-off per condition would be used, and how good are those answers?
- **Part 2 (trust rules):** for each condition, which of our answer sources is right often enough that its "present" (or its "absent") could decide the answer?

**Decision it informs:** a proposal for a trust table (who may decide present, who may decide absent, per condition).
Nothing is applied by this test. A trust table would be a new checklist version with its own review.

## 2. Sources

| Source | Answer | Scans |
|---|---|---|
| NV-Reason-CT | Yes / No, set A (current wording) of PT02 | 83 |
| COLIPRI | score from `runs/PT01_colipri_prompts_20261126_073924` (set A), turned into yes / no in part 1 | 85 |
| The tool's own answer | present / absent / cannot tell in `checklist_checklist_v3_9.json` of the latest agent run | 58 |

MedGemma and Hulu-Med answered only 20 scans. They are listed in the result but cannot reach the numbers the rule asks for.
The tool's answer rests on TotalSegmentator measurements; "cannot tell" is neither present nor absent and is counted separately.

## 3. Scans and reference

- The local CT-RATE validation scans; all development data. Reference: CT-RATE report labels ("not mentioned" is not "absent").
- Conditions: those with a CT-RATE label that the source answers. Conditions with fewer than 10 positive scans are listed and marked "too few".
- Separate, descriptive only: the clinician reviewer's grades on four scans.

## 4. Part 1: cut-offs

- Cut-off per condition = the score value that maximises (share found + share correctly absent - 1) on the fitting scans;
  candidates are the midpoints between neighbouring score values; on a tie the lowest candidate is taken.
- **Honest answers by leaving patients out:** each patient's scans are answered with a cut-off fitted on all other patients.
  The figures reported are from these left-out answers only.
- The four clinician-graded scans' patients are never used to fit a cut-off (standing rule: no thresholds tuned on reported cases).
  They receive left-out answers like everyone else.
- Reported per condition: found / missed / extra / correctly absent, balanced accuracy, and how much the cut-off moves between folds.
- Stated in advance: a cut-off fitted on about 70 patients is unstable for rare conditions; it is a proposal, not a validated threshold.

## 5. Part 2: trust rule

For one source and one condition, on the scans that source answered:

- **Its "present" may decide** if it said present on at least 10 scans, at least 90 % of those are labelled present,
  and the lower end of the 95 % interval is above 75 %.
- **Its "absent" may decide** if there are at least 10 labelled-present scans, it found at least 90 % of them
  (lower end of the interval above 75 %), and at least 90 % of its "absent" answers are on unlabelled scans.

Changed from the first sketch of this rule, before any figure was computed: the "absent" rule now asks that the source finds
at least 90 % of the labelled scans. Without that, a source that answers "absent" to everything would pass for every rare condition.

- Interval: bootstrap over patients, 2000 resamples, seed 20261126.
- Where two sources that may decide disagree on a scan, the answer is "cannot tell".
- Where no source may decide, the answer stays as today.

## 6. What is reported

- The table source x condition: present calls, share right, interval, share found, interval, passes "present" / passes "absent".
- The proposed trust table, and how many of today's answers it would change on these scans. Because the table is chosen on the same scans,
  that count describes the proposal and is not a measure of its quality.
- The three conditions where checklist_v3_11 already lets NV-Reason-CT decide "present" (medical material, hiatal hernia, bronchiectasis),
  shown against the rule.
- The four clinician-graded scans: each source's answers against the grades.

## 7. Run

```
.venv/Scripts/python.exe scripts/trust_rules.py --pt02-run runs/PT02_nvreason_prompts_<time> --colipri-run runs/PT01_colipri_prompts_20261126_073924
```

Output: `runs/TR01_trust_rules_<time>`; result file `docs/prompt_tests/TR01_cutoffs_and_trust_rules_RESULT.md`.

## 8. Claims

- **May claim:** which sources meet the stated bars on these development scans, and which do not.
- **May not claim:** that any source is validated, or a performance figure. Report labels penalise a source for correct findings the report did not mention.

## 9. Confirmation

A trust table would need one run on untouched scans before use. That needs new scans and a decision on the data budget.

Changes after freezing: none
