# PT02: NV-Reason-CT - question wording, reasoning, report-first, location-first, voting - RESULT (2026-11-26)

Plan: [PT02_nvreason_prompts_PLAN.md](PT02_nvreason_prompts_PLAN.md); plan unchanged since freezing: yes.

- Run folder: `runs/PT02_nvreason_prompts_20261126_134837` (`summary.md`, `summary.json`, `per_condition.csv`, `answers.jsonl`)
- Prompt sets: `configs/nvreason_prompt_sets_v1.json`, SHA-256 `0c6db88ed28394ff...` (full value in the run manifest)
- Scans run / failed: 83 / 0. Patients: 68. Answers: 10,541. Unreadable answers: 0 in every set.
- Check of the set-up: on 5 scans, 3 questions each were also asked the ordinary way; all 15 answer texts were identical.
- 17 conditions scored (ground-glass has no label); 12 have 10 or more positive scans and enter the main measure.

## Result, experiment 1 (way of asking)

**No set met the pre-stated rule. Set A, the current wording, stays.**

| Set | Idea | Macro balanced accuracy | Difference to A | 95 % interval | Found | Missed | Extra |
|---|---|---|---|---|---|---|---|
| A (baseline) | Current wording | 0.814 | | | 163 | 59 | 97 |
| V | Vote of three wordings | 0.812 | -0.001 | -0.005 to +0.002 | 163 | 59 | 99 |
| B | Reasoning on | 0.814 | 0.000 | -0.002 to +0.002 | 163 | 59 | 98 |
| C | Definition in the question | 0.810 | -0.004 | -0.013 to +0.005 | 161 | 61 | 95 |
| D | Report first | 0.814 | +0.001 | -0.024 to +0.029 | 162 | 60 | 81 |
| E | Location first | 0.828 | +0.014 | -0.019 to +0.052 | 184 | 38 | 184 |

Balanced accuracy: 0.5 = coin flip, 1 = perfect. Found / missed / extra are totals over the 17 scored conditions.
The two rewordings on their own: A2 0.810, A3 0.814.

## Rule for "better": not met

- **V, B, C:** the interval of the difference includes zero. The answers are almost the same as with set A.
  Of 1,411 answers, reasoning (B) changed 3, the vote (V) 6 and the definition (C) 12; report first (D) changed 69 and location first (E) 118.
- **D (report first):** the interval includes zero, and two conditions with 10 or more positive scans dropped by more than 0.05
  (mosaic attenuation 0.94 to 0.79, pericardial effusion 0.86 to 0.79). It gave 16 fewer "extra" answers and gained on
  nodule, emphysema, atelectasis and cardiomegaly, so the effect is mixed, not absent.
- **E (location first):** the interval includes zero, and three conditions dropped by more than 0.05 (emphysema 0.76 to 0.67,
  lymphadenopathy 0.79 to 0.71, mosaic attenuation 0.94 to 0.88). It found 21 more labelled findings but gave 87 more "extra"
  answers: asked "where", the model names a place more readily than it says "None".

Per-condition figures for every set are in `summary.md` of the run folder.

## Result, experiment 2 (do three wordings agree?)

The three wordings gave the same answer on 98.4 % of 1,411 scan-conditions. Set A is wrong against the report label on 156 of them.

| Flag | Flagged | Wrong among flagged | Wrong among not flagged | Share of all errors flagged | Pre-stated test |
|---|---|---|---|---|---|
| The three wordings disagree | 22 | 8 (36 %) | 148 of 1,389 (11 %) | 5 % | met |
| Yes-score of A between 0.05 and 0.95 | 76 | 29 (38 %) | 127 of 1,335 (10 %) | 19 % | met |

Both flags meet the pre-stated test (error rate at least twice as high, at least 20 flagged). Disagreement is rare, so it marks
only 8 of the 156 errors. The Yes-score of the single current question marks 29 of them and needs no extra questions.
Most errors (127 of 156) are answers the model gives with a score near 0 or 1: neither flag sees them.

## Decision

Set A stays as the way of asking NV-Reason-CT. No confirmation run is needed, because nothing is being changed.
Showing a "less certain" mark from the Yes-score would be a display or policy change of its own; it is not decided here.

## Against the clinician reviewer's grades (4 scans, descriptive only)

| Set | Graded answers | Same | Present found | Present missed | Absent called present |
|---|---|---|---|---|---|
| A | 72 | 60 | 21 | 9 | 3 |
| V, B, C, D | 72 | 60 | 21 | 9 | 3 |
| E | 72 | 55 | 24 | 6 | 11 |

Four scans cannot separate the sets; the table only shows that E's extra "present" answers also appear against the grades.

## What this does and does not show

- It shows that, on these 83 scans, none of the five alternatives answers better than the current wording under the rule,
  and that this model's Yes / No answers hardly move when the question is reworded or reasoning is switched on.
- It does not show how good NV-Reason-CT is. 0.814 is on development scans, against report-mention labels, for a model trained
  on reports from the same hospital. "Extra" answers may be correct findings the report did not mention.
- It says nothing reliable about the five conditions with fewer than 10 positive scans (peribronchial thickening,
  arterial and coronary calcification, hiatal hernia, medical material).
- With 12 conditions and 68 patients the test can only see large differences; the intervals for D and E are about 0.05 to 0.07 wide.
- Five sets were compared with A, one more than the template advises.

## Deviations from the plan

None. (The editor session restarted during the run; the run itself was not interrupted and was not resumed.)
