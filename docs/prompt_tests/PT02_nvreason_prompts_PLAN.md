# PT02: NV-Reason-CT - question wording, reasoning, report-first, location-first, voting - PLAN (frozen 2026-11-26, before any answer was compared with a label)

Template: [PROMPT_TEST_TEMPLATE.md](../PROMPT_TEST_TEMPLATE.md).

## 1. Question

Does another way of asking NV-Reason-CT give more correct present / absent answers than the wording we use now?
And: do three harmless rewordings of one question agree, and does disagreement mark the answers that are wrong?

**Decision it informs:** which way of asking produces the AI-reader answers shown on reviewer pages. Under
checklist_v3_11 the reader's "present" is the tool's answer for three conditions, so a change of wording would be a
new policy version with its own review. This test alone changes no page and no policy.

## 2. Model

- `nvidia/NV-Reason-CT` @ `386b93e0`, environment `environments/nvreason`, settings of `configs/nvreason_ct_v1.json`.
- Fixed: the model's own image processing (384 mm cube, 2 mm, chest), sampling off, bfloat16.
- The scan is processed once per scan and reused for every question. On the first scan and every 20th, three
  questions are also asked the ordinary way; the answer text must be identical or the run stops.

## 3. Scans

- 83 of the 85 CT-RATE validation scans held locally (69 or 70 patients; counted in the result). Two scans are left
  out because left / right was not verified from anatomy in any QC run: valid_64_a_3, valid_693_a_2.
- **Earlier use: development.** All were used before to design rules or compare models. 24 of them were already
  answered by this model with set A. A result here supports a choice of method only.

## 4. Reference

- CT-RATE labels (from the report text, by the data set's authors) for 17 of the 18 conditions.
  Ground-glass has no CT-RATE label: it is asked but not scored.
- Limits: "not labelled" means "not mentioned in the report", not "absent", so some "extra" answers may be right.
  The model was trained on CT-RATE training reports from the same hospital, which favours it on these labels.
- Separate, descriptive only: the clinician reviewer's grades on four scans.

## 5. Prompt sets

Config: `configs/nvreason_prompt_sets_v1.json` (SHA-256 recorded in the run manifest).

| Set | Idea | Question (X = the condition) | Reasoning |
|---|---|---|---|
| A (baseline) | Wording used so far | "Is X present in this CT? Answer only Yes or No." | off |
| A2 | Harmless rewording | "Does this CT show X? Answer only Yes or No." | off |
| A3 | Harmless rewording | "Is there X in this CT scan? Answer only Yes or No." | off |
| V | Vote of A, A2, A3 | present when at least two say present (no extra questions) | off |
| B | Reasoning on | same as A | on |
| C | Definition in the question | "Is X (plain definition) present in this CT? Answer only Yes or No." | off |
| D | Report first | the model writes its structured report; then "Based on the CT and your report, is X present? Answer only Yes or No." | on for the report, off for the answer |
| E | Location first | "Where in this CT is X? Answer with the location, or answer None if it is not present." | off |

Compared with A: V, B, C, D, E (five sets; the template advises three or four, so one of five looking best by chance is more likely, which is why the rule uses an interval).
A2 and A3 are parts of V and of the stability check; they are reported but cannot replace A on their own.

No example text from any patient is used. The definitions in set C were written by us before the run.

Reading the answers (fixed in the config): Yes -> present, No -> absent; for set E a first word of none / no / nowhere /
absent / not -> absent, any other text -> present. Anything else is "unreadable": counted, never guessed.

## 6. Measure

- **Counts per condition and set, always reported:** found, missed, extra, correctly absent.
- **Main measure:** balanced accuracy per condition = (share of labelled scans found + share of unlabelled scans answered absent) / 2,
  then the plain average over conditions with 10 or more positive scans ("macro balanced accuracy"). 0.5 = no better than a coin, 1 = perfect.
  It is used because a plain accuracy would reward answering "absent" to everything.
- An unreadable answer counts as wrong.
- **Secondary:** AUROC of the Yes-score per condition, for sets that have a score (A, A2, A3, C, D).
- **Smallest number of positives to interpret a condition:** 10 positive scans. Others are listed and marked.
- **Interval:** paired bootstrap over patients, 2000 resamples, seed 20261126, 2.5th to 97.5th percentile of the difference to set A.
- **Wording stability (experiment 2):** share of scan-conditions where A, A2 and A3 agree; error rate of A for unanimous and for
  split scan-conditions; the same split for the Yes-score of A (below 0.05 or above 0.95 against in between), to see which flags errors better.

## 7. Rule for "better"

A set replaces set A only if all three hold:

1. the 95 % interval of its macro-balanced-accuracy difference to A lies entirely above zero;
2. no condition with 10 or more positive scans is more than 0.05 below A in balanced accuracy;
3. at most 1 % of its answers are unreadable.

If several sets meet the rule, the one with the highest macro balanced accuracy is chosen.

- **If met:** the set becomes the proposed way of asking. It is adopted only after a confirmation run (point 11) and as a new policy version.
- **If not met:** set A stays. "No difference shown" is reported as the result.

For experiment 2 there is no switch rule. Stated in advance: disagreement is called a useful flag only if the error rate of split
scan-conditions is at least twice that of unanimous ones and the split group holds at least 20 scan-conditions.

## 8. Run

```
environments/nvreason/Scripts/python.exe scripts/prompt_test_nvreason.py --sets A,A2,A3,B,C,D,E
.venv/Scripts/python.exe scripts/prompt_test_yesno.py --run runs/PT02_nvreason_prompts_<time>
```

Output folder: `runs/PT02_nvreason_prompts_<time>`. The first script never reads a label.
A one-scan timing check (`runs/PT02_timing_check_*`, valid_126_a_1) was made before freezing; its answers were not compared with labels.

## 9. Result file

`docs/prompt_tests/PT02_nvreason_prompts_RESULT.md`

## 10. Claims

- **May claim:** on these development scans, set X did or did not answer better than the current wording, under the rule above.
- **May not claim:** a performance figure for the model, validation, or anything about a single patient.

## 11. Confirmation

Planned only if a set meets the rule: one run with that set and set A on untouched scans. That needs new scans, which needs a
decision on the data budget.

Changes after freezing: none
