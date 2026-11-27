# Prompt test: reusable plan template

Use this template whenever someone wants to know whether a different way of asking a model gives better answers.
It works for a classifier that compares a scan with sentences (COLIPRI, CT-CLIP) and for a model that answers in
words (NV-Reason-CT, MedGemma). Copy the "Plan form" at the end into `docs/prompt_tests/PTnn_<model>_<topic>_PLAN.md`,
fill it in, and freeze it before any score is computed.

A filled example: [PT01_colipri_prompts_PLAN.md](prompt_tests/PT01_colipri_prompts_PLAN.md).

## The one rule

**Decide everything before you look at any result.** The prompts, the scans, the measure and the rule for
"better" are written down and frozen first. If you change any of them after seeing scores, the test becomes a
new test with a new number, and the old result is kept.

Why: with small numbers, one of several prompt versions will look best by chance. Choosing after looking makes
that chance look like a finding.

## Steps

| Step | What you do | What you write down |
|---|---|---|
| 1. Question | State what you want to learn and which decision it informs | One sentence each |
| 2. Model | Fix the model, its exact version and its settings | Model id, revision or weights hash, environment |
| 3. Scans | List the scans and say how each was used before | Scan list, patients, "development" or "untouched" |
| 4. Reference | Say what counts as the truth and how good that truth is | Source of labels and its known limits |
| 5. Prompt sets | Write every prompt version in a config file | File path and SHA-256, one baseline named |
| 6. Measure | Choose one main measure and the unit of counting | Measure, per scan or per patient, how ties are handled |
| 7. Rule for "better" | Say what result would make you switch | A rule that can be checked by a program |
| 8. Run | Run all prompt sets on the same scans in one go | Run folder, versions, hashes |
| 9. Report | Fill the result tables; report everything, including losers | Result file |
| 10. Claim | Say what the result allows and what it does not | "Choice of method" or "performance figure" |
| 11. Confirm | If you want a figure to quote, repeat once on untouched scans | A second, separate plan |

## Step notes

**1. Question.** Name the decision. "Should the classifier score shown on reviewer pages use prompt set B instead of A?"
is a good question. "Which prompt is best?" is not, because it has no end.

**2. Model.** Change only the prompt. Same weights, same image preprocessing, same precision. If the model is
asked in words, also fix whether reasoning is on, the maximum answer length, and that sampling is off.

**3. Scans.**
- Count by patient as well as by scan. Two reconstructions of one patient are not two independent cases.
- Mark every scan "development" (already looked at, used to design rules or prompts) or "untouched".
- A result on development scans supports a choice of method. Only untouched scans support a performance figure.
- Keep a ledger: once a scan set has been used to choose something, it is development data from then on.

**4. Reference.** Say where the labels come from and what is wrong with them. For CT-RATE the labels come from the
report text, so "not labelled" means "not mentioned", not "absent". Clinician grades are stronger but few.

**5. Prompt sets.**
- Always include the current prompts as the baseline.
- Keep the number of sets small (three or four). Each extra set is another chance to win by luck.
- Text used to build prompts (example sentences, report passages) must not come from the patients being tested.
- Say for each set what idea it tests: other wording, several wordings averaged, synonyms, real report passages.
- Freeze the file. Record its SHA-256 in the run.

**6. Measure.**
- For a score (0 to 1): use a ranking measure such as AUROC per condition, then the average over conditions.
  It needs no cut-off, so no threshold is tuned.
- For a yes / no answer: count found, missed and extra per condition. Do not collapse into one accuracy figure.
- State the smallest number of positive cases below which a condition is listed but not interpreted.
- Give an interval, not only a point. Resample patients, not scans.

**7. Rule for "better".** Write it so a program can check it. Example: "Set B replaces set A only if the 95 %
interval of the difference in average AUROC lies above zero and no condition with ten or more positive scans
drops by more than 0.05." If the rule is not met, the baseline stays. "No difference shown" is a valid result.

**8. Run.**
- Everything runs locally. No scan, mask or report text is sent to an outside service.
- Encode each scan once and reuse it for every prompt set where the model allows it. Then prompt sets differ
  only in text, and the run is fast and exactly repeatable.
- Save raw scores before any summary. Record model revision, package versions, config hashes and the git commit.
- Failed scans are listed, never dropped silently.

**9. Report.** Use the same tables every time (see the form). Report every prompt set that was run. Report
conditions where the new prompts are worse.

**10. Claim.** Allowed after a development-scan test: "we chose set B as our method". Not allowed: "the model
reaches 0.xx", "validated", or any statement about a single patient. A prompt change must not silently change
a decision rule: if a score feeds a rule, that is a new policy version with its own review.

**11. Confirm.** One run, on untouched scans, with the chosen set and the baseline only. No further choices.

## Extra checks for models that answer in words

| Check | How | Why |
|---|---|---|
| Wording stability | Ask the same question with a harmless change (capital letter, "a" / "the") | Shows which answers sit near the line |
| Answer score | Record the weight on "Yes" against "No" for the first answer word | Finds borderline answers; it is not a probability of disease |
| Offered answers | Note whether "cannot tell" was offered | A model never offered it will never say it |
| Reasoning on / off | Treat as two prompt sets | It can change answers and costs much more time |
| Repeatability | Run the same prompt twice | With sampling off the answers must be identical |

## Common mistakes

- Picking the best prompt on a set of scans and quoting its score on the same scans.
- Trying one more wording "just to see" after the results are in, without a new plan.
- Building example prompts from the reports of the test patients.
- Counting scans when several come from the same patient.
- Reading a rare condition (fewer than about ten positive cases) as proof either way.
- Tuning a cut-off on the same scans and calling the result accuracy.
- Reporting only the conditions that improved.

## Plan form (copy from here)

```
# PTnn: <model> - <topic> - PLAN (frozen <date>, before any score was computed)

1. Question:
   Decision it informs:
2. Model: <id> @ <revision / weights SHA-256>; environment; settings that stay fixed:
3. Scans: <how many scans / patients>; list file:
   Earlier use: development | untouched (say which and why)
4. Reference: <label source>; known limits:
5. Prompt sets (config file, SHA-256):
   A (baseline):
   B:
   C:
   Source of any example text, and how test patients are excluded:
6. Main measure:            Unit: scan | patient
   Secondary measures:
   Smallest number of positives to interpret a condition:
   Interval: <method, resamples, seed>
7. Rule for "better":
   What happens if it is met:            If it is not met:
8. Run command(s):
   Output folder:
9. Result file:
10. What this test may claim:
    What it may not claim:
11. Confirmation planned: yes | no; on which scans:
Changes after freezing: none | <dated list; each one makes this a new test>
```

## Result form (copy from here)

```
# PTnn: <model> - <topic> - RESULT (<date>)

Plan: <link>; plan unchanged since freezing: yes | no
Run folder:            Git commit:            Config SHA-256:
Scans run / failed:            Patients:

| Prompt set | Main measure | Difference to baseline | 95 % interval |
|---|---|---|---|

| Condition | Positive scans | A | B | C | Best |
|---|---|---|---|---|---|

Rule for "better": met | not met, because:
Decision:
Conditions that got worse:
What was not tested:
Scan sets that are development data from now on:
```
