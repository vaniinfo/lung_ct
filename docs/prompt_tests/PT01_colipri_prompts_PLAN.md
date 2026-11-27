# PT01: COLIPRI - prompt wording for the finding score - PLAN (frozen 2026-11-26, before any score was computed)

Template: [PROMPT_TEST_TEMPLATE.md](../PROMPT_TEST_TEMPLATE.md).

## 1. Question

Do other ways of writing the text prompts rank scans better than the short prompts we use now
("X present" / "No X present")?

**Decision it informs:** which prompt set produces the classifier score shown as evidence on reviewer pages.
The score is evidence only. It cannot make any checklist answer present or absent (policy since checklist_v3_5),
so this test cannot change a checklist decision.

## 2. Model

- `microsoft/colipri` @ `892e42ec`, package colipri 0.1.3, environment `environments/colipri`.
- Fixed: image preprocessing (package default: RAS, 2 mm, -1000..1000 HU, 192^3), pooled and normalised image
  embedding, softmax temperature 0.07 from the checkpoint. Only the text side changes.
- Each scan is encoded once; the same image embedding is used for every prompt set.

## 3. Scans

- All 85 CT-RATE validation scans held locally (70 patients; some patients have two reconstructions).
- **Earlier use: development.** All 85 were used before to design checklist rules or to compare models.
  A result here supports a choice of method only.

## 4. Reference

- CT-RATE labels for the 18 abnormalities, derived from the report text by the data set's authors.
- Limits: "not labelled" means "not mentioned in the report", not "absent". Labels are per scan.
- Ground-glass has no CT-RATE label: it is scored but has no reference and is left out of every summary.

## 5. Prompt sets

Config: `configs/colipri_prompt_sets_v1.json` (SHA-256 recorded in the run manifest).

| Set | Idea | How the two prompts per condition are built |
|---|---|---|
| A (baseline) | Current short prompts | "X present" / "No X present", exactly as in `configs/colipri_ct_v1.json` |
| B | Several sentence forms, same name | Four sentence forms per condition, embeddings averaged |
| C | Several names for the condition | The short form over two to four names (for example hiatal hernia, hiatus hernia), embeddings averaged |
| D | Real report passages ("native", as in the COLIPRI paper) | Mean embedding of 50 Findings sections from scans labelled with the condition and of 50 from scans not labelled with it |

Averaging: each text embedding is normalised, the embeddings of one side are averaged, and the average is normalised again.

**Source of the passages for set D, and how test patients are excluded:** the CT-RATE validation report table
already held locally. All 70 patients of the 85 test scans are removed first. One report per remaining patient
(the first reconstruction of the first scan). 50 positive and 50 negative reports per condition are drawn with a
fixed seed (20261126). Reports longer than the text encoder's 512 tokens are cut off by the tokenizer; the number
cut is recorded. COLIPRI was trained on CT-RATE training reports, not on these.

Known weakness of set D, stated in advance: the passages are written in the same hospital's style as the test
scans' reports, which the labels come from. A gain for D may not carry over to other hospitals.

## 6. Measure

- **Main measure:** AUROC per condition, at scan level, from the score for "present" (0.5 = coin flip, 1 = perfect ranking;
  ties count half). Then the plain average over the 18 CT-RATE conditions ("macro AUROC").
- **Secondary:** average precision per condition and its average.
- No cut-off is used or tuned.
- **Smallest number of positives to interpret a condition:** 10 positive scans. Conditions with fewer are listed and marked "too few to interpret".
- **Interval:** paired bootstrap over patients (all scans of a drawn patient go together), 2000 resamples, seed 20261126,
  2.5th to 97.5th percentile of the difference in macro AUROC to set A. A condition without both a positive and a
  negative scan in a resample is left out of that resample's average.

## 7. Rule for "better"

A set replaces set A only if both hold:

1. the 95 % interval of its macro-AUROC difference to set A lies entirely above zero, and
2. no condition with 10 or more positive scans has an AUROC more than 0.05 below set A.

If several sets meet the rule, the one with the highest macro AUROC is chosen.

- **If met:** the chosen set becomes the proposed method. It is adopted for display only after a confirmation run (point 11).
- **If not met:** set A stays. "No difference shown" is reported as the result.

## 8. Run

```
environments/colipri/Scripts/python.exe scripts/prompt_test_colipri.py --prompt-sets configs/colipri_prompt_sets_v1.json
```

Output: `runs/PT01_colipri_prompts_<time>/` with image embeddings, text embeddings, raw scores per scan and set,
`summary.json`, `summary.md`, `run_manifest.json`. Everything runs locally.

## 9. Result file

`docs/prompt_tests/PT01_colipri_prompts_RESULT.md`.

## 10. What this test may claim

- May claim: which of four prompt sets ranks these 85 development scans best against report labels, and whether the pre-stated rule is met.
- May not claim: a performance figure for COLIPRI, a validated prompt set, anything about a single patient,
  or anything about the rare conditions (hiatal hernia has 4 positive scans, peribronchial thickening 5, medical material 6).

## 11. Confirmation

Planned only if the rule is met: one run of set A and the chosen set on CT-RATE validation scans not downloaded so far.
That needs a new download and is a separate decision.

Changes after freezing: none.
