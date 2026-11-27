# TR02: cut-offs and trust rules on 400 new CT-RATE scans, fitted on one half and confirmed on the other - PLAN (frozen 2026-11-26, before any of the new scans was downloaded)

Same principle as [PROMPT_TEST_TEMPLATE.md](../PROMPT_TEST_TEMPLATE.md): everything is decided before looking.
Follow-up of [TR01](TR01_cutoffs_and_trust_rules_RESULT.md), where 83 scans were too few to judge most conditions.

## 1. Questions

1. With more scans, which COLIPRI cut-off per condition would be used, and is it stable?
2. Which source is right often enough that its "present" (or its "absent") could decide, and does that hold on scans that were not used to choose it?
3. Do the proposals of TR01 (its cut-offs; COLIPRI's "absent" for consolidation) hold on scans TR01 never saw?
4. For the three conditions where checklist_v3_11 lets NV-Reason-CT decide "present": how often does the report agree, on more scans?

**Decision it informs:** whether a trust table is proposed as a new checklist version. Nothing is applied by this test.

## 2. Scans

- 400 scans of 400 patients from the CT-RATE validation split (1,304 patients), none of them among the 70 patients used so far.
- **Chosen without looking at any label:** patients sorted, shuffled with seed 20261126, the first 400 taken; scan "a" of each patient;
  the standard-kernel reconstruction where one exists (same rule as the earlier sets), else reconstruction 1.
- **Split fixed by the same shuffle, before download:** the first 200 patients are the **fit half**, the next 200 the **confirm half**.
- Stored in `D:/datasets/ct_rate/calib400` (downloads, converted volumes, labels and report text, orientation masks). Not in the repository.
- Checks on every scan, as before: values are HU; left / right, front / back and top / bottom from anatomy (heart, liver, lungs, trachea).
  Scans that fail either check are listed and left out. The organ masks for this check are made with TotalSegmentator's fast setting
  (3 mm), which is enough for organ positions; they are used for nothing else.
- Note: the validation reports of some of these patients were used in PT01 to build the "native" prompts of set D. Set D was not adopted
  and is not used here; set A does not use any report text.

## 3. Sources and reference

| Source | Answer |
|---|---|
| NV-Reason-CT | Yes / No with the current wording (set A of `configs/nvreason_prompt_sets_v1.json`), unchanged |
| COLIPRI | score with the current short prompts (set A of `configs/colipri_prompt_sets_v1.json`), turned into yes / no by a cut-off |

- The tool's own answers are not part of this test: they need the full measurement pipeline on every scan.
- Reference: CT-RATE report labels ("not mentioned" is not "absent"). The limits of TR01 apply unchanged.

## 4. Method

Bars, cut-off rule and intervals are those of TR01, unchanged (`scripts/trust_rules.py`: `RULE`, `fit_cutoff`, `judge`).

**Fit half (200 patients)**
- COLIPRI cut-off per condition, fitted on all fit-half scans. For judging COLIPRI inside the fit half, each patient is answered with a
  cut-off fitted without that patient.
- Each source and condition is judged against the bars. This gives the **proposed trust table**.

**Confirm half (200 patients), looked at once, after the fit half is written to disk**
- COLIPRI answers use the cut-offs frozen on the fit half.
- Each source and condition is judged against the same bars.
- **An entry of the trust table is "confirmed" only if it meets the bars in both halves.** An entry that meets them in one half only is reported as "not confirmed".
- Stability of a cut-off: balanced accuracy in the confirm half with the frozen cut-off, next to the fit-half figure.

**TR01's proposals on all new scans** (TR01 was frozen before these scans existed, so all 400 are untouched for it)
- TR01's cut-offs applied unchanged; COLIPRI's "absent" for consolidation judged against the bars.

**The three checklist_v3_11 conditions** (medical material, hiatal hernia, bronchiectasis): NV-Reason-CT's "present" calls against the labels, on all new scans, with interval.

Conditions with fewer than 10 positive scans in a half are listed and marked "too few".

## 5. What is reported

- Scans downloaded / failed / left out by each check, with reasons.
- Per condition: positives in each half; cut-off; found / missed / extra and balanced accuracy in each half.
- The table source x condition for each half, and the list of confirmed entries (it may be empty).
- The checks of TR01's proposals and of the three v3_11 conditions.
- Descriptive only, marked as such: the same figures on all 400 scans pooled.

## 6. Run

```
fetch_ctrate_calibration.bat        (asks for the Hugging Face token; then runs the steps below unattended)
  environments/vlm/Scripts/python.exe scripts/ctrate_fetch_calibration.py
  environments/segmentation/Scripts/python.exe scripts/segment_orientation_organs.py --qc-run runs/P1_qc_ctrate_calib400_<time>
  .venv/Scripts/python.exe scripts/anatomy_contrast_check.py --qc-run runs/P1_qc_ctrate_calib400_<time> --seg-root D:/datasets/ct_rate/calib400/derived --write-scan-list D:/datasets/ct_rate/calib400/verified_scans.json
  environments/nvreason/Scripts/python.exe scripts/prompt_test_nvreason.py --sets A --scan-list D:/datasets/ct_rate/calib400/verified_scans.json --label TR02_nvreason_calib400
  environments/colipri/Scripts/python.exe scripts/prompt_test_colipri.py --scan-list D:/datasets/ct_rate/calib400/verified_scans.json --scores-only --label TR02_colipri_calib400
.venv/Scripts/python.exe scripts/trust_rules_confirm.py --nvreason-run runs/TR02_nvreason_calib400_<time> --colipri-run runs/TR02_colipri_calib400_<time>
```

The model steps never read a label. Result file: `docs/prompt_tests/TR02_calibration_400_RESULT.md`.

## 7. Claims

- **May claim:** which entries met the bars in both halves of 400 scans from one hospital, against report labels.
- **May not claim:** validation, a performance figure, or anything about another hospital. Both models were trained on reports from this hospital.
  Report labels count a correct finding the report did not mention as an error; that is not fixed by more scans.

## 8. After this test

A confirmed entry is a proposal for a new checklist version with its own review. The confirm half is then used and cannot confirm anything again.

Changes after freezing: none
