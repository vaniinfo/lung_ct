# Pleural effusion: what we did and what we learned (E14 to E17)

Date: 2026-11-23. Research use only. All results are on development data with open pages; none is a blinded
validation result. Full numbers: `runs/E17_pleural_model_comparison_prep_20261123/REPORT.md` and
`runs/E16_empty_mask_root_cause_20261123/REPORT.md` (run folders are local, not in Git).

## The finding in one paragraph

For the paper, the useful finding is broader than which model won. **Small or empty masks can miss
expert-labelled effusions, and large predicted volumes can come from the wrong anatomy. Volume alone cannot resolve
either failure.** Two independent segmentation pipelines showed both failures on the same public scans. A rule of the
form "more than X mL means effusion, less than Y mL means none" therefore cannot be trusted in either direction
without a check of where the mask is.

## What we did

| Step | Question | What we did | Outcome |
|---|---|---|---|
| E14 | How does our current effusion mask compare with expert outlines? | Ran the current pipeline (TotalSegmentator effusion task) on 155 PleThora scans; 77 have an expert effusion outline | Median Dice 0.38; six scans with an expert outline got an empty mask |
| E16 | Why were those six empty? | Traced each from source CT through cropping, resampling, network and postprocessing | The fluid was inside the cropped box every time. The loss happens inside the network. Cause not established: two hypotheses (unusual density, thin fluid) remain unseparated |
| Policy (checklist v3_7, v3_8) | What may the page say from a mask alone? | A mask above the limit is shown as "suggested by segmentation", never "present". A mask below the lower limit is "cannot tell", never "absent". Limits (20 mL / 5 mL) kept, relabelled "provisional reporting limits" | Wording change only; no threshold changed |
| E17 | Would a different published model fix the misses? | Ran one alternative (Sexauer et al. 2022, nnU-Net v1) with its released settings on a fixed set of 213 scans; no tuning, no filters | No clear benefit; none of the six misses recovered; new error types appeared |

## What we learned

### 1. An empty mask is not evidence of absence

- Six of 77 expert-outlined effusions (9 to 150 mL) got an empty mask from the current pipeline.
- The alternative pipeline recovered none of them. On three it produced a few voxels in the wrong place.
- Seven more scans had a mask under 5 mL from the current pipeline although an expert had outlined fluid.
- Consequence for the product: "no mask" must read "cannot tell", not "no effusion".

### 2. A large mask is not evidence of presence

- The alternative labelled 125 mL to 1.6 L of kidney, abdominal muscle or brain as effusion on five long scans.
- Both pipelines put tens to hundreds of mL on lung tumour in scans coded "no effusion"
  (for example 99 to 100% of the mask inside the tumour outline on four scans).
- Consequence: a volume above a limit supports "suggested by segmentation" at most. Where the mask sits has to be
  checked before anything stronger is said.

### 3. Counting empty masks hides what happened

The alternative had 5 empty masks instead of 6. That looks like one miss recovered. It is not: three targets stayed
empty, three got a non-empty mask that was not on the fluid (best Dice 0.003), and two other scans became empty.
Per-case tables are needed; summary counts alone mislead.

### 4. The pipeline around the network matters as much as the network

- The current pipeline crops to the chest before the network. The alternative reads the whole scan. That design
  difference, not the learned weights, explains the outside-chest errors.
- We compared two pipelines as we implemented them. The alternative's authors used their own downsampling tool, which
  is not published and was not reproduced. Our result is not a judgement of their published pipeline.
- The alternative's own command does not run on Windows; we drove the same functions from one process.

### 5. Both pipelines miss the same scans

Shared misses are unlikely to be a quirk of one model. They may be thin fluid, fluid of unusual density, or cases
where readers would disagree with the reference. We have not separated these. Additional expert adjudication could
clarify the reference question; it is not done.

### 6. Automatic sanity checks fail in both directions too

Our "is the lung label mostly air?" check raised 21 warnings. Reviewed one by one, 20 were diseased lung correctly
outlined and 1 was a real error. The same check missed five scans where the lung label included air outside the body
or tissue outside the chest. A warning list has to be reviewed case by case, and a quiet check proves little.

### 7. Reference and provenance details changed the answers

- PleThora has several reviewer files per scan. The official reference is the first-reviewer file of the June 2020
  release; an older spreadsheet was out of date for 35 scans.
- The separate thoracic-cavity outline is "space a healthy lung would fill". Its low overlap with the effusion
  outline is expected and is not evidence against the effusion reference.
- Scans coded "no effusion" have no outline and were never verified negative. We report mask states for them and no
  false-positive rate.
- A rounding step once made "0 mL above 10% probability" look like a contradiction; exact voxel counts resolved it.

## What we do not claim

- The two models are not shown to be equivalent, and the current model is not validated.
- No sensitivity, specificity or accuracy figure for new patients follows from this work.
- The limits of 20 mL and 5 mL are provisional reporting limits, not diagnostic boundaries.
- The cause of the six misses is not known.

## Numbers to cite (development data, 77 expert-outlined PleThora scans)

| | Current pipeline | Alternative pipeline |
|---|---|---|
| Median Dice | 0.376 | 0.414 |
| Empty masks | 6 | 5 |
| Scans with Dice 0.7 or more | 16 | 19 |
| Median absolute volume error | 47.7 mL | 42.2 mL |
| Mean absolute volume error | 77.8 mL | 94.7 mL (75.1 mL without one 89 cm scan; sensitivity analysis) |
| Scans improved / worsened / similar (Dice step 0.10) | reference | 6 / 4 / 67 |

## Decision and status

- Current pipeline retained. No clear replacement benefit demonstrated.
- The six target misses remain unresolved.
- No further model runs or training without a new experiment brief.

## Possible next questions (each needs its own brief)

1. Do a second expert's readings of the shared misses agree with the reference?
2. Can a location check (is the mask in the pleural space, outside tumour, inside the chest) be defined and tested on
   scans not used to design it?
3. Do a small number of verified-negative scans make a specificity statement possible?
