# Paper novelty — auditing grounded chest CT model outputs

See also [RELATED_WORK.md](RELATED_WORK.md): comparison of ThoraxCheck (this project; Python package `lung_ct`) with RadAgent, MedScribe and other CT systems.

Living document. This file records candidate contributions, supporting evidence,
limitations, and the experiments needed to evaluate them.

A contribution does not require proving that nobody has published anything
similar. It requires a clearly documented distinction from related work and
evidence that the distinction matters.

## Current research direction

We test whether automatic post-generation checks can identify errors in the
finding–mask pairs emitted by a grounded chest CT foundation model.

The primary checks concern:
- side;
- lobe;
- empty masks;
- contradictions between structured answers and generated finding statements.

Attenuation checks are an exploratory addition, evaluated separately.

Checks use separately computed anatomical segmentation and CT attenuation.
They can return "cannot assess" with a reason when their own inputs are inadequate.

Blinded clinician adjudication of flagged and unflagged findings establishes
whether flags correspond to actual errors and how many errors the checker misses.

The primary evaluation concerns errors within generated finding–mask pairs.
It does not establish completeness of abnormality detection.

Passing the audit means no tested inconsistency was found.
It does not establish clinical correctness.

Working title:
"Auditing Anatomical and Attenuation Consistency in Grounded Chest CT Model Outputs"

## What is established prior work

The following are not standalone novelty claims:

- Detecting conditions from chest CT.
- Generating reports or answering CT questions.
- Producing masks linked to findings.
- Calculating volumes from segmentation masks.
- Using anatomical measurements and HU features in report generation.
- Evaluating localization or anatomical grounding.
- Combining models and tools in a CT agent.
- Running models locally.
- Obtaining clinician feedback.

Relevant prior work includes:

| Project | Relevant overlap |
|---|---|
| HoloCT | Finding-grounded reporting, segmentation, VQA and grounding evaluation |
| ReXGroundingCT and its challenge | Finding-level segmentation and anatomical localization evaluation |
| ORGANIC3D | Evaluation and improvement of anatomical image–text alignment |
| EviGround-CT | Post-generation verification through sentence-conditioned grounding |
| MedScribe | Reporting using segmentation, quantitative features and HU information |
| CheapCT / MeasureVQA | Measurement-derived reference answers for CT model evaluation |

Our focused search has not established that the exact proposed audit is new.
A documented comparison with the closest methods remains required.

Do not describe EviGround-CT as requiring expert masks during deployment
without verifying its full methods. Reference masks used for evaluation are
different from inputs required at inference.

## Candidate contribution

A post-generation audit that examines a model's emitted finding–mask pairs
against anatomical constraints and measured attenuation, returns interpretable
flags or cannot-assess reasons, and is validated against clinician judgments.

The distinction to investigate is the combination of:
1. Auditing already-generated finding–mask pairs on ordinary CT scans.
2. Separately computed anatomical checks.
3. Explicit handling of inadequate audit evidence.
4. Clinician validation of both flagged and unflagged findings.
5. Measurement of the trade-off between retained findings and errors.
6. Evaluation across paired reconstructions.

This is a candidate contribution, not an established priority claim.

## Current status

| Component | Status |
|---|---|
| HoloCT-4B local inference | Run on four development cases |
| Finding-mask volumes and lobe intersections | Implemented in exploratory hybrid script |
| Attenuation summaries | Exploratory measurements available |
| Primary flagging module | Not yet built as a frozen module |
| Review protocol | audit_review_protocol_v0_draft.json; draft |
| Blinded per-finding viewer | Planned; not yet built |
| Finding inventory and sampling manifest | Planned |
| Clinician adjudication | Not yet completed |
| Same-input reproducibility | Identical outputs for one scan in two processes |
| Paired-reconstruction metadata review | Completed for 15 pairs |
| Fresh-case evaluation | Not started |

Update these statuses only when supported by saved artifacts or completed runs.

## Development observations: hypothesis generation only

Four already-seen cases exposed candidate issues including:
- a bilateral description with predominantly unilateral mask support;
- possible text–mask lobe mismatches;
- a finding statement with an empty mask;
- possible contradictions between structured answers and finding statements;
- attenuation distributions warranting visual review.

These are automated observations, not clinician-confirmed errors.

The exploratory hybrid decision rule added one detected reference finding
and one extra positive relative to HoloCT alone.

We have not demonstrated a detection advantage.
The proposed paper does not require one.

Do not use these cases for confirmatory evaluation or report their
post-hoc rules as prespecified.

## Evidence and interpretation limits

### Separately computed does not mean statistically independent

The finding mask comes from HoloCT.
Attenuation is measured inside that mask.
Anatomical segmentation and HoloCT use the same CT.
Different models may share training sources.

Agreement groups are groups to evaluate, not calibrated confidence levels.

### Anatomy checks can also fail

A location discrepancy may originate from:
- incorrect text;
- incorrect finding segmentation;
- incorrect anatomical segmentation;
- ambiguous boundaries;
- invalid spatial alignment.

Tiny masks near fissures require particular care.
An audit that cannot resolve these issues should return cannot assess.

### Attenuation is not a universal diagnostic rule

Ground-glass opacity is not defined by a universal median-HU cutoff.
The conventional low-attenuation emphysema measure is a voxel fraction,
not the median HU of an arbitrary model mask.

Retain median, selected percentiles and prespecified voxel fractions
as descriptive measurements.

Attenuation flags remain exploratory until their definitions and incremental
value are evaluated. Do not select bands to fit the four development cases.

### Anatomical compartment matters

Pleural effusions require pleural/side checks rather than interpretation as
disease occupying a lung lobe.

Report mask volume outside the anatomical regions.
Use union volumes to avoid double-counting overlapping masks.
Distinguish empty masks from nonempty volumes rounded to zero.

## Experiment 1 — clinician-validated audit

Question:
Do automatic flags identify actual errors in generated finding–mask pairs?

Primary checks:
- side inconsistency;
- lobe inconsistency;
- empty mask;
- contradictory outputs.

Evaluate attenuation as a separate add-on.

### Review design

Review all flagged findings and a randomly selected sample of unflagged findings.
Preserve the full inventory and each finding's inclusion probability.

Initial reader view:
- CT;
- exact HoloCT finding statement;
- corresponding finding mask.

Hide:
- audit verdict;
- checklist outputs;
- agreement group;
- sampling stratum;
- original radiology report;
- calculated attenuation summaries.

Save and lock the initial assessment, then reveal anatomical boundaries
for a separate anatomical-mask assessment.

Preserve amendments with timestamps.

### Grading

Allow multiple error types:
- text error;
- finding-mask error;
- anatomical-mask error.

Also provide exclusive overall outcomes:
- no identified error;
- cannot adjudicate.

Record anatomy assessed / not assessed.
Not assessed must never be counted as correct.

### Outcomes

- Proportion of flagged findings with adjudicated errors.
- Proportion of adjudicated errors detected by the checker.
- False-positive rate among findings adjudicated as having no error.
- Audit coverage and cannot-assess rate.
- Error rates by flag type and finding type.
- Incremental benefit and false alarms from attenuation checks.

Use weighted estimates where sampling requires them.
Account for findings clustered within patients.
Handle non-assessable judgments explicitly.

Reviewing generated findings does not measure abnormalities the model omitted.

## Experiment 2 — selective reporting

Question:
Does withholding flagged findings reduce error while retaining useful findings?

Evaluate:
- error rate versus retained coverage;
- correct findings withheld;
- errors retained;
- reliability of prespecified agreement groups.

Do not label "both agree" as high confidence before evaluation.
Do not replace cannot-assess outcomes with unverified model answers
and describe that as resolved uncertainty.

Use the same frozen outputs and clinician reference as Experiment 1.
Do not tune selection rules on the evaluation judgments.

## Experiment 3 — reconstruction and generation stability

Current paired data:
- 4 Siemens pairs: matched on recorded geometry; recorded kernels differ.
- 11 Philips pairs: kernel, matrix, spacing and slice thickness differ.

Report these groups separately.
Do not describe all 15 pairs as a kernel-only experiment.

Verify patient/acquisition pairing, coverage, geometry, contrast information,
and recorded reconstruction settings before inclusion.

Run types:
1. Same-input, same-settings repeats: execution reproducibility.
2. Different-seed repeats: sampling variability.
3. Matched seeds across reconstruction pairs: reconstruction-related variation.

Outcomes:
- fixed-condition answer flips;
- changes in emitted findings;
- mask changes;
- changes in audit flags.

Instability alone does not identify which output is wrong.

### Finding matching

Fixed-condition yes/no comparisons do not require geometric matching.

For mask stability, use a frozen matching procedure with:
- fixed finding-type parsing;
- candidate geometric matches;
- deterministic one-to-one assignment;
- explicit unmatched, ambiguous and split/merge outcomes.

Dice >= 0.1 or centroid distance <= 30 mm is a provisional candidate rule,
not an approved final matching definition.

Empty masks cannot establish geometric correspondence.

## Reference data and provenance

Public finding masks can support localization evaluation where a valid
reference correspondence exists.

An unmatched generated finding must not automatically be scored false:
the annotation may not cover that finding.

Verify:
- exact dataset release and accessible splits;
- patient-level overlap between development and evaluation data;
- possible overlap with model training sources;
- correspondence between reconstruction identifiers;
- which findings each reference annotation covers.

Published test-list membership alone does not establish absence of exposure
through other training datasets.

Clinician review is central to evaluating the audit.
It is not merely an optional confirmation of automated scores.

## Future work outside the primary paper

### Airway evidence for bronchiectasis

Potentially aligned with the group's clinical goals, but requires separate
validation of landmarks, airway measurements and airway–artery correspondence.

Current graph depth is not anatomical generation.
Processing-sensitive branch counts and tortuosity must not be presented as
validated disease evidence.

### Local deployment

Running within a 24 GB GPU budget is an implementation advantage.
Report measured memory, runtime and dependencies, but do not claim local
execution alone as scientific novelty.

## Immediate milestone and stopping point

- [ ] Define the shared, versioned finding and review schema.
- [ ] Build the separate blinded viewer.
- [ ] Implement and test the primary flagging module.
- [ ] Build the finding inventory and sampling manifest writer.
- [ ] Complete additional reproducibility checks.
- [ ] Finalize matching rules on development cases.
- [ ] Complete a reader usability trial using the four already-seen cases.
- [ ] Document the closest-work comparison.
- [ ] Freeze protocol, rules, parser, model revision and sampling procedure.
- [ ] Begin fresh-case evaluation.

No additional model is required to complete this milestone.

## Related work links

- HoloCT: https://arxiv.org/abs/2611.06471
- ReXGrounding Challenge: https://arxiv.org/abs/2611.23862
- ORGANIC3D: https://arxiv.org/abs/2611.23529
- EviGround-CT: https://arxiv.org/abs/2610.33038
- MedScribe: https://arxiv.org/abs/2605.01779
- CheapCT / MeasureVQA: https://arxiv.org/abs/2607.22771

## Revision log

| Revision | Change |
|---|---|
| Original: 2026-11-15 | Broad candidate claims for a grounded lung CT agent |
| Audit-focused revision | Replaced broad novelty claims with a clinician-validated finding–mask audit; separated exploratory attenuation and future airway work |
