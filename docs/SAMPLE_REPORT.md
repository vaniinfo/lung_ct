# Sample chest CT report — and how the agent fills it

**Research use only.** This is an illustration of the report *format* and of which pipeline step
supplies each part. It is not a clinical report, and nothing here should be read as a diagnosis.

Related: [Project plan](PROJECT_PLAN.md) · [Paper novelty](PAPER_NOVELTY.md) · [Status](STATUS.md)

Running example: **coronacases_006** (public COVID-19-CT-Seg dataset). Numbers below are the agent's
real outputs from run `runs/A1_agent_coronacases_006_20261115_093149` (classifier run
`runs/P3_colipri_2cases_20261115_093143`).

---

## 1. What a typical chest CT report looks like

Radiologists usually write a chest CT report in fixed sections, roughly like this
(illustrative wording, not a real patient):

> **CT CHEST**
>
> **Clinical information:** Cough and fever. Query pneumonia.
> **Technique:** Non-contrast CT of the chest, 1 mm slices.
> **Comparison:** None.
>
> **Findings**
> - **Lungs and airways:** Patchy opacities in the right lower lobe. No suspicious nodule.
>   No emphysema. Airways are patent; no bronchiectasis.
> - **Pleura:** No pleural effusion or pneumothorax.
> - **Heart and mediastinum:** Heart size normal. No pericardial effusion. No enlarged lymph nodes.
> - **Upper abdomen:** Unremarkable.
> - **Bones:** No acute abnormality.
>
> **Impression**
> 1. Right lower lobe predominant opacities, consistent with infection in the right clinical context.

## 2. Which step fills which part

| Report section | Who fills it | Agent output for coronacases_006 | Status |
|---|---|---|---|
| Clinical information | The person ordering the scan (not in the image) | not available for this dataset | **Not filled**: must come from the request |
| Technique | Step 0 (scan check) | Non-contrast volume, real HU values (−1023 to 2217); slice spacing from the file | Partly: contrast and kernel need DICOM tags, which NIfTI files lack |
| Comparison | Needs a prior scan | none available | **Not filled** |
| Lungs: are there opacities? | Step 3 readers + Step 1 dense-tissue measure (E10) | MedGemma: present. Hulu-Med: **absent**. Measured: 102 mL dense tissue | Filled, with a **flag** (readers disagree) |
| Lungs: which side? | Step 1 measurements (E11–E13) decide; Step 4 checks readers | Both lungs, **strongly right-predominant**: right 95 mL vs left 7 mL (93% right) | Filled, **flag**: MedGemma said "both sides" without the right predominance |
| Lungs: which zone? | Step 1 (E14) | Lower zone: 81% lower, 18% middle, 1% upper | Filled; MedGemma agrees |
| Lungs: peripheral or central? | Step 1 (E15) | Mixed: 70% in the outer third | Filled, **flag**: MedGemma said "peripheral" |
| Lungs: which lobes? | Step 1 (E20–E24) | Mainly right lower lobe (10.3% of the lobe dense); other lobes under 1% | Filled |
| Lungs: how severe? | Step 1 (E25) | Dense-tissue severity score 6 of 25 (proxy, not a radiologist score) | Filled, low trust |
| Lungs: type of opacity (ground-glass vs consolidation) | Step 2 classifier + Step 3 readers | Classifier suggests consolidation (0.99), ground-glass (0.60) | Shown as "classifier suggests", **uncalibrated** |
| Lungs: nodules | Step 2 classifier | Classifier 0.81 | **Unreliable** (see note below); needs a nodule detector |
| Lungs: emphysema | Step 1 (E30) + Step 2 | 6.3% of lung below −950 HU; classifier 0.82 | Filled as a measurement; cutoff not validated here |
| Airways / bronchiectasis | Step 1 airway tree (E40–E42) + Step 2 | 203 branches, deepest generation 11; classifier 1.00 | Measurements unvalidated; classifier score **unreliable** |
| Pleura: effusion | Step 2 classifier | 0.01 (no effusion suggested) | Shown as "classifier suggests" |
| Pleura: pneumothorax | — | not measured yet | **Not filled** |
| Heart: size, pericardial effusion | Step 2 classifier | Cardiomegaly 0.00; pericardial effusion 0.08 | Shown as "classifier suggests"; a heart-size measurement is planned |
| Mediastinum: lymph nodes | Step 2 classifier | Lymphadenopathy 0.00 | Shown only; no measurement |
| Upper abdomen, bones | — | not analysed | **Not filled**: outside this project's scope |
| Impression | Step 5 report writer, using Step 4 decisions | see section 3 | Filled; every claim cites an evidence ID |
| Disagreements to check | Step 4 cross-check | 7 flags: 3 about opacities and location (listed below), the readers disagreeing with each other, and 3 "classifier suggests a finding no reader mentioned" (nodule, emphysema, bronchiectasis), which are likely noise (see section 4) | Always listed for the doctor |

## 3. The agent's draft for coronacases_006

What the agent would hand to the doctor (written from the rule-based report and cross-check; the
MedGemma-written version uses the same evidence and must cite the same IDs):

> **CT CHEST — AI research draft (not a clinical report)**
>
> **Technique:** Non-contrast chest CT, density values verified as Hounsfield units.
>
> **Findings — lungs**
> - Lung opacities present in **both lungs, strongly right-predominant**: dense tissue 102 mL,
>   right 95 mL, left 7 mL (93% right). [E10, E11, E12, E13]
> - **Lower zone predominant** (81% lower, 18% middle, 1% upper). [E14]
> - **Mixed** peripheral and central distribution (70% in the outer third). [E15]
> - Mainly the **right lower lobe** (10.3% of the lobe); other lobes under 1%. [E20–E24]
> - Dense-tissue severity score **6 of 25** (proxy measure). [E25]
> - 6.3% of lung below −950 HU (emphysema measure, not validated here). [E30]
> - Classifier suggests consolidation and ground-glass opacity (uncalibrated scores). [E51, E52]
>
> **Please check (disagreements):**
> 1. MedGemma described opacities in **both lungs** without noting that 93% are on the right. [E13]
> 2. Hulu-Med reported **no opacities**; measurements find 102 mL of dense tissue. [E10]
> 3. MedGemma said **peripheral**; measurements show a mixed distribution (70% outer third). [E15]
>
> **Limitations:** dense-tissue measure is a stand-in for an opacity outline and under-counts faint
> haze; classifier scores are uncalibrated; thresholds were set on a 10-case COVID pilot that
> includes this case.

For comparison, the expert annotation for this case (never shown to the agent) is: one-sided right,
lower zone, mixed. The agent's measured answers matched on zone and peripheral/central; on side, the
agent says "both lungs, strongly right-predominant", because 7 mL of dense tissue was found on the left.

## 4. Important caveat: the classifier is not reliable yet

On our scans, the zero-shot classifier (COLIPRI) gives very high scores to findings that are unlikely
in these COVID cases. Examples: bronchiectasis 1.00 for coronacases_006; lung nodule 0.99 and hiatal
hernia 0.79 for coronacases_001. The scores are uncalibrated for this dataset. Until they are
calibrated (for example on normal scans and on labelled data such as RAD-ChestCT or a CT-RATE subset),
the report shows them only as "classifier suggests", never as findings.

## 5. What would complete the report

| Missing part | What would fill it |
|---|---|
| Opacity outline (instead of the dense-tissue stand-in) | A trained opacity segmenter (pipeline step 1 upgrade) |
| Reliable finding scores | Classifier calibration on normal and labelled scans |
| Pneumothorax, pleural effusion size | TotalSegmentator `pleural_pericard_effusion`; a pneumothorax measure |
| Heart size | Heart mask, then a CT heart-size or cardiothoracic-ratio-style measure |
| Bronchiectasis evidence | Validated airway diameters and airway-to-artery ratio |
| Clinical information, comparison | From the imaging request and prior scans, not the image |
