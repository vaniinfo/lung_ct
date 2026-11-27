# Dense lung: what the tool keeps, what it calls it, and why (dense_lung_inventory_v2, checklist_v3_6)

Status: development version, 2026-11-23. Not validated. Research use only.

## 1. What changed

Up to `checklist_v3_5` the consolidation measure removed two kinds of dense lung before counting:

| Step | Rule | What happened to the tissue |
|---|---|---|
| Fluid proximity | within 15 mm of the predicted pleural-fluid outline | moved to row E88 and named "likely compressed lung / atelectasis"; made atelectasis "present" from 30 mL |
| Posterior band | most posterior 20% of each lung, per axial slice | dropped from the measure; reported only as context (E90) |

From `checklist_v3_6` (`dense_lung_inventory_v2`):

- **Nothing is removed by location.** Every consolidation-range voxel (-300 to 100 HU) left after lung-edge erosion,
  vessel / airway removal and opening is kept in a retained inventory: rows E92 (mL), E93 (by lobe), E94 (tags).
- **The two rules become spatial tags**: `fluid_adjacent`, `posterior`, `fluid_adjacent_and_posterior`,
  `no_spatial_tag`. A tag says where a region lies. It names no diagnosis.
- **The regions are called "dense-lung candidates"**, HU-selected, not findings.
- **Fluid-adjacent density is no longer named compressed lung.** E88 keeps its value but is renamed, and atelectasis
  is no longer answered "present" from it.
- **The old consolidation limits (present from 30 mL, none below 5 mL) are not applied to the inventory.** They were
  set for the smaller baseline region.

What did not change: the historical baseline rows (E85, E88, E90) are still computed and stored with the same values;
the consolidation answer still comes from E85 with the same limits; no HU band, limit or gate moved; existing
evidence files, checklist files and viewer packs are not rewritten.

## 2. Why: definitions and guidance

Sources are listed in section 5. Quotations are kept short; the rest is paraphrase.

**Consolidation (Fleischner Society glossary, Hansell et al. 2008).** A homogeneous increase in lung attenuation that
"obscures the margins of vessels and airway walls"; an air bronchogram may be present. Pathologically, alveolar air
is replaced by exudate or another disease product. The glossary adds that the attenuation of consolidated lung is
rarely helpful for the differential diagnosis.

**Atelectasis (same glossary).** Reduced inflation of all or part of the lung. On imaging: reduced volume together
with increased attenuation, often with displacement of fissures, bronchi, vessels, diaphragm, heart or mediastinum.
The defining feature is **volume loss**, which the tool does not measure. Position next to fluid is not part of the
definition.

**Ground-glass opacity (same glossary).** Hazy increased attenuation with bronchial and vascular margins preserved.
Causes listed include partial airspace filling, interstitial thickening, partial collapse of alveoli and increased
capillary blood volume. So partial collapse can itself look like ground-glass.

**Air bronchogram (same glossary).** Air-filled bronchi against airless lung. It occurs when alveolar air has been
absorbed (atelectasis), replaced (for example pneumonia), or both. It therefore does not separate the two.

**Dependent opacity.** The Fleischner CT glossary (Austin et al. 1996) has an entry "dependent opacity" (subpleural
increased attenuation in dependent lung), and the 2008 glossary describes opacity in the posteroinferior lung of a
supine patient that disappears on prone imaging as atelectasis of normal lung. Guidance that follows from this:
dependent density is settled by **prone images**, or by showing that it persists and has other features of disease.
A single supine scan cannot settle it. We did not retrieve the full text of the 1996 entry; the wording above is from
secondary citations and should be checked against the original before it is quoted in a paper.

**What follows for the tool.**

- A fixed posterior band cannot tell gravity-related density from disease in the same place. Pneumonia,
  aspiration and oedema are common in dependent lung. Removing the band hides real findings (valid_832, valid_301);
  keeping it without a tag would over-read normal dependent density. Hence: keep and tag.
- Lung next to an effusion is often passively collapsed, but consolidation and effusion also occur together
  (parapneumonic effusion). Proximity alone does not establish atelectasis. Hence: keep, tag, do not name.

## 3. Limits of attenuation on non-contrast CT

- **Enhancement is what separates them, and we have none.** Studies that distinguish atelectasis from pneumonia by
  attenuation used contrast-enhanced scans: collapsed lung enhances more than infected lung (Edwards et al. 2016,
  CT pulmonary angiograms, 55 patients; a spectral-detector study in 2021 reported the same on contrast images and no
  separating parameter without contrast). The cut-offs they report depend on protocol and contrast phase.
  CT-RATE scans are non-contrast, so none of this applies to our measurements.
- **The HU band is not specific.** On a non-contrast scan -300 to 100 HU contains consolidation, atelectasis, dense
  ground-glass, dependent density, tumour, haemorrhage, fibrosis, vessels and airway walls that the vessel step
  missed, and partial volume at the lung edge, fissures and diaphragm.
- **HU depends on acquisition**: kernel, slice thickness, dose and noise, scanner calibration, and above all lung
  inflation. The tool has no independent assessment of inspiration (see the inspiration audit in
  `runs/D1_dev_audit_20261122/`).
- **The region depends on other predicted outlines**: the lung outline (dense lung is sometimes left out of it), the
  vessel / airway outlines, and for the fluid tag the pleural-fluid outline, which the PleThora check (E14) showed to
  be unreliable for small and some larger effusions. The fluid tag is only as good as that outline.
- **The posterior band is geometric**: 20% of the front-to-back extent per slice, assuming a supine patient. It is
  not a measurement of dependency.
- **Lobe assignment** comes from predicted lobe outlines and is uncertain near fissures and when lobes are distorted.
- **A volume in mL is a property of the selected region**, not a measurement of a finding.

## 4. What may and may not be claimed

- May: "X mL of dense-lung candidates, in these lobes, of which Y mL next to predicted pleural fluid and Z mL in the
  posterior band."
- May not: that the candidates are consolidation, atelectasis or dependent density; that their absence excludes any
  of these; any accuracy figure for the inventory. Diagnostic accuracy has to be evaluated against expert assessment
  or suitable annotated data before a new presence / absence rule is built on it.
- HoloCT statements ("consolidation with air bronchograms in the left lower lobe") are a model's interpretation.
  They are reported in their own section and are never merged into the measurement rows.

## 5. Sources

- Hansell DM, Bankier AA, MacMahon H, McLoud TC, Mueller NL, Remy J. Fleischner Society: Glossary of Terms for
  Thoracic Imaging. Radiology 2008;246:697-722. Entries read: consolidation, atelectasis, ground-glass opacity, air
  bronchogram (via https://thoracickey.com/fleischner-society-glossary-of-terms-for-thoracic-imaging/; that copy is
  cut off after "intralobular lines").
- Austin JH, Mueller NL, Friedman PJ, et al. Glossary of terms for CT of the lungs: recommendations of the
  Nomenclature Committee of the Fleischner Society. Radiology 1996;200:327-331. Entry "dependent opacity": cited from
  secondary sources, full text not retrieved.
- Edwards RM, Godwin JD, Hippe DS, Kicska G. A Quantitative Approach to Distinguish Pneumonia From Atelectasis Using
  Computed Tomography Attenuation. J Comput Assist Tomogr 2016;40:746-751. https://pubmed.ncbi.nlm.nih.gov/27560011/
- Consolidated lung on contrast-enhanced chest CT: the use of spectral-detector computed tomography parameters in
  differentiating atelectasis and pneumonia. Heliyon 2021. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8170158/
- The 2024 update of the Fleischner glossary (Bankier et al., Radiology 2024) was not retrieved; check whether its
  wording for these entries differs before citing.

## 6. Where it lives

- Rows: `lung_ct.agent.conditions.dense_lung_inventory_rows` (E92-E94), used by new runs and by the replay.
- Checklist: `configs/checklist_v3_6.json`, `lung_ct.agent.checklist.add_dense_candidates`.
- Replay on the development cohort: `scripts/dense_lung_inventory_replay.py`, results in
  `runs/E15_dense_lung_inventory_replay_<timestamp>/`.
- Tests: `tests/test_v3_6.py`.
