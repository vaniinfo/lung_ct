# Related work: CT agents and grounded CT reporting, compared with lung_ct

Status: living document, started 2026-11-27. Research use only.
Related files: [PAPER_NOVELTY.md](PAPER_NOVELTY.md) (candidate contributions and what is not novel),
[research/Chest CT model candidates.md](research/Chest%20CT%20model%20candidates.md) (model survey with sources and figures),
[PIPELINE_FLOW.md](PIPELINE_FLOW.md) (how our system works).

## 1. Comparison at a glance

| System | Modality, date | Measures from masks | Fixed checklist | Can say "can't tell" | Image-quality checks | Cites evidence per claim | Small, local | Doctor review page |
|---|---|---|---|---|---|---|---|---|
| **Ours (lung_ct)** | CT, 2026 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| RadAgent | CT, Apr 2026 | 🟡 masks only | ✅ | — | — | 🟡 trace | — | — |
| MedScribe | CT, May 2026 | ✅ | ✅ | — | — | 🟡 retrieval | ✅ | — |
| 3DMedAgent | CT, ICML 2026 | 🟡 size, HU | — | — | — | 🟡 memory | 🟡 30B option | — |
| NV-Reason-CT | CT, Sep 2026 | — | ✅ | 🟡 refusals | 🟡 bad input | — | ✅ | — |
| MedRegion-CT, Reg2RG | CT, 2025 | 🟡 volumes | — | — | — | 🟡 per region | 🟡 7B | — |
| EXACT | CT, Apr 2026 | 🟡 organ maps | ✅ | — | — | 🟡 heatmaps | — | — |
| CXRAgent, MedRAX | X-ray only | — | — | — | — | 🟡 support / refute | 🟡 MedRAX | — |

✅ has it · 🟡 partly (what is there is named) · — **not described in the paper**, not "proved absent". The marks are our reading of each paper; they need a check against the
full text before any of them goes into a publication (see section 5).

What the columns mean for us:
- **Measures from masks**: numbers (volumes, HU, ratios, counts) computed from segmentation masks and used in the answer.
- **Fixed checklist**: the same list of conditions answered for every scan.
- **Can say "can't tell"**: an explicit answer when the evidence is not enough, instead of forcing present / absent.
- **Image-quality checks**: gates such as image noise, lung inflation, contrast, HU validity and orientation that change what may be answered.
- **Cites evidence per claim**: each answer names the measurement rows (E-numbers), model outputs and limits it rests on.
- **Small, local**: runs on one workstation GPU, no data sent anywhere.
- **Doctor review page**: a page where a clinician sees the scan, the answers and their evidence, and grades each answer.

## 2. The two closest systems

**[RadAgent](https://arxiv.org/abs/2604.15231)** (arXiv 2604.15231) is the most similar idea. A language model calls tools step by step: TotalSegmentator (lobes; an effusion
segmenter), the CT-CLIP classifier and CT-Chat, with a Qwen3-14B model planning and calling the tools. It writes a report and keeps a trace
you can inspect. It beat CT-CHAT by 5.8 macro-F1 points and resisted misleading hints 83.7 % of the time against 58.9 %.
The difference: the language model decides everything and the CT-CLIP classifier gets a full vote. It takes no measurements and never
says "can't tell". It also runs on 8 large GPUs. Think of it as a smart intern who uses tools well but always gives an answer.

**[MedScribe](https://arxiv.org/abs/2605.01779)** (arXiv 2605.01779) is the closest on measurements. It turns TotalSegmentator masks into about 86 numbers: volumes,
HU averages and left / right (and front / back) fractions split along anatomical midlines. It runs locally on Llama 3.1-8B and reaches a
macro-F1 of about 0.32 on CT-RATE (CT-CHAT alone 0.09, MedGemma alone 0.10; MedGemma with the same tools 0.31). On the 10 % of scans with
oversized lung masks, F1 fell 8.9 points. The final language model still decides, with no abstention and no citations inside the report.

## 3. The others in one line each

- **3DMedAgent** (ICML 2026): an agent with size and HU tools and a memory of earlier steps; a 30B option. No fixed checklist, no abstention described.
- **NV-Reason-CT** (Sep 2026): a 3D CT reasoning model with a fixed question list; it can refuse and flag bad input. It is a reader, not a
  measuring system. We use it as one source (AI reader) inside our checklist, never as the only one for "present" (checklist_v3_12).
- **MedRegion-CT, Reg2RG** (2025): cut the volume into regions with an outside segmenter and report per region. Gains over models without
  regions were small (Reg2RG F1 0.253 vs 0.212; MedRegion-CT 0.450 vs 0.435), and lung and pleura were the weakest regions.
- **EXACT** (Apr 2026): zero-shot classifier with 18 organ-masked anomaly maps (heatmaps): a probability and a rough location. No licence stated.
- **CXRAgent, MedRAX**: chest X-ray agents with tools that support or refute a claim. Not CT.
- Supporting study, **DeepTumorVQA**: accurate measurement tools raised a 9B model from 38.5 % to 48.5 %; removing them cut it to 32.3 %;
  10 % noise in the measurements cost 10 points; letting a small model look at image crops hurt. Predicted masks instead of expert masks kept 93 % of the gain.

Figures in sections 2 and 3 are as reported by the papers and collected in
[research/Chest CT model candidates.md](research/Chest%20CT%20model%20candidates.md), which holds the links.

## 4. What is different about lung_ct

1. **The rules decide, not a language model.** Each answer comes from a written, versioned rule (`configs/checklist_v3_12.json`):
   measurement first; a mask alone gives "cannot tell"; "present" from two agreeing sources where no measurement decides.
2. **"Cannot tell" is a normal answer**, with the reason given, and an empty or small predicted mask never means "absent".
3. **Image-quality gates** (HU validity, left / right from anatomy, image noise, lung inflation, contrast) change what may be answered.
4. **Every answer cites its evidence** (E-numbers, model scores, limits) on the page and in the exported CSV.
5. **Models are sources, not judges.** The classifier (COLIPRI) and the AI reader (NV-Reason-CT) are shown; they count only where a rule says so,
   and those rules were tested before use (prompt tests PT01 / PT02, calibration TR01 / TR02 on 400 held-apart patients).
6. **Runs on one RTX 3090, fully local**; nothing is sent to an online service.
7. **A doctor review page** in a fixed format (`docs/REVIEW_PAGE_FORMAT.md`), with grades returned as CSV, and a two-reader review (RS01).

## 5. Limits of this comparison (read before quoting it)

- The comparison is about **design**, not accuracy. lung_ct has no validated accuracy figure: all results so far are on development scans,
  against CT-RATE report labels, which count a finding the report did not mention as an error. MedScribe's 0.32 macro-F1 is not comparable with
  anything we have measured.
- "Not described in the paper" is not "does not have it". Each mark needs a check against the full paper (methods and supplement) before a publication.
- Features we have are not all new: measuring from masks, fixed checklists, local models and clinician feedback each exist elsewhere
  ([PAPER_NOVELTY.md](PAPER_NOVELTY.md), "What is established prior work"). The claim worth testing is the **combination**: rule-based decisions
  with abstention, quality gates and per-claim evidence, checked by clinicians.
- Our own weak points: the rules and cut-offs were fitted on development scans from one hospital; the AI reader and the classifier were trained on
  reports from the same hospital; and how often our "present" answers are right is what the RS01 two-reader review is meant to show.


## 6. What none of the papers we found do, and an honest critique

(Also in the README, "Similar work".)

### What none of the papers we found do

- Fixed rules make the final call from measurements; no language model decides an answer.
- The checklist has separate "cannot tell" and "cannot assess" answers.
- Quality gates block a call when image noise, lung inflation or contrast make it unsafe.
- A classifier score alone never decides. Since checklist_v3_5 it cannot make anything "absent"; since checklist_v3_12 it can count towards
  "present" only as the second vote next to the AI reader, for three conditions, at cut-offs checked on held-apart scans.
- Every answer names the evidence rows (E-numbers) it rests on. When a language model writes the report, a validator checks that every finding
  cites existing evidence IDs and that every number in it comes from a cited row (`src/lung_ct/agent/report.py`); the review pages use the rule-based report.

### Honest critique

- **We are not more accurate yet.** Our recall is low (for example 15 of 63 findings found on test20b), and our development sets are about
  20 patients each, against their thousands. Reviewers will ask for F1 on the full CT-RATE validation set next to MedScribe and NV-Reason-CT.
  Our newness is in safety and traceability, not in scores.
- **"Not described" is not "not done".** The comparison is from reading the papers, not their code. Commercial tools (for example Aidence,
  Coreline, Siemens AI-Rad) and GitHub projects were not searched. Commercial products measure emphysema and calcium routinely, so
  "measures from masks" alone is not new.
- **Measurements are getting common.** MedScribe and 3DMedAgent already feed volumes and HU to a language model. Our real claim is narrower:
  deterministic rules plus abstention.
- **Two items in the grid are not fully checked:** MedScribe's calcium and heart-size features at code level, and the full text of the NV-Reason-CT paper.

## Change log

- 2026-11-27: written from the project lead's comparison table and notes, with figures from `research/Chest CT model candidates.md`.
- 2026-11-27: section 6 added (what no paper does; honest critique) from the project lead's notes; the classifier point updated to checklist_v3_12.
