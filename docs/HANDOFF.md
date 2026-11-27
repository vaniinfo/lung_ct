# Handoff: continuing the lung_ct work in Claude Code (VS Code)

Written 2026-11-17 at the end of a long Claude chat session, so a new Claude Code session on this
PC can pick up without re-reading the chat. Read this, then `docs/PROJECT_PLAN.md` (revision log at
the top is the most up-to-date history).

## Who and what
- Grounded lung CT agent for a university research group; one clinician reviewer. Research use only.
- The user prefers plain explanations (about 11th-grade level) with concrete examples, specifics kept.
- GPU: one RTX 3090 (24 GB), Windows. Each GPU model has its own environment under `environments/`
  (`segmentation`, `vlm` = MedGemma, `hulumed`, `colipri`); core code runs in `.venv`.
  GPU stages run one after another, in separate processes.

## Rules that must not be broken
- Never send CT images, masks or reports to any online service or API. Everything runs locally.
- Hugging Face token: environment variable only (`HF_TOKEN`), never written to a file or log.
- Commit or push only when the user asks; no public push. Never commit credentials or data.
- Check every scan: HU valid, left/right from anatomy (heart left, liver right), never from array order.
- Do not tune thresholds on the cases you report results on (see "test sets" below).

## Pipeline (as of 2026-11-17; superseded: the current flow is in docs/PIPELINE_FLOW.md)
0. QC: HU, anatomy-based orientation check, contrast (`scripts/anatomy_contrast_check.py`)
1. Measurements: TotalSegmentator masks -> evidence rows (`src/lung_ct/agent/evidence.py`,
   `src/lung_ct/agent/conditions.py`): lungs/lobes, dense-lung proxy, LAA-950, image noise (E69),
   contrast (E70), effusions (E71-E73), heart ratio (E74-E77), aortic calcium (E78),
   ground-glass / consolidation ranges outside the back-of-lung band (E80-E87), dependent band (E90).
2. Classifier: COLIPRI zero-shot, 19 findings, uncalibrated (`scripts/agent_classify.py`).
3. AI readers: MedGemma 1.5 4B and Hulu-Med-7B answer a checklist question
   (`configs/*_ct_checklist_v1.json`, schema in `src/lung_ct/vlm/checklist_schema.py`).
   In the pilot both said "absent" to everything, so in checklist_v2 they are shown, not counted.
4. Checklist: `src/lung_ct/agent/checklist.py` + `configs/checklist_v2.json` (20 conditions,
   present / absent / cannot tell / cannot assess; measurements decide; classifier alone can only
   support "absent"; pulmonary embolism = cannot assess on non-contrast).
5. Report: MedGemma text-only (`scripts/agent_write_report.py`), must cite evidence ids, word
   "opacity" banned; rule-based fallback.
Output: `review_checklist.html` per case (`scripts/agent_render_checklist.py`), scores vs CT-RATE
labels (`scripts/score_checklist.py`). Orchestrator: `scripts/agent_demo.py`.

## Data
- CT-RATE validation split only, from `dataset/valid_fixed` (the original `dataset/valid` files have
  placeholder headers: front/back and left/right lost). Fixed files are already HU with -8192
  padding; `scripts/ctrate_fetch.py` handles this.
- `data/raw/ct_rate/cases.json` holds report text + labels for every downloaded case, with `set`:
  - `pilot10` (valid_5, 1037, 27, 583, 1153, 1138, 957, 827, 77, 638): used to build and adjust
    checklist v1 -> v2. Do not report v2 performance on these as a test result.
  - `test20`: 20 new patients (`run_ctrate_test20.bat`, seed 20261117) to judge checklist_v2.
    Run and scored on 2026-11-17, so it is now a seen set: do not adjust rules on it and then
    report on it. Any checklist_v3 change needs a third fresh set.

## Where things stand (updated 2026-11-18)
- Pilot (pilot10, checklist_v1): `runs/E6_checklist_scores_20261117_133752`.
- First honest test (test20, checklist_v2, sharp `_a_1` scans): `runs/E6_checklist_scores_20261117_205523`.
  Of 59 labelled findings: 12 found, 11 missed, 36 cannot tell; 15 extras. Zero extras for pleural
  effusion, pericardial effusion, enlarged heart. Emphysema and aortic calcium tracked image grain;
  smoothing (v2) erased emphysema holes and added ground-glass; lung next to big effusions counted
  as consolidation.
- Cause of the grain: CT-RATE stores two reconstructions per scan; `_a_1` is often the sharp lung
  kernel (Philips YA, Siemens Bl56f/Br60f). 1447 of 1564 validation scans also have a standard
  kernel (Philips B/A, Siemens Br40f). Labels are identical across the two.

## Checklist v3 (built 2026-11-18, `configs/checklist_v3.json`, default in agent_run_case.py)
- Soft reconstruction: `scripts/ctrate_fetch.py --recon soft` (also `--same-patients-as SET`).
- No smoothing. Per-measurement noise limits on blood robust SD (E69): emphysema, calcium,
  ground-glass 45 HU; consolidation 60 HU. Over the limit = not measured = cannot tell.
- Breath-in check: E95 lung volume, E96 median lung HU; poor if > -780 HU or < 3000 mL. Then
  emphysema cannot be absent and ground-glass cannot be present.
- Emphysema on lung minus airways (E31); low value = cannot tell (no "absent" from a low number or
  from the classifier alone); per-lobe LAA (E32) and dark clusters per litre (E33) shown, not used.
- Dense lung within 15 mm of pleural fluid (E88) = compressed lung: removed from consolidation;
  atelectasis present if E88 >= 30 mL.
- 26 tests pass (`tests/test_checklist.py`).

## How to run the v3 experiment (from VS Code)
Copy `docs/vscode_tasks.json` to `.vscode/tasks.json` once (the remote chat tools cannot write to
`.vscode`). Then Terminal > Run Task, or run the bat directly:
1. `run_ctrate_v3.bat download` - needs the Hugging Face token (asked in the terminal; never saved).
   Downloads the soft versions of the 20 test20 patients (set `test20_soft`) and 20 NEW patients
   (set `test20b`, seed 20261118, at least 3 per target finding, 4 normals). About 5 GB.
2. `run_ctrate_v3.bat soft` - test20_soft: segmentation, anatomy check, classifier, then the agent
   with checklist_v2 (no report writer) AND checklist_v3; scores in
   `runs/E6_checklist_scores_test20soft_v2_*` and `..._test20soft_v3_*`.
   Read: v2 sharp (test20) vs v2 soft = effect of the reconstruction alone; v2 soft vs v3 soft =
   effect of the new rules on the same patients (not a fresh test).
3. `run_ctrate_v3.bat test20b` - the honest score for v3: `runs/E6_checklist_scores_test20b_v3_*`.
Logs: `runs/test_logs/ctrate_v3_<stage>.log`. AI readers are skipped (they do not vote since v2).
Each stage takes about 1.5-2 h on the RTX 3090; the agent step also runs MedGemma for the report.

## Checklist v3_1 (2026-11-18, now the default)
- `configs/checklist_v3_1.json` = v3 + heart gate (no 'present' enlarged heart on a shallow breath).
- `scripts/rescore_checklist.py` re-applies a config to finished runs (writes
  `checklist_<version>.json`; score with `score_checklist.py --checklist-file ...`). Use it to see a
  rule change on existing runs; not a fresh test.
- Fresh test: `run_ctrate_v3.bat download_c` (token) then `run_ctrate_v3.bat test20c`
  (VS Code tasks 4 and 5 in `docs/vscode_tasks.json`; re-copy it to `.vscode/tasks.json`).
  Scores: `runs/E6_checklist_scores_test20c_v3_1_*`.

## Checklist v3_2 (2026-11-18, now the default): wording only
- Same decisions as v3_1. 'absent' by measurement -> display 'none measured' + statement with the
  detection limit (`absent_text` per condition in the config); 'absent' from the classifier alone ->
  'not suggested'. The report writer must say 'no measurable ...'; the validator flags 'no X'.

## Disk and meeting pack (2026-11-19)
- The CT-RATE download cache was deleted (15 GB); converted scans are in `data/raw/ct_rate/volumes`.
  Re-converting from the cache now needs a new download (token).
- Meeting pack: `runs/M1_meeting_pack_20261119/index.html` (summary, questions for the doctors) and
  4 case pages. Local only: it contains CT-RATE images and report text.

## Interactive viewer pages (2026-11-19)
- `scripts/build_viewer_page.py`: one self-contained HTML per case with a NiiVue (WebGL2) CT viewer:
  scroll slices, lung / soft-tissue / bone windows, overlays on/off (pleural and pericardial fluid,
  consolidation and ground-glass ranges, compressed lung, aortic calcium, heart, back-of-lung band),
  and checklist rows that jump to their evidence. Viewer code vendored in `scripts/vendor/`
  (NiiVue 0.69.0, BSD-2, sha256 in NIIVUE_LICENSE.txt); works offline.
- Needs about 6 GB RAM (recomputes the overlays with the agent's own code), so run it on the PC:
  `build_meeting_viewers.bat` (VS Code task "Meeting pack: build interactive viewer pages").
- Pages contain CT-RATE images and report text: share only inside the research group.

## Reviewer grading (2026-11-19)
- Viewer pages include a grading panel (`scripts/viewer_grading.py`): per condition the reviewer picks
  present / absent / can't tell, severity, confidence, comment; agreement with the agent is computed;
  "Save grades (CSV)" downloads `grades_<case>_<reviewer>_<date>.csv` (autosave in localStorage).
  Not blinded (decided: testing phase, show everything).
- Older pages: `scripts/add_grading_to_viewer.py` adds the panel without recomputing.
- Collect returned CSVs: `python scripts/collect_grades.py --folder <folder>` ->
  `runs/E7_reviewer_grades_<time>/summary.md` (doctor vs agent, doctor vs CT-RATE labels, comments).

## Next steps
1. After the v3 runs: compare the three score tables, open a few `review_checklist.html` pages
   (an effusion case, an emphysema case, a nodule case), write the results into PROJECT_PLAN.md.
2. Meeting with the clinician reviewer / the project lead: which conditions matter most; is "cannot tell" acceptable;
   would the clinician reviewer score a few cases himself (report labels miss unreported findings).
3. Bigger tools for the "cannot tell" gaps: nodule detector (MONAI lung nodule bundle), lymph-node
   size, bronchiectasis (airway vs artery), fibrosis.
4. Pending: update `docs/PAPER_NOVELTY.md`; commit the work on a branch (only when the user asks).
5. Leftover test folders the user may delete: `runs/A1_agent_valid_5_a_1_20261117_161555`,
   `runs/A1_agent_valid_5_a_1_20261117_235726`, `runs/A1_agent_valid_5_a_1_20261118_121017`,
   `runs/E6_checklist_scores_20261117_162026`, `runs/test_logs/_pages_test.txt`.

## Tests
`.venv\Scripts\python.exe -m pytest -q tests` (checklist tests: `tests/test_checklist.py`).
