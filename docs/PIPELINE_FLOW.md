# ThoraxCheck pipeline flow (current)

Status: written 2026-11-27 for checklist_v3_12 and viewer v7. This file is the one place that says how a scan becomes a review page.
`docs/HANDOFF.md` ("Pipeline") describes the flow of 2026-11-17 and is kept as history. The page itself is defined in
`docs/REVIEW_PAGE_FORMAT.md`. The command that runs all steps is `run_full_pipeline.sh`.

```
CT-RATE scan (D:/datasets/...)  ->  QC (HU, left/right)  ->  segmentation  ->  measurements  ->  classifier + AI reader
        ->  checklist answers (policy checklist_vX)  ->  rule-based report  ->  review page / pack  ->  reader grades (CSV)
```

## 0. Data and QC

| Step | Script | Environment | Output |
|---|---|---|---|
| Download, convert to HU, spacing fix | `ctrate_fetch.py` / `ctrate_fetch_calibration.py` | vlm | `D:/datasets/<set>/volumes`, `cases.json` (report + labels), `runs/P1_qc_*` |
| Left / right and orientation from anatomy (heart left, liver right, heart in front, trachea above) | `segment_orientation_organs.py` (fast masks) or step 1 below, then `anatomy_contrast_check.py` | segmentation, core | `runs/P1_qc_verified_*` |

Only scans that are HU-valid **and** orientation-verified go further. Failures are listed, never dropped.
Scans used by the pipeline are reached through `data/raw/ct_rate/volumes` (hard links for scans stored in `D:/datasets`).

## 1. Ten pipeline steps (`run_full_pipeline.sh <QC run> <label> <pack> [from step]`)

| # | Step | Script | Environment | Output |
|---|---|---|---|---|
| 1 | TotalSegmentator: lobes, trachea, heart, liver, aorta; lung vessels / airways; pleural + pericardial fluid | `segment_totalseg.py` | segmentation | `data/derived/<scan>/segmentation/`, `runs/P2_seg_*` |
| 2 | TotalSegmentator lung nodules | `segment_extra_task.py --task lung_nodules` | segmentation | same folder, `runs/P2_seg_lung_nodules_*` |
| 3 | COLIPRI classifier score per condition (0 to 1, uncalibrated) | `agent_classify.py` | colipri | `runs/P3_colipri_*` |
| 4 | Agent run: evidence rows (measurements E-numbers), checklist v3_9, rule-based report | `agent_demo.py` -> `agent_run_case.py` | core | `runs/A1_agent_<scan>_*` (evidence.json, checklist files, report) |
| 5 | Dense-lung inventory rows E92-E94 | `dense_lung_inventory_replay.py` | core | `runs/E15_dense_lung_inventory_<label>` |
| 6 | Nodule candidate rows E97-E98 | `nodule_evidence_rows.py` | core | `runs/E19_nodule_rows_<label>` |
| 7 | AI reader NV-Reason-CT: free report + 18 yes / no | `run_nvreason.py` | nvreason | `runs/P3_nvreason_*` |
| 8 | Checklist answers with the current policy and the reader | `rescore_with_inventory_rows.py --config configs/checklist_v3_12.json --reader NV-Reason-CT=<run>` | core | `checklist_checklist_v3_12.json` in each agent run |
| 9 | NV-Segment-CT second segmentation (display only) | `run_nvsegment.py` | nvsegment | `runs/P2_nvsegment_*` |
| 10 | Review pack (one page per scan + index) | `build_policy_pack.py --viewer v7 --reference-columns --cases-file ...` -> `build_viewer_page.py` | core | `runs/M<n>_policy_<version>_<label>/` |

GPU steps run one after another in separate processes. Every output is kept, so other report formats can be built later without re-running models.

## 2. How one checklist answer is decided (checklist_v3_12)

For each of the 20 checklist conditions, in this order:

1. **Needs contrast** and the scan is non-contrast -> "cannot assess".
2. **A measurement exists and is decisive** (outside its grey zone, image not too noisy) -> the measurement decides: "present" or "absent"
   (for example aortic calcium, heart-to-chest ratio, ground-glass and consolidation ranges).
   A predicted mask below its limit never gives "absent": it gives "cannot tell" (empty or low outline).
3. **A predicted mask alone is positive** (pleural fluid >= 20 mL outlined, nodule candidates) -> "cannot tell", with what was outlined in the statement.
4. **Two-source rule** (pleural effusion, aortic calcification, coronary calcification only), where the answer is still "cannot tell":
   "present" if NV-Reason-CT says present **and** a second source agrees: COLIPRI above that condition's cut-off
   (0.8197 / 0.0620 / 0.1346), or for pleural effusion >= 20 mL fluid outlined. The answer names both sources.
5. **Otherwise** "cannot tell", with the evidence listed (classifier score, reader answer, measurements).
   A classifier score alone never decides; a reader's "absent" never gives "absent".
6. **Image-quality gates**: poorly inflated lungs block "absent" for emphysema and "present" for ground-glass / enlarged heart.

The exact rule text, limits and cut-offs are in `configs/checklist_v3_12.json` (`decision_rule`, `conditions`, `changes_from_v3_11`).
Code: `src/lung_ct/agent/checklist.py` (`build_checklist`, `_two_source`).

## 3. What the reader sees and sends back

- The page (`docs/REVIEW_PAGE_FORMAT.md`): scan, the tool's answer and its basis, the AI reader's line, the CT-RATE report label, the measurement
  panel, the rule-based report and the AI reader's report; the reader grades each condition.
- Grades come back as CSV (form `grades_v2`) into `runs/grades_received/`; `collect_grades.py` gathers them.

## 4. Experiments that feed the rules (not part of the page flow)

Prompt tests and calibration (`docs/prompt_tests/`): PT01 (COLIPRI wording), PT02 (NV-Reason-CT wording), TR01 / TR02 (cut-offs and trust rules),
RS01 (two-reader review of model against report). A rule change from these is a new checklist version and an entry in `docs/STATUS.md`.

## Change log

- 2026-11-27: written (checklist_v3_12, viewer v7). Note: `run_full_pipeline.sh` still re-scores and builds with checklist_v3_11 in steps 8 and 10;
  it will take the policy as a parameter once the running 36-scan job has finished (a running bash script must not be edited). v3_12 is applied afterwards with step 8's command.
