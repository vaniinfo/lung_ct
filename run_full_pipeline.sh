#!/usr/bin/env bash
# Full tool pipeline (the one behind pack M14) on the scans of one verified QC run, then the standard review pack.
# Every intermediate result is kept in runs/ and data/derived/ so other report formats can be built from it later.
#
# Usage (Git Bash, repository root):
#   bash run_full_pipeline.sh runs/P1_qc_verified_rs01_first10_<time> rs01_first10 M15 [FROM_STEP]
# FROM_STEP (default 1) restarts at that step and reuses the newest outputs of the earlier steps.
# Steps (each in its own environment, GPU stages one after another):
#   1 TotalSegmentator: lobes, trachea, heart, liver, aorta, lung vessels / airways, pleural + pericardial effusion
#   2 TotalSegmentator lung_nodules           3 COLIPRI classifier
#   4 agent run with checklist_v3_9 (evidence, measurements, rule-based report)
#   5 dense-lung inventory rows (E92-E94)    6 nodule rows (E97-E98)
#   7 NV-Reason-CT report + 18 yes / no       8 re-score with checklist_v3_11 (AI reader decides 'present' for three conditions)
#   9 NV-Segment-CT (second segmentation, display only)
#  10 review pack (docs/REVIEW_PAGE_FORMAT.md: build_policy_pack.py --viewer v7, open page, report label column)
set -euo pipefail
QC="$1"; LABEL="$2"; PACK="$3"; FROM="${4:-1}"
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
LOGDIR=runs/test_logs/pipeline_${LABEL}; mkdir -p "$LOGDIR"
CASES=$(.venv/Scripts/python.exe -c "import json,sys;print(' '.join(c['case_id'] for c in json.load(open(sys.argv[1]+'/run_manifest.json'))['cases']))" "$QC")
N=$(echo $CASES | wc -w)
latest() { ls -dt runs/$1 2>/dev/null | head -1; }
step() { echo "[$(date +%H:%M:%S)] $*"; }
do_step() { [ "$FROM" -le "$1" ]; }
E15="runs/E15_dense_lung_inventory_${LABEL}"
E19="runs/E19_nodule_rows_${LABEL}"
PAGES="runs/test_logs/pages_${LABEL}.txt"

if do_step 1; then
  step "1/10 segmentation ($N scans)"
  environments/segmentation/Scripts/python.exe scripts/segment_totalseg.py --qc-run "$QC" --cases $CASES \
    --extra-total heart liver aorta --extra-task pleural_pericard_effusion > "$LOGDIR/1_seg.log" 2>&1
fi
if do_step 2; then
  step "2/10 lung nodules"
  environments/segmentation/Scripts/python.exe scripts/segment_extra_task.py --task lung_nodules --cases $CASES > "$LOGDIR/2_nodules.log" 2>&1
fi
if do_step 3; then
  step "3/10 COLIPRI classifier"
  environments/colipri/Scripts/python.exe scripts/agent_classify.py --qc-run "$QC" --cases $CASES > "$LOGDIR/3_classify.log" 2>&1
fi
CL=$(latest "P3_colipri_${N}cases_*")
if do_step 4; then
  step "4/10 agent runs (classifier $CL)"
  .venv/Scripts/python.exe scripts/agent_demo.py --qc-run "$QC" --cases $CASES --no-readers --classifier-run "$CL" \
    --checklist-config configs/checklist_v3_9.json --no-report --pages-out "$PAGES" > "$LOGDIR/4_agent.log" 2>&1
fi
RUNS=$(.venv/Scripts/python.exe -c "import sys,pathlib;print(' '.join(str(pathlib.Path(l.strip()).parent).replace(chr(92),'/') for l in open(sys.argv[1]) if l.strip()))" "$PAGES")
if do_step 5; then
  step "5/10 dense-lung inventory rows"
  .venv/Scripts/python.exe scripts/dense_lung_inventory_replay.py --pages-file "$PAGES" --out "$E15" > "$LOGDIR/5_inventory.log" 2>&1
fi
if do_step 6; then
  step "6/10 nodule rows"
  .venv/Scripts/python.exe scripts/nodule_evidence_rows.py --runs $RUNS --base-rows "$E15" --out "$E19" > "$LOGDIR/6_nodule_rows.log" 2>&1
fi
if do_step 7; then
  step "7/10 NV-Reason-CT"
  environments/nvreason/Scripts/python.exe scripts/run_nvreason.py --qc-run "$QC" --cases $CASES > "$LOGDIR/7_nvreason.log" 2>&1
fi
NVR=$(latest "P3_nvreason_${N}cases_*")
if do_step 8; then
  step "8/10 re-score with checklist_v3_11 (reader $NVR)"
  .venv/Scripts/python.exe scripts/rescore_with_inventory_rows.py --pages-file "$PAGES" --rows "$E19" \
    --config configs/checklist_v3_11.json --compare configs/checklist_v3_10.json --reader "NV-Reason-CT=$NVR" --out "$E19/rescore_v3_11.json" > "$LOGDIR/8_rescore.log" 2>&1
fi
if do_step 9; then
  step "9/10 NV-Segment-CT"
  environments/nvsegment/Scripts/python.exe scripts/run_nvsegment.py --qc-run "$QC" --cases $CASES > "$LOGDIR/9_nvsegment.log" 2>&1
fi
NVS=$(latest "P2_nvsegment_${N}cases_*")
if do_step 10; then
  step "10/10 review pack $PACK"
  .venv/Scripts/python.exe - $RUNS > "runs/test_logs/cases_${LABEL}.json" <<'EOF'
import json, sys
from pathlib import Path
print(json.dumps([{"case": json.loads((Path(r) / "run_manifest.json").read_text())["case_id"], "run": Path(r).name} for r in sys.argv[1:]]))
EOF
  .venv/Scripts/python.exe scripts/build_policy_pack.py --pack "$PACK" --folder "runs/${PACK}_policy_v3_11_${LABEL}" --checklist checklist_v3_11 \
    --rows "$E19" --reference-columns --viewer v7 --reader-run "$NVR" --second-seg-run "$NVS" \
    --cases-file "runs/test_logs/cases_${LABEL}.json" > "$LOGDIR/10_pack.log" 2>&1
fi
step "done: runs/${PACK}_policy_v3_11_${LABEL}"
