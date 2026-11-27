@echo off
REM Condition-checklist version on the 10 verified CT-RATE cases (GPU, offline). About 60-90 min.
REM  1) MedGemma and Hulu-Med answer the checklist question (prompt checklist_v1)
REM  2) agent: measurements (effusions, heart ratio, aortic calcium, ground-glass / consolidation),
REM     checklist, MedGemma report, one-page review_checklist.html per case
REM  3) score the checklist against the CT-RATE report labels
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set QC=runs/P1_qc_verified_P1_qc_ctrate_10cases_20261116_173748_174459
set LOG=runs\test_logs\ctrate_checklist.log
echo [1/4] MedGemma checklist reader (log: %LOG%) ...
environments\vlm\Scripts\python.exe scripts\run_medgemma.py --qc-run %QC% --n 10 --config configs/medgemma_ct_checklist_v1.json > %LOG% 2>&1
for /f "delims=" %%d in ('dir /b /ad /o-n runs\P3_vlm_10cases_*') do (set MG=runs/%%d& goto :mg)
:mg
echo     %MG%
echo [2/4] Hulu-Med checklist reader ...
environments\hulumed\Scripts\python.exe scripts\run_hulumed.py --qc-run %QC% --n 10 --config configs/hulumed_ct_checklist_v1.json >> %LOG% 2>&1
for /f "delims=" %%d in ('dir /b /ad /o-n runs\P3_hulumed_10cases_*') do (set HM=runs/%%d& goto :hm)
:hm
echo     %HM%
echo [3/4] Agent with checklist on 10 cases ...
.venv\Scripts\python.exe scripts\agent_demo.py --qc-run %QC% --reader medgemma=%MG% --reader hulumed=%HM% ^
  --classifier-run runs/P3_colipri_10cases_20261116_180250 ^
  --cases valid_5_a_1 valid_1037_a_1 valid_27_a_1 valid_583_a_1 valid_1153_a_1 valid_1138_a_1 valid_957_a_1 valid_827_a_1 valid_77_a_1 valid_638_a_1 >> %LOG% 2>&1
echo [4/4] Scoring against CT-RATE report labels ...
.venv\Scripts\python.exe scripts\score_checklist.py --pages-file runs/test_logs/agent_demo_pages.txt >> %LOG% 2>&1
echo Finished.
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern 'done:|step 4b|step 5|\*\*all|E6_|rror|failed' | Select-Object -Last 40"
echo.
pause
