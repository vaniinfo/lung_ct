@echo off
REM Fresh CT-RATE test set (test20): 20 NEW patients (none from the 10-case pilot), corrected files,
REM then segmentation, anatomy check, classifier, AI readers (shown only), checklist_v2 agent, scoring.
REM Needs your Hugging Face token for the download (about 5 GB). Total about 3-4 hours.
REM The token is read from HF_TOKEN or typed here; it is never written to a file.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_TELEMETRY=1
set LOG=runs\test_logs\ctrate_test20.log
if not defined HF_TOKEN (
  echo HF_TOKEN is not set. Paste your Hugging Face token ^(read access^) and press Enter:
  set /p HF_TOKEN=
)
echo [1/7] Downloading 20 new cases (log: %LOG%) ...
environments\vlm\Scripts\python.exe scripts\ctrate_fetch.py --source fixed --n 20 --normals 4 --seed 20261117 --min-per-target 3 --exclude-used --set-name test20 > %LOG% 2>&1
set RC=%ERRORLEVEL%
set HF_TOKEN=
set HF_HUB_OFFLINE=1
if not "%RC%"=="0" echo Some downloads failed - see the log; continuing with the rest.
for /f "delims=" %%d in ('dir /b /ad /o-d runs\P1_qc_ctrate_20cases_*') do (set QC=runs/%%d& goto :qc)
:qc
echo     %QC%
echo [2/7] Segmenting (2-4 min per case) ...
environments\segmentation\Scripts\python.exe scripts\segment_totalseg.py --qc-run %QC% --n 20 --extra-total heart liver aorta --extra-task pleural_pericard_effusion >> %LOG% 2>&1
echo [3/7] Anatomy check ...
.venv\Scripts\python.exe scripts\anatomy_contrast_check.py --qc-run %QC% >> %LOG% 2>&1
for /f "delims=" %%d in ('dir /b /ad /o-d runs\P1_qc_verified_P1_qc_ctrate_20cases_*') do (set VQ=runs/%%d& goto :vq)
:vq
echo     %VQ%
echo [4/7] Classifier (COLIPRI) ...
environments\colipri\Scripts\python.exe scripts\agent_classify.py --qc-run %VQ% --n 20 >> %LOG% 2>&1
for /f "delims=" %%d in ('dir /b /ad /o-d runs\P3_colipri_*') do (set CL=runs/%%d& goto :cl)
:cl
echo [5/7] AI readers, checklist question (shown on the page, not counted) ...
environments\vlm\Scripts\python.exe scripts\run_medgemma.py --qc-run %VQ% --n 20 --config configs/medgemma_ct_checklist_v1.json >> %LOG% 2>&1
for /f "delims=" %%d in ('dir /b /ad /o-d runs\P3_vlm_*') do (set MG=runs/%%d& goto :mg)
:mg
environments\hulumed\Scripts\python.exe scripts\run_hulumed.py --qc-run %VQ% --n 20 --config configs/hulumed_ct_checklist_v1.json >> %LOG% 2>&1
for /f "delims=" %%d in ('dir /b /ad /o-d runs\P3_hulumed_*') do (set HM=runs/%%d& goto :hm)
:hm
echo [6/7] Agent, checklist_v2 ...
.venv\Scripts\python.exe scripts\agent_demo.py --qc-run %VQ% --all-from-qc --reader medgemma=%MG% --reader hulumed=%HM% --classifier-run %CL% >> %LOG% 2>&1
echo [7/7] Scoring against the CT-RATE reports ...
.venv\Scripts\python.exe scripts\score_checklist.py --pages-file runs/test_logs/agent_demo_pages.txt >> %LOG% 2>&1
echo Finished.
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern 'excluding|done:|verified|\*\*all|E6_|rror|FAIL' | Select-Object -Last 30"
echo.
pause
