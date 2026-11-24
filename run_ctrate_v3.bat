@echo off
REM Checklist v3 experiment (see docs/HANDOFF.md). Usage:
REM   run_ctrate_v3.bat download   - download soft (standard-kernel) scans: the 20 test20 patients + 20 NEW patients (needs HF token, ~5 GB)
REM   run_ctrate_v3.bat soft       - test20_soft: segment, check, classify, agent v2 AND v3, score both  (~1.5-2 h)
REM   run_ctrate_v3.bat test20b    - test20b (new patients): segment, check, classify, agent v3, score     (~1.5-2 h)
REM   run_ctrate_v3.bat all        - everything above in order
REM   run_ctrate_v3.bat download_c - 20 MORE new patients (set test20c, soft scans; needs HF token)
REM   run_ctrate_v3.bat test20c    - test20c: segment, check, classify, agent v3_1, score (fresh test of v3_1)
REM Logs: runs\test_logs\ctrate_v3_<stage>.log . AI readers are skipped (they do not vote in v2/v3).
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_TELEMETRY=1
if "%1"=="download" goto download
if "%1"=="soft" goto soft
if "%1"=="test20b" goto test20b
if "%1"=="all" goto download
if "%1"=="download_c" goto download_c
if "%1"=="test20c" goto test20c
echo Usage: run_ctrate_v3.bat download ^| soft ^| test20b ^| all ^| download_c ^| test20c
exit /b 2

:download
set LOG=runs\test_logs\ctrate_v3_download.log
if not defined HF_TOKEN (
  echo HF_TOKEN is not set. Paste your Hugging Face token ^(read access^) and press Enter:
  set /p HF_TOKEN=
)
echo [download 1/2] soft versions of the test20 patients (log: %LOG%) ...
environments\vlm\Scripts\python.exe scripts\ctrate_fetch.py --source fixed --recon soft --same-patients-as test20 --set-name test20_soft > %LOG% 2>&1
echo [download 2/2] 20 NEW patients, soft versions ...
environments\vlm\Scripts\python.exe scripts\ctrate_fetch.py --source fixed --recon soft --n 20 --normals 4 --seed 20261118 --min-per-target 3 --exclude-used --set-name test20b >> %LOG% 2>&1
set HF_TOKEN=
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern 're-fetching|excluding|NO soft|done:|FAIL|rror' | Select-Object -Last 12"
if not "%1"=="all" exit /b 0

:soft
set HF_HUB_OFFLINE=1
set LOG=runs\test_logs\ctrate_v3_soft.log
for /f "delims=" %%d in ('dir /b /ad /o-d runs\P1_qc_ctrate_test20_soft_*') do (set QC=runs/%%d& goto :soft_qc)
:soft_qc
echo [soft] QC run %QC% (log: %LOG%)
call :prep %QC% test20_soft
.venv\Scripts\python.exe scripts\agent_demo.py --qc-run %VQ% --all-from-qc --no-readers --classifier-run %CL% --checklist-config configs/checklist_v2.json --no-report --pages-out runs/test_logs/pages_test20soft_v2.txt >> %LOG% 2>&1
.venv\Scripts\python.exe scripts\score_checklist.py --pages-file runs/test_logs/pages_test20soft_v2.txt --label test20soft_v2 >> %LOG% 2>&1
.venv\Scripts\python.exe scripts\agent_demo.py --qc-run %VQ% --all-from-qc --no-readers --classifier-run %CL% --checklist-config configs/checklist_v3.json --pages-out runs/test_logs/pages_test20soft_v3.txt >> %LOG% 2>&1
.venv\Scripts\python.exe scripts\score_checklist.py --pages-file runs/test_logs/pages_test20soft_v3.txt --label test20soft_v3 >> %LOG% 2>&1
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern 'verified,|\*\*all|E6_|rror|FAIL' | Select-Object -Last 12"
if not "%1"=="all" exit /b 0

:test20b
set HF_HUB_OFFLINE=1
set LOG=runs\test_logs\ctrate_v3_test20b.log
for /f "delims=" %%d in ('dir /b /ad /o-d runs\P1_qc_ctrate_test20b_*') do (set QC=runs/%%d& goto :b_qc)
:b_qc
echo [test20b] QC run %QC% (log: %LOG%)
call :prep %QC% test20b
.venv\Scripts\python.exe scripts\agent_demo.py --qc-run %VQ% --all-from-qc --no-readers --classifier-run %CL% --checklist-config configs/checklist_v3.json --pages-out runs/test_logs/pages_test20b_v3.txt >> %LOG% 2>&1
.venv\Scripts\python.exe scripts\score_checklist.py --pages-file runs/test_logs/pages_test20b_v3.txt --label test20b_v3 >> %LOG% 2>&1
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern 'verified,|\*\*all|E6_|rror|FAIL' | Select-Object -Last 12"
exit /b 0

:download_c
set LOG=runs\test_logs\ctrate_v3_download_c.log
if not defined HF_TOKEN (
  echo HF_TOKEN is not set. Paste your Hugging Face token ^(read access^) and press Enter:
  set /p HF_TOKEN=
)
echo [download_c] 20 MORE new patients, soft versions (log: %LOG%) ...
environments\vlm\Scripts\python.exe scripts\ctrate_fetch.py --source fixed --recon soft --n 20 --normals 4 --seed 20261119 --min-per-target 3 --exclude-used --set-name test20c > %LOG% 2>&1
set HF_TOKEN=
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern 'excluding|NO soft|done:|FAIL|rror' | Select-Object -Last 8"
exit /b 0

:test20c
set HF_HUB_OFFLINE=1
set LOG=runs\test_logs\ctrate_v3_test20c.log
for /f "delims=" %%d in ('dir /b /ad /o-d runs\P1_qc_ctrate_test20c_*') do (set QC=runs/%%d& goto :c_qc)
:c_qc
echo [test20c] QC run %QC% (log: %LOG%)
call :prep %QC% test20c
.venv\Scripts\python.exe scripts\agent_demo.py --qc-run %VQ% --all-from-qc --no-readers --classifier-run %CL% --checklist-config configs/checklist_v3_1.json --pages-out runs/test_logs/pages_test20c_v3_1.txt >> %LOG% 2>&1
.venv\Scripts\python.exe scripts\score_checklist.py --pages-file runs/test_logs/pages_test20c_v3_1.txt --label test20c_v3_1 >> %LOG% 2>&1
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern 'verified,|\*\*all|E6_|rror|FAIL' | Select-Object -Last 12"
exit /b 0

REM ---- shared: segmentation, anatomy check, classifier; sets VQ and CL ----
:prep
echo   segmenting ...
environments\segmentation\Scripts\python.exe scripts\segment_totalseg.py --qc-run %1 --n 40 --extra-total heart liver aorta --extra-task pleural_pericard_effusion >> %LOG% 2>&1
echo   anatomy check ...
.venv\Scripts\python.exe scripts\anatomy_contrast_check.py --qc-run %1 >> %LOG% 2>&1
for /f "delims=" %%d in ('dir /b /ad /o-d runs\P1_qc_verified_P1_qc_ctrate_%2_*') do (set VQ=runs/%%d& goto :prep_vq)
:prep_vq
echo   classifier on %VQ% ...
environments\colipri\Scripts\python.exe scripts\agent_classify.py --qc-run %VQ% --n 40 >> %LOG% 2>&1
for /f "delims=" %%d in ('dir /b /ad /o-d runs\P3_colipri_*') do (set CL=runs/%%d& goto :prep_cl)
:prep_cl
echo   agent ...
exit /b 0
