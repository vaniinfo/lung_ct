@echo off
REM CT-RATE front/back fix: re-convert the downloaded cases with the y axis reversed (offline, no token),
REM then re-segment and re-run the anatomy + contrast check on the NEW QC run. About 30-40 min total.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set LOG=runs\test_logs\ctrate_fix_orientation.log
echo [1/3] Re-converting with the front/back fix (log: %LOG%) ...
environments\vlm\Scripts\python.exe scripts\ctrate_fetch.py --n 10 --normals 2 --flip-ap > %LOG% 2>&1
if errorlevel 1 (echo Conversion failed - see the log. & pause & exit /b 1)
for /f "delims=" %%d in ('dir /b /ad /o-n runs\P1_qc_ctrate_10cases_*') do (set QC=runs/%%d& goto :got)
:got
echo     new QC run: %QC%
echo [2/3] Segmenting (about 2-4 min per case) ...
environments\segmentation\Scripts\python.exe scripts\segment_totalseg.py --qc-run %QC% --n 10 --extra-total heart liver aorta --extra-task pleural_pericard_effusion >> %LOG% 2>&1
if errorlevel 1 echo Segmentation reported a problem - see the log.
echo [3/3] Anatomy check and contrast detection ...
.venv\Scripts\python.exe scripts\anatomy_contrast_check.py --qc-run %QC% >> %LOG% 2>&1
echo Finished.
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern '^\[valid|done:|rror|FAIL|verified' | Select-Object -Last 30"
echo.
pause
