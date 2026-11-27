@echo off
REM Re-convert the 10 corrected CT-RATE cases from the local download (offline, no token), then re-segment
REM the 3 cases that came out blank (valid_5, valid_27, valid_638) and re-run the anatomy check. About 10 min.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set LOG=runs\test_logs\ctrate_fix_hu.log
echo [1/3] Re-converting (log: %LOG%) ...
environments\vlm\Scripts\python.exe scripts\ctrate_fetch.py --n 10 --normals 2 --source fixed > %LOG% 2>&1
if errorlevel 1 (echo Conversion failed - see the log. & pause & exit /b 1)
for /f "delims=" %%d in ('dir /b /ad /o-n runs\P1_qc_ctrate_10cases_*') do (set QC=runs/%%d& goto :got)
:got
echo     new QC run: %QC%
echo [2/3] Re-segmenting the 3 blank cases ...
environments\segmentation\Scripts\python.exe scripts\segment_totalseg.py --qc-run %QC% --cases valid_5_a_1 valid_27_a_1 valid_638_a_1 --extra-total heart liver aorta --extra-task pleural_pericard_effusion >> %LOG% 2>&1
if errorlevel 1 echo Segmentation reported a problem - see the log.
echo [3/3] Anatomy check and contrast detection (all 10) ...
.venv\Scripts\python.exe scripts\anatomy_contrast_check.py --qc-run %QC% >> %LOG% 2>&1
echo Finished.
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern '^\[valid|done:|rror|FAIL' | Select-Object -Last 35"
echo.
pause
