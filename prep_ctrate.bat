@echo off
REM CT-RATE prep: TotalSegmentator (lobes, airways, vessels + heart, liver, aorta, pleural/pericardial
REM effusion), then the left/right anatomy check and contrast detection. About 2-4 min per case.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set QC=runs/P1_qc_ctrate_10cases_20261115_175650
set LOG=runs\test_logs\ctrate_prep.log
echo [1/2] Segmenting 10 CT-RATE cases (log: %LOG%) ...
environments\segmentation\Scripts\python.exe scripts\segment_totalseg.py --qc-run %QC% --n 10 --extra-total heart liver aorta --extra-task pleural_pericard_effusion > %LOG% 2>&1
if errorlevel 1 echo Segmentation reported a problem - see the log.
echo [2/2] Anatomy (left/right) check and contrast detection ...
.venv\Scripts\python.exe scripts\anatomy_contrast_check.py --qc-run %QC% >> %LOG% 2>&1
echo Finished.
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern '^\[valid|done:|rror|FAIL' | Select-Object -Last 30"
echo.
pause
