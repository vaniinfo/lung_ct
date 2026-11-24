@echo off
REM Step 2 of the Hulu-Med test: one case (coronacases_001). Run only after Claude has
REM reviewed the downloaded model code and pinned the revision in configs\hulumed_ct_v1.json.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
echo Running Hulu-Med-7B on coronacases_001 ... (log: runs\test_logs\hulumed_test_001.log)
environments\hulumed\Scripts\python.exe scripts\run_hulumed.py --qc-run runs/P1_qc_20cases_20261107_223515 --cases coronacases_001 > runs\test_logs\hulumed_test_001.log 2>&1
echo Finished with exit code %ERRORLEVEL%  (0 = ok)
powershell -NoProfile -Command "Get-Content runs\test_logs\hulumed_test_001.log -Tail 15"
echo.
pause
