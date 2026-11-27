@echo off
REM Hulu-Med-7B on all 10 HU cases (coronacases_001-010). About 15 s per case after loading.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
echo Running Hulu-Med-7B on 10 cases ... (log: runs\test_logs\hulumed_10cases.log)
environments\hulumed\Scripts\python.exe scripts\run_hulumed.py --qc-run runs/P1_qc_20cases_20261107_223515 --n 10 > runs\test_logs\hulumed_10cases.log 2>&1
echo Finished with exit code %ERRORLEVEL%  (0 = ok)
powershell -NoProfile -Command "Get-Content runs\test_logs\hulumed_10cases.log -Tail 12"
echo.
pause
