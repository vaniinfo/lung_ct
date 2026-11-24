@echo off
REM One-case Qwen3-VL test run (coronacases_001). Double-click to run.
REM Output: runs\P3_qwen3vl_1cases_<time>\  Log: runs\test_logs\qwen_test_001.log
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_TELEMETRY=1
echo.
echo Running Qwen3-VL-8B on coronacases_001 ...
echo The FIRST run downloads the model (about 17 GB) into models\qwen3vl - this can take a while.
echo Progress is written to runs\test_logs\qwen_test_001.log
echo.
environments\vlm\Scripts\python.exe scripts\run_qwen.py --qc-run runs/P1_qc_20cases_20261107_223515 --cases coronacases_001 > runs\test_logs\qwen_test_001.log 2>&1
echo Finished with exit code %ERRORLEVEL%  (0 = ok)
echo.
echo ---- last lines of the log ----
powershell -NoProfile -Command "Get-Content runs\test_logs\qwen_test_001.log -Tail 15"
echo.
pause
