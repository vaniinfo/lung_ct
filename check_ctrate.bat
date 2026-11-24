@echo off
REM Re-run only the anatomy (left/right, front/back, top/bottom) and contrast check (fast, no GPU).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\anatomy_contrast_check.py --qc-run runs/P1_qc_ctrate_10cases_20261115_175650 > runs\test_logs\ctrate_check.log 2>&1
powershell -NoProfile -Command "Get-Content runs\test_logs\ctrate_check.log | Select-String -Pattern '^\[valid|done:'"
echo.
pause
