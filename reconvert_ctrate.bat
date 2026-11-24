@echo off
REM Re-convert the already-downloaded CT-RATE cases (offline, no token needed).
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set LOG=runs\test_logs\ctrate_reconvert.log
echo Re-converting CT-RATE cases from the local download (log: %LOG%) ...
environments\vlm\Scripts\python.exe scripts\ctrate_fetch.py --n 10 --normals 2 > %LOG% 2>&1
echo Finished with exit code %ERRORLEVEL%  (0 = ok)
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern '^\[|done:|rror' | Select-Object -Last 15"
echo.
pause
