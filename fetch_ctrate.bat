@echo off
REM Download 10 mixed CT-RATE validation cases (about 2-3 GB) and prepare them for the pipeline.
REM CT-RATE is gated: first accept its terms at https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
REM The token is read from HF_TOKEN or typed here; it is never written to a file.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_TELEMETRY=1
if not defined HF_TOKEN (
  echo HF_TOKEN is not set. Paste your Hugging Face token ^(read access^) and press Enter:
  set /p HF_TOKEN=
)
set LOG=runs\test_logs\ctrate_fetch.log
echo Downloading and converting CT-RATE cases (log: %LOG%) ...
environments\vlm\Scripts\python.exe scripts\ctrate_fetch.py --n 10 --normals 2 > %LOG% 2>&1
echo Finished with exit code %ERRORLEVEL%  (0 = ok)
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern 'selected|valid_|done:|FAIL|rror|403|401|gated' | Select-Object -Last 30"
set HF_TOKEN=
echo.
pause
