@echo off
REM CT-RATE with CORRECTED headers: download the same 10 cases from dataset/valid_fixed (about 2-3 GB),
REM convert, segment, then run the anatomy + contrast check. About 45-60 min in total.
REM The token is read from HF_TOKEN or typed here; it is never written to a file.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_TELEMETRY=1
if not defined HF_TOKEN (
  echo HF_TOKEN is not set. Paste a NEW Hugging Face token ^(read access^) and press Enter:
  set /p HF_TOKEN=
)
set LOG=runs\test_logs\ctrate_fetch_fixed.log
echo [1/3] Downloading and converting from valid_fixed (log: %LOG%) ...
environments\vlm\Scripts\python.exe scripts\ctrate_fetch.py --n 10 --normals 2 --source fixed > %LOG% 2>&1
set RC=%ERRORLEVEL%
set HF_TOKEN=
if not "%RC%"=="0" (echo Download/convert failed - see the log. & pause & exit /b 1)
for /f "delims=" %%d in ('dir /b /ad /o-n runs\P1_qc_ctrate_10cases_*') do (set QC=runs/%%d& goto :got)
:got
echo     new QC run: %QC%
echo [2/3] Segmenting (about 2-4 min per case) ...
environments\segmentation\Scripts\python.exe scripts\segment_totalseg.py --qc-run %QC% --n 10 --extra-total heart liver aorta --extra-task pleural_pericard_effusion >> %LOG% 2>&1
if errorlevel 1 echo Segmentation reported a problem - see the log.
echo [3/3] Anatomy check and contrast detection ...
.venv\Scripts\python.exe scripts\anatomy_contrast_check.py --qc-run %QC% >> %LOG% 2>&1
echo Finished.
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern 'volume source|^\[valid|done:|rror|FAIL|401|403|404|gated' | Select-Object -Last 35"
echo.
pause
