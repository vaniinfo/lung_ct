@echo off
REM Run the three image readers on the VERIFIED CT-RATE cases (cases that failed the anatomy check are skipped):
REM MedGemma 1.5, Hulu-Med-7B, COLIPRI classifier. Offline, GPU. About 30-45 min in total.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
for /f "delims=" %%d in ('dir /b /ad /o-n runs\P1_qc_verified_P1_qc_ctrate_*') do (set QC=runs/%%d& goto :got)
:got
echo Using verified QC run: %QC%
set LOG=runs\test_logs\ctrate_readers.log
echo [1/3] MedGemma (log: %LOG%) ...
environments\vlm\Scripts\python.exe scripts\run_medgemma.py --qc-run %QC% --n 10 > %LOG% 2>&1
echo [2/3] Hulu-Med ...
environments\hulumed\Scripts\python.exe scripts\run_hulumed.py --qc-run %QC% --n 10 >> %LOG% 2>&1
echo [3/3] COLIPRI classifier ...
environments\colipri\Scripts\python.exe scripts\agent_classify.py --qc-run %QC% --n 10 >> %LOG% 2>&1
echo Finished.
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern 'done|rror|FAIL|-> ' | Select-Object -Last 20"
echo.
pause
