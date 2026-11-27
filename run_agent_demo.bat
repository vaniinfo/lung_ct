@echo off
REM Grounded agent demo: all pipeline steps for the two demo cases.
REM Step 2 classifier (COLIPRI, GPU) -> steps 0,1,3,4 (core) -> step 5 report (MedGemma, GPU) -> review page.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set HF_HOME=%~dp0models\colipri\hf
set LOG=runs\test_logs\agent_demo.log
echo Running the full agent pipeline (log: %LOG%) ...
.venv\Scripts\python.exe scripts\agent_demo.py --cases coronacases_006 coronacases_001 > %LOG% 2>&1
echo Finished with exit code %ERRORLEVEL%  (0 = ok)
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern 'step |done:|REFUS|FAIL|Error|review.html' | Select-Object -Last 25"
for /f "usebackq delims=" %%p in ("runs\test_logs\agent_demo_pages.txt") do (
  start "" "%%p"
  goto :opened
)
:opened
pause
