@echo off
REM Full grounded-agent pipeline on the 10 verified CT-RATE cases, using the reader and classifier runs
REM already made by run_ctrate_readers.bat. Report writer = MedGemma (GPU). About 30-50 min.
REM Each case gets a one-page review.html with the CT-RATE radiologist report alongside.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set LOG=runs\test_logs\ctrate_agent.log
echo Running the agent on 10 CT-RATE cases (log: %LOG%) ...
.venv\Scripts\python.exe scripts\agent_demo.py --qc-run runs/P1_qc_verified_P1_qc_ctrate_10cases_20261116_173748_174459 ^
  --reader medgemma=runs/P3_vlm_10cases_20261116_175326 ^
  --reader hulumed=runs/P3_hulumed_10cases_20261116_175845 ^
  --classifier-run runs/P3_colipri_10cases_20261116_180250 ^
  --cases valid_5_a_1 valid_1037_a_1 valid_27_a_1 valid_583_a_1 valid_1153_a_1 valid_1138_a_1 valid_957_a_1 valid_827_a_1 valid_77_a_1 valid_638_a_1 > %LOG% 2>&1
echo Finished with exit code %ERRORLEVEL%  (0 = ok)
powershell -NoProfile -Command "Get-Content %LOG% | Select-String -Pattern 'step 4|validation|failed|rror|review.html' | Select-Object -Last 40"
echo.
pause
