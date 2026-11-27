@echo off
REM NV-Reason-CT on ONE CT-RATE scan: a report and 18 yes/no questions. About 2-3 minutes after loading.
REM Everything runs on this computer; the scan is not sent anywhere.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
echo Running NV-Reason-CT on valid_434_a_1 ... (log: runs\test_logs\nvreason_test.log)
environments\nvreason\Scripts\python.exe scripts\run_nvreason.py --qc-run runs/P1_qc_verified_P1_qc_ctrate_20cases_20261117_181838_191500 --cases valid_434_a_1 > runs\test_logs\nvreason_test.log 2>&1
powershell -NoProfile -Command "Get-Content runs\test_logs\nvreason_test.log -Tail 6"
pause
