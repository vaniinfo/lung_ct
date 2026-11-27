@echo off
REM NV-Segment-CT on ONE CT-RATE scan: all supported classes. About 15-25 seconds after loading.
REM Everything runs on this computer; the scan is not sent anywhere.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
echo Running NV-Segment-CT on valid_301_a_2 ... (log: runs\test_logs\nvsegment_test.log)
environments\nvsegment\Scripts\python.exe scripts\run_nvsegment.py --qc-run runs/P1_qc_verified_P1_qc_ctrate_test20c_20cases_20261118_141946_150359 --cases valid_301_a_2 > runs\test_logs\nvsegment_test.log 2>&1
powershell -NoProfile -Command "Get-Content runs\test_logs\nvsegment_test.log -Tail 4"
pause
