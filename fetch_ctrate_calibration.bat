@echo off
REM TR02 calibration set: 400 new CT-RATE validation patients, chosen at random without looking at labels.
REM Plan: docs\prompt_tests\TR02_calibration_400_PLAN.md
REM Data go to D:\datasets\ct_rate\calib400 (not into the repository). Everything runs locally.
REM CT-RATE is gated: accept its terms at https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
REM The token is read from HF_TOKEN or typed here; it is never written to a file and is cleared after the download.
REM
REM Steps: 1 download and convert   2 organ masks for the left/right check   3 left/right check
REM        4 NV-Reason-CT answers   5 COLIPRI scores.   No step compares an answer with a label.
REM Safe to start again after an interruption: finished downloads and masks are skipped.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_TELEMETRY=1
set DATA=D:/datasets/ct_rate/calib400
if not defined HF_TOKEN (
  echo HF_TOKEN is not set. Paste your Hugging Face token ^(read access^) and press Enter:
  set /p HF_TOKEN=
)
echo [1/5] Downloading and converting 400 scans (log: runs\test_logs\TR02_1_fetch.log). This is the long step ...
environments\vlm\Scripts\python.exe scripts\ctrate_fetch_calibration.py --root %DATA% > runs\test_logs\TR02_1_fetch.log 2>&1
set FETCH=%ERRORLEVEL%
set HF_TOKEN=
if %FETCH%==2 (
  echo Access was refused. Check the token and that the CT-RATE terms are accepted, then start this file again.
  goto end
)
if not exist "%DATA%\latest_qc_run.txt" (
  echo The download did not finish. See runs\test_logs\TR02_1_fetch.log and start this file again.
  goto end
)
set /p QC=<"%DATA%\latest_qc_run.txt"
echo       download step finished with exit code %FETCH% (0 = all ok, 1 = some scans failed and are listed). QC run: %QC%
set HF_HUB_OFFLINE=1
echo [2/5] Organ masks for the left/right check (log: runs\test_logs\TR02_2_organs.log) ...
environments\segmentation\Scripts\python.exe scripts\segment_orientation_organs.py --qc-run %QC% > runs\test_logs\TR02_2_organs.log 2>&1
echo [3/5] Left/right check (log: runs\test_logs\TR02_3_check.log) ...
.venv\Scripts\python.exe scripts\anatomy_contrast_check.py --qc-run %QC% --seg-root %DATA%/derived --write-scan-list %DATA%/verified_scans.json > runs\test_logs\TR02_3_check.log 2>&1
if not exist "%DATA%\verified_scans.json" (
  echo The check did not write a scan list. See runs\test_logs\TR02_3_check.log
  goto end
)
echo [4/5] NV-Reason-CT, current wording (log: runs\test_logs\TR02_4_nvreason.log) ...
environments\nvreason\Scripts\python.exe scripts\prompt_test_nvreason.py --sets A --scan-list %DATA%/verified_scans.json --label TR02_nvreason_calib400 > runs\test_logs\TR02_4_nvreason.log 2>&1
echo       exit code %ERRORLEVEL%
echo [5/5] COLIPRI scores (log: runs\test_logs\TR02_5_colipri.log) ...
environments\colipri\Scripts\python.exe scripts\prompt_test_colipri.py --scan-list %DATA%/verified_scans.json --scores-only --label TR02_colipri_calib400 > runs\test_logs\TR02_5_colipri.log 2>&1
echo       exit code %ERRORLEVEL%
echo All five steps have run. Nothing was compared with a label; the scoring is a separate step.
:end
echo.
pause
