@echo off
REM Download the CT-CLIP zero-shot checkpoint (1.77 GB) from the gated CT-RATE repository into models\ct_clip.
REM Only model weights are downloaded; no CT image, mask or report is sent anywhere.
REM First accept the terms at https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
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
set LOG=runs\test_logs\ctclip_fetch.log
echo Downloading CT-CLIP_v2.pt (log: %LOG%) ...
environments\vlm\Scripts\python.exe scripts\ctclip_fetch.py --which base > %LOG% 2>&1
echo Finished with exit code %ERRORLEVEL%  (0 = ok)
type %LOG%
set HF_TOKEN=
echo.
pause
