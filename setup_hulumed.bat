@echo off
REM Step 1 of the Hulu-Med test: create environments\hulumed and DOWNLOAD the model.
REM Safe to re-run: skips finished installs and resumes a broken download (up to 6 tries).
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set UV_CACHE_DIR=%~dp0.uv_cache
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONIOENCODING=utf-8
set LOG=runs\test_logs\hulumed_setup.log
set PY=environments\hulumed\Scripts\python.exe
where uv >nul 2>&1 || (echo uv was not found on PATH. Tell Claude. & pause & exit /b 1)
if exist %PY% (
  echo Environment already exists - skipping package installs.
) else (
  echo Creating environments\hulumed and installing packages ...
  (
    uv venv environments\hulumed --python 3.11
    uv pip install --python %PY% torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python %PY% transformers==4.51.2 accelerate==1.7.0 "tokenizers>=0.21,<0.22" "huggingface_hub>=0.30,<1.0" nibabel numpy pillow pyyaml einops timm opencv-python imageio ffmpeg-python
    uv pip install --python %PY% decord
  ) > %LOG% 2>&1
)
uv pip install --python %PY% hf_xet >> %LOG% 2>&1
echo Downloading Hulu-Med-7B (about 16 GB) into models\hulumed - resumes where it stopped ...
for /L %%i in (1,1,6) do (
  echo   attempt %%i ...
  %PY% -c "from huggingface_hub import snapshot_download as s; p=s('ZJU-AI4H/Hulu-Med-7B', revision='258594714a0d3835eb2c9e4cc165a4242e606d71', cache_dir=r'models\hulumed\hub'); print('SNAPSHOT_OK', p)" >> %LOG% 2>&1 && goto :done
  timeout /t 15 >nul
)
echo DOWNLOAD FAILED after 6 attempts - see %LOG%
pause
exit /b 1
:done
echo Download complete.
powershell -NoProfile -Command "Get-Content %LOG% -Tail 3"
echo.
echo Setup done. Next: double-click run_hulumed_test.bat
pause
