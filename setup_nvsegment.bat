@echo off
REM NV-Segment-CT (second segmentation, for comparison with TotalSegmentator): create environments\nvsegment and DOWNLOAD the model (about 0.9 GB, to D:).
REM Safe to re-run: skips finished installs and a finished download. No token is needed (public model).
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set UV_CACHE_DIR=%~dp0.uv_cache
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONIOENCODING=utf-8
set LOG=runs\test_logs\nvsegment_setup.log
set PY=environments\nvsegment\Scripts\python.exe
set REV=afb51518689f71e6abb367ee6301b2cd0225c66a
set SNAP=models\nvsegment\hub\models--nvidia--NV-Segment-CT\snapshots\%REV%
where uv >nul 2>&1 || (echo uv was not found on PATH. Tell Claude. & pause & exit /b 1)
if exist %PY% (
  echo Environment already exists - skipping package installs.
) else (
  echo Creating environments\nvsegment and installing packages ...
  (
    uv venv environments\nvsegment --python 3.11
    uv pip install --python %PY% torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
    uv pip install --python %PY% "monai>=1.5.1" "nibabel>=5.3.3" scipy numpy safetensors huggingface_hub einops
  ) > %LOG% 2>&1
)
echo Downloading settings files ...
%PY% -c "from huggingface_hub import snapshot_download as s; print('FILES_OK', s('nvidia/NV-Segment-CT', revision='%REV%', cache_dir=r'models\nvsegment\hub', ignore_patterns=['*.safetensors','*.pt','*.pth','*.bin']))" >> %LOG% 2>&1 || (echo FAILED - see %LOG% & pause & exit /b 1)
if exist %SNAP%\vista3d_pretrained_model\model.safetensors (
  echo Weights already present.
) else (
  echo Downloading the weights, about 0.9 GB ...
  if not exist models\nvsegment\dl mkdir models\nvsegment\dl
  %PY% scripts\fetch_public_file.py https://huggingface.co/nvidia/NV-Segment-CT/resolve/%REV%/vista3d_pretrained_model/model.safetensors models\nvsegment\dl\model.safetensors 871894112 120ed013722a22780cc01b75cb5c18e4658d69879a983885abf8fa411c9f8f42 >> %LOG% 2>&1 || (echo DOWNLOAD FAILED - see %LOG% & pause & exit /b 1)
  move models\nvsegment\dl\model.safetensors %SNAP%\vista3d_pretrained_model\model.safetensors >nul
)
echo Setup done. Next: double-click run_nvsegment_test.bat
pause
