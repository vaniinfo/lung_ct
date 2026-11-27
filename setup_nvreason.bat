@echo off
REM NV-Reason-CT (third AI reader): create environments\nvreason and DOWNLOAD the model (about 10 GB, to D:).
REM Safe to re-run: skips finished installs, resumes a broken download. No token is needed (public model).
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set UV_CACHE_DIR=%~dp0.uv_cache
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONIOENCODING=utf-8
set LOG=runs\test_logs\nvreason_setup.log
set PY=environments\nvreason\Scripts\python.exe
set REV=386b93e034983f6c1fc841a43833a1b6a0cd9c13
set SNAP=models\nvreason\hub\models--nvidia--NV-Reason-CT\snapshots\%REV%
where uv >nul 2>&1 || (echo uv was not found on PATH. Tell Claude. & pause & exit /b 1)
if exist %PY% (
  echo Environment already exists - skipping package installs.
) else (
  echo Creating environments\nvreason and installing packages ...
  (
    uv venv environments\nvreason --python 3.11
    uv pip install --python %PY% torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
    uv pip install --python %PY% transformers==5.6.2 accelerate==1.13.0 "monai>=1.5.1" "nibabel>=5.3.3" dynamic-network-architectures==0.4.3 scipy numpy Pillow timm einops safetensors huggingface_hub
  ) > %LOG% 2>&1
)
echo Downloading code and settings files ...
%PY% -c "from huggingface_hub import snapshot_download as s; print('FILES_OK', s('nvidia/NV-Reason-CT', revision='%REV%', cache_dir=r'models\nvreason\hub', ignore_patterns=['*.safetensors']))" >> %LOG% 2>&1 || (echo FAILED - see %LOG% & pause & exit /b 1)
if exist %SNAP%\model.safetensors (
  echo Weights already present.
) else (
  echo Downloading the weights, about 10 GB - resumes where it stopped ...
  if not exist models\nvreason\dl mkdir models\nvreason\dl
  %PY% scripts\fetch_public_file.py https://huggingface.co/nvidia/NV-Reason-CT/resolve/%REV%/model.safetensors models\nvreason\dl\model.safetensors 10644167648 655c15e59dfa1957f7d5f475b1702d689fb7107160f21256028391bb5f6e6ba5 >> %LOG% 2>&1 || (echo DOWNLOAD FAILED - see %LOG% & pause & exit /b 1)
  move models\nvreason\dl\model.safetensors %SNAP%\model.safetensors >nul
)
echo Setup done. Next: double-click run_nvreason_test.bat
pause
