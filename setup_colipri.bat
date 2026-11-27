@echo off
REM Step 1 of adding the finding classifier: create environments\colipri and DOWNLOAD COLIPRI.
REM Nothing from the downloads is executed here. Claude reviews the downloaded code first.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set UV_CACHE_DIR=%~dp0.uv_cache
set HF_HOME=%~dp0models\colipri\hf
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONIOENCODING=utf-8
set LOG=runs\test_logs\colipri_setup.log
set PY=environments\colipri\Scripts\python.exe
where uv >nul 2>&1 || (echo uv was not found on PATH. Tell Claude. & pause & exit /b 1)
if exist %PY% (
  echo Environment already exists - skipping package installs.
) else (
  echo Creating environments\colipri and installing packages ...
  (
    uv venv environments\colipri --python 3.11
    uv pip install --python %PY% torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python %PY% colipri==0.1.3 "transformers==4.51.2" "timm<1.0.23" nibabel hf_xet
  ) > %LOG% 2>&1
)
echo Downloading COLIPRI weights and its tokenizer into models\colipri ...
for /L %%i in (1,1,6) do (
  echo   attempt %%i ...
  %PY% -c "from huggingface_hub import hf_hub_download as h, snapshot_download as s; w=h('microsoft/colipri','model.safetensors'); t=s('microsoft/BiomedVLP-CXR-BERT-specialized', revision='5157bdba1437a3aed316dacb1a5b68edf96b9902', allow_patterns=['*.json','*.txt','*.py','*.safetensors','*.bin']); print('WEIGHTS_OK', w); print('TOKENIZER_OK', t)" >> %LOG% 2>&1 && goto :done
  timeout /t 15 >nul
)
echo DOWNLOAD FAILED after 6 attempts - see %LOG%
pause
exit /b 1
:done
echo Download complete.
powershell -NoProfile -Command "Get-Content %LOG% -Tail 3"
echo.
echo Tell Claude setup is done - it will review the downloaded code before anything runs.
pause
