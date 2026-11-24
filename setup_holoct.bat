@echo off
REM HoloCT comparison (branch holoct-test): clone the code, create environments\holoct, download HoloCT-4B (9.4 GB).
REM Public model (MIT), no Hugging Face token needed or used. Safe to re-run.
cd /d "%~dp0"
if not exist runs\test_logs mkdir runs\test_logs
set UV_CACHE_DIR=%~dp0.uv_cache
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONIOENCODING=utf-8
set LOG=runs\test_logs\holoct_setup.log
set PY=environments\holoct\Scripts\python.exe
set COMMIT=0d3b0a12f5ab0083dbeb81a60c49535404517305
where uv >nul 2>&1 || (echo uv was not found on PATH. Tell Claude. & pause & exit /b 1)
if not exist models\holoct\src (
  echo Cloning HoloCT code at the reviewed commit ...
  git clone https://github.com/alibaba-damo-academy/HoloCT.git models\holoct\src > %LOG% 2>&1
  git -C models\holoct\src checkout %COMMIT% >> %LOG% 2>&1
)
if exist %PY% (
  echo Environment already exists - skipping package installs.
) else (
  echo Creating environments\holoct and installing packages ...
  (
    uv venv environments\holoct --python 3.10
    uv pip install --python %PY% torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python %PY% -r models\holoct\src\requirements.txt hf_xet
  ) >> %LOG% 2>&1
)
echo Downloading HoloCT-4B (9.4 GB) into models\holoct\weights - resumes where it stopped ...
%PY% -c "from huggingface_hub import snapshot_download as s; p=s('Alibaba-DAMO-Academy/HoloCT-4B', local_dir=r'models\holoct\weights\HoloCT-4B', token=False); print('SNAPSHOT_OK', p)" >> %LOG% 2>&1 || (echo DOWNLOAD FAILED - see %LOG% & pause & exit /b 1)
echo Setup done. Next:
echo   environments\holoct\Scripts\python.exe scripts\run_holoct.py --set test20c --findings --grounded
echo   .venv\Scripts\python.exe scripts\score_holoct.py --run runs\P3_holoct_... --compare runs\E6_checklist_scores_test20c_v3_1_...
pause
