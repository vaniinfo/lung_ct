# Setup, running and tests

Moved out of the README on 2026-11-27 (project lead's request); content unchanged.
The README describes what the tool is and how it works; this file says how to install and run it.

## Setup (Windows 11 + NVIDIA GPU; tested on one RTX 3090)

Python 3.11. Data, model weights and run outputs (`data/`, `models/`, `runs/`) are not in the repository. Keep everything on D: (C: is nearly full).

```bash
git clone https://github.com/vaniinfo/lung_ct.git
cd lung_ct
python -m venv .venv && source .venv/Scripts/activate && pip install -e ".[test]"     # core environment
```

GPU environments, one per model, each run as a separate process:

```bash
# segmentation: TotalSegmentator + CUDA 12.4 PyTorch
uv venv environments/segmentation --python 3.11
uv pip install --python environments/segmentation/Scripts/python.exe torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python environments/segmentation/Scripts/python.exe totalsegmentator==2.18.0
```

```
setup_colipri.bat      environments\colipri  + COLIPRI weights (models\colipri)
setup_nvreason.bat     environments\nvreason + NV-Reason-CT weights, about 10 GB (models\nvreason), checksum verified
setup_nvsegment.bat    environments\nvsegment + NV-Segment-CT weights, about 0.9 GB (models\nvsegment), checksum verified
```

The CT-RATE downloaders run in `environments/vlm` (transformers, nibabel, huggingface_hub; see the history file for its exact versions).
On Windows, TotalSegmentator needs BLAS threads capped at 1 (the scripts set this).

## Running it

**New CT-RATE scans** (download, convert, check orientation): `fetch_ctrate_calibration.bat` (asks for the token), or `scripts/ctrate_fetch.py`
followed by `scripts/anatomy_contrast_check.py`.

**Full pipeline on verified scans** (all ten steps, each output kept):

```bash
bash run_full_pipeline.sh runs/P1_qc_verified_<...> <label> <pack name> [start step]
```

Steps 8 and 10 of the script still use checklist_v3_11; to apply the current policy to finished runs and build the pages:

```bash
python scripts/rescore_with_inventory_rows.py --pages-file runs/test_logs/pages_<label>.txt --rows runs/E19_nodule_rows_<label> \
    --config configs/checklist_v3_12.json --compare configs/checklist_v3_11.json --reader NV-Reason-CT=runs/<P3_nvreason run> --out runs/E19_nodule_rows_<label>/rescore_v3_12.json
python scripts/build_policy_pack.py --pack <M..> --folder runs/<M.._policy_v3_12_label> --checklist checklist_v3_12 --rows runs/E19_nodule_rows_<label> \
    --reference-columns --viewer v7 --reader-run runs/<P3_nvreason run> --second-seg-run runs/<P2_nvsegment run> --cases-file runs/test_logs/cases_<label>.json
```

**Page checks** (real browser, file://):

```bash
python scripts/test_slab_viewer.py --page runs/<pack>/<page>.html --browser edge --gpu-mode igpu --viewer-build checklist_viewer_v7_canvas2d_axial25_mip10_two_segmentations
python scripts/test_seg_viewer.py  --page runs/<pack>/<page>.html --second-seg runs/<P2_nvsegment run> --browser edge
```

**Returned grades**: CSV files from the pages go into `runs/grades_received/`; `scripts/collect_grades.py` gathers them.

## Tests

```bash
python -m pytest -q          # 367 tests, core environment, no GPU
```
