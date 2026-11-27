# Lung CT Research - Project Guidelines

## Project Scope
Reproducible local research pipeline: ingest public 3D chest CT, validate geometry/intensity,
segment lung anatomy, calculate measurements, run CT-capable LVLM, generate reviewable case reports.

Initial scope: public data, pretrained inference, 1 -> 5 -> 20 case feasibility batch.
No model training, commercial deployment, autonomous diagnosis, or clinical validation claims.

## Phase Sequence
- Phase 0: Environment detection, bootstrap, GPU validation
- Phase 1: Data ingestion and QC (COVID-19 CT dataset from Zenodo)
- Phase 2: Baseline segmentation (TotalSegmentator) + volume quantification
- Phase 3: Local MedGemma 1.5 4B CT inference
- Phase 4: Reviewable outputs and feasibility experiments (E0-E4)
- Phase 5: Airway measurement extension in Python (lung volumes, airway centrelines,
  branch lengths, tortuosity); MATLAB is unavailable and is not a dependency

## Hardware
- Windows 11 Pro, AMD Ryzen ~4300MHz, 128GB RAM
- NVIDIA RTX 3090 24GB VRAM, Driver 610.88, CUDA UMD 13.3
- D: drive has 6.8TB free (all data/models here), C: drive nearly full

## Geometry Requirements
- Preserve NIfTI affine, voxel spacing, direction, origin, intensity scaling
- LPS/RAS conversion must be explicit; laterality cannot be inferred from array index
- Interpolate CT with appropriate method, labels with nearest-neighbor
- Validate geometry before any overlay operation
- Volume in mL = voxel_count * voxel_volume_mm3 / 1000

## Validation Rules
- Never assume all NIfTI preserves HU; verify intensity scale per case
- Never manufacture HU with arbitrary min-max scaling
- Exclude display-scaled volumes from HU-dependent analysis
- Reference masks are reference, not predictions - keep clearly separated
- Empty masks and failed runs must be reported, never silently omitted
- Infection masks are NOT automatic fibrosis/GGO labels

## Airway Measurement Rules
- Stack: SimpleITK/nibabel (geometry, physical coordinates), scikit-image (3D skeletonisation),
  Skan (branch paths, lengths, endpoint distances), explicit documented definitions for
  branch count, generation and tortuosity. No MATLAB.
- AirQuant is a methodological reference only: label outputs as our implementation
  ("lung_ct airway tree"), never claim equivalence to AirQuant or use its name for our results
- Branch = edge between endpoints/bifurcations; document pruning, contraction, root and
  generation convention; lengths in physical mm (never voxel counts); report staircase correction
- Validate on synthetic tubes/bifurcations first, then inspect real CT overlays
- Diameter and tapering: later, separately validated extension. Never present a
  distance-transform radius as a validated airway diameter. Wall thickness: deferred
- Fewer detected branches do not indicate airway loss (visibility depends on segmentation,
  resolution, inspiration, reconstruction)

## Definition of Done
Each phase must produce:
- Tested, runnable commands documented in README
- docs/STATUS.md updated with: implemented, tested, blocked, next actions
- Run artifacts in runs/ with full provenance (timestamps, versions, hashes)
- Tests passing for that phase's functionality

## Environment Isolation
- `core`: image ingestion, QC, metrics, reports, tests
- `segmentation`: TotalSegmentator + PyTorch CUDA
- `vlm`: MedGemma, Transformers, Accelerate, quantization
- Exchange NIfTI and JSON/CSV on disk between environments
- Run GPU stages serially in separate processes

## Data Storage
- No fixed size limit on new data or model weights (the 20 GB / 30 GB limits of the initial scope were removed by the project lead on 2026-11-26; D: had 6.5 TB free)
- New downloaded datasets go under D:/datasets/<dataset>/ (project lead, 2026-11-26), outside the repository; data already in data/raw stays where it is
- All data, models and environments go on D:; never on C: (nearly full)
- Record the size and source of every download in the run manifest
- COVID CT dataset ~1.1GB (20 cases + masks)
- Do not mirror CT-RATE in full

## Security
- Never send CT images/masks/reports to external endpoints
- Never commit credentials, tokens, or secrets
- Keep HF tokens in environment variables only
- No automatic public repository push

## Review Pages
- Every page sent to a reader follows docs/REVIEW_PAGE_FORMAT.md and is built with scripts/build_viewer_page.py; changes to that format need the project lead's approval first
