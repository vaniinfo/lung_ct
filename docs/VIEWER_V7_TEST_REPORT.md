# Viewer v7: outlines of two segmentation models - technical test report (2026-11-26)

Technical checks only. They show that the page draws what the mask files hold, in the right place and colour.
They do not show that either segmentation is anatomically right, and they are not clinician usability feedback.

## What v7 adds to v6

A second bar under the outline bar, "Segmentation models (predictions; one at a time)":

| Group | Buttons | Source |
|---|---|---|
| TotalSegmentator (used by the tool) | Lung lobes, Trachea, Airways, Heart + aorta | the masks the measurements already use |
| NV-Segment-CT (second model, comparison only) | All structures, Lung lobes, Trachea, Heart + great vessels, Bones, Other organs, Airway, Class 'lung tumor', Other tumour / vessel classes | `scripts/run_nvsegment.py` |

- One outline at a time, on native slices only (as in v6). The bar is hidden in "Axial 2.5 mm" and "MIP 10 mm".
- The same structure has the same colour in both models. The structure under the cross-hair is named under the scan.
- A class with no voxels gets no button; the bar says "Nothing outlined: ...".
- Nothing else changes: opening view, slabs, grading form, CSV export, checklist answers.

## Conventions

- Each model is stored as label maps: one byte per voxel = class number, on the page's grid, cropped to its bounding box, lossless.
  `tsseg` and `tsaw` (TotalSegmentator; airways are a separate map because they lie inside the lobes), `nvseg` (the "segment everything" map),
  `nvsegx` (the seven extra classes).
- The extra classes can overlap each other. They are shown from one map with a stated priority (class 'lung tumor', airway, bone lesion,
  hepatic tumor, hepatic vessel, pancreatic tumor, colon cancer primaries); the number of overlapping voxels is recorded in the page
  (22,509 on valid_301, 2,972 on valid_832, 0 on the other two).
- Before a page is built, every mask file must have the scan's shape and affine; otherwise the build stops.
- A label map is decoded when its button is first pressed. Buttons of the same map share the decoded copy. Choosing another map, or a slab view, releases it.

## Results

| Test | Result |
|---|---|
| Synthetic volumes, six storage orders / flips, 9 buttons x 16 images (axial, coronal, sagittal) each | 49 of 49 checks; 166,320 pixels per variant, 0 class and 0 colour mismatches |
| Four real pages (pack M14), every button against the ORIGINAL mask files read with SimpleITK, 11 images each | 8 of 8 checks on each page; 0 class and 0 colour mismatches |
| Left-lung classes on the image side marked L | pass on all four |
| One decode per label map; released when switched off | pass on all four |
| v6 page checks (slabs against the original CT, zoom, scrolling, grading, CSV export) run on the v7 pages | 16 of 16 on all four |
| Pack M13 files | unchanged (hashes equal PACK_INFO.json) |

Edge, headless, from file://, integrated graphics (`--gpu-mode igpu`).

| Page | File size | Largest label map, decoded | Decode time | One axial image without / with all structures | Memory after four rounds through all buttons, against start-up |
|---|---|---|---|---|---|
| valid_301 | 67.7 MB (+1.2) | 47 MB | 55 ms | 2.2 / 3.4 ms | not higher |
| valid_541 | 67.4 MB (+0.9) | 33 MB | 37 ms | 3.0 / 5.3 ms | not higher |
| valid_832 | 127.2 MB (+2.0) | 99 MB | 108 ms | 3.6 / 5.7 ms | not higher |
| valid_988 | 97.0 MB (+1.4) | 71 MB | 75 ms | 2.7 / 4.9 ms | not higher |

"+" is the growth of the file against the M13 page of the same case.

## Not tested

- A reader's laptop. The clinician reviewer's earlier crash is not shown to be fixed or affected by this.
- Whether three rows of buttons above the scan are acceptable on a small screen: the bars take more height than in v6.
- Chrome and Firefox for this version.
- Anatomical correctness of any outline.

Run folders: `runs/V12_seg_test_*`. Scripts: `scripts/test_seg_viewer.py`, `scripts/test_slab_viewer.py --viewer-build ...`.
