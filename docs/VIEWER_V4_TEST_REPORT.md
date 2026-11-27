# 2D canvas reviewer page (viewer v4): test report

Date: 2026-11-22. Status: tested on the development workstation only. **Not yet opened on the clinician reviewer's
laptop.** A short usability check on his computer is needed before the next grading batch.

## Why a new viewer

The earlier pages draw the scan with a WebGL library (NiiVue 0.69.0). It rebuilds full-size 3D textures on
every refresh. On an integrated graphics adapter the graphics process grew by gigabytes per window change and
then died. The new page uses no WebGL and no viewer library: it draws one 2D slice at a time on an ordinary
canvas.

## Before and after (valid_832, Edge 158, integrated graphics adapter of this workstation)

| | Viewer v3 (WebGL) | Viewer v4 (2D canvas) |
|---|---|---|
| File size | 128.9 MB | 125.2 MB |
| Browser launch to scan shown | 9.9 s | 4.9 s |
| Browser memory after start-up | 4.2 GB (graphics process 2.9 GB) | 1.0-1.1 GB (graphics process 0.16 GB) |
| After 4 window changes | graphics process 18.4 GB | no change |
| After switching through the outlines | 23-33 GB, then the graphics process died | no change |
| After 200 window changes, 570 scroll steps, 30 view switches, 56 outline switches | not reachable | 0.98 GB; no growth |

An empty Edge window on the same machine uses about 0.7 GB, so the page itself adds roughly 0.3-0.45 GB.

Viewer v3 (`runs/M2_location_first_20261122/*_v3.html`) is **not** a fix for integrated graphics and should not
be sent as one.

## Memory by scan size (integrated graphics, MB of private memory, all browser processes)

"Baseline" = same browser with an empty page. "Peak" = highest value while loading (sampled every 0.5 s).
"Sustained" = a few seconds after loading, after a forced garbage collection. "After use" = the same after the
whole interaction block. Working set was recorded too (see each `result.json`).

| Scan (voxels shown) | File | Browser | Baseline | Peak | Sustained | After use |
|---|---|---|---|---|---|---|
| valid_1179, smallest (512 x 512 x 200) | 47 MB | Chrome / Edge | 439 / 702 | 587 / 949 | 649 / 953 | 664 / 789 |
| valid_541 | 67 MB | Chrome / Edge | 436 / 706 | 749 / 1064 | 662 / 983 | 719 / 821 |
| valid_301 | 66 MB | Chrome / Edge | 439 / 643 | 800 / 1087 | 668 / 914 | 714 / 828 |
| valid_988 | 96 MB | Chrome / Edge | 444 / 677 | 832 / 1033 | 719 / 939 | 780 / 883 |
| valid_832 | 125 MB | Chrome / Edge | 434 / 702 | 1072 / 1289 | 825 / 1041 | 890 / 978 |
| valid_434, 1024 x 1024 matrix | 150 MB | Chrome / Edge | 441 / 707 | 1083 / 1234 | 775 / 1035 | 928 / 951 |
| valid_411, largest (768 x 768 x 253) | 190 MB | Chrome / Edge | 435 / 692 | 1092 / 1338 | 816 / 1136 | 896 / 1005 |

- The page adds about 0.2-0.45 GB over an empty browser, depending on scan size. Peak while loading stayed at or
  below about 1.4 GB in total (highest single value in any run: 1409 MB, largest scan, Edge).
- Retained growth after the interaction block: between -220 and +153 MB. The largest increase (+153 MB) was the
  1024-matrix scan in Chrome. No run showed growth that continued with more interaction.
- The 0.3-0.5 GB figure was a target, not a promise: it holds for the page's own share here, not for the
  whole browser, and not necessarily on other machines.

## Integrity check and memory

The scan is verified in the browser against a checksum and SHA-256 values computed from the original array at
build time. Three modes were measured on valid_832 (Edge, integrated graphics): SHA-256 per 8 MB piece
(default), one SHA-256 over the whole volume, and no SHA-256. Peak memory was 1225-1289, 1179 and 1217 MB.
The differences are within run-to-run variation, so hashing does not create a measurable start-up peak at
this sampling rate. The per-piece mode is kept.

## What was tested

26 test runs, 558 checks; 557 passed. The one failure was a real defect in the first run (a wheel notch moved
1.2 images instead of 1); it was fixed and every later run passed. Runs: `runs/V2_canvas_test_*`.

Geometry (the main risk):
- A phantom defined in patient terms (bright block at the patient's left, front, top) was stored in all 48 axis
  orders and flips with unequal voxel spacing (0.8 x 1.3 x 2.5 mm), using hand-built affines. In Chrome and Edge
  every stored version was drawn as the same picture in axial, coronal and sagittal view, pixel for pixel, with
  the block on the correct screen side, the physical aspect ratio, and "go to mm" landing on the right image.
- Oblique scans are refused at build time with an explicit message (limit 0.5 degrees). They are never shown as
  if axis-aligned. All 58 scans held so far are stored LPS with 0 degrees obliquity, so other orientations are
  covered by the phantom only.
- Real cases: 7 drawn slices per page (3 axial, 2 coronal, 2 sagittal) were compared with SimpleITK's reading
  of the original scan file, which shares no code with the page builder. They are equal voxel for voxel, after
  the clip noted below. Drawn outlines (pleural, pericardial, heart) equal the original mask files pixel for
  pixel.
- "Show in scan": the cross-hair lands within half a voxel of the physical target on every axis.
- Supplementary only: heart centre on the screen-right half, liver centre on the screen-left half.

Behaviour (every real page, Chrome 157 and Edge 158, integrated graphics, opened from file://):
- Start-up: axial, plain CT, lung window, no outline decoded, no WebGL or viewer library in the file.
- Wheel and arrow keys, window presets, contrast by right-drag and by Shift+drag, three views, zoom by
  Ctrl+wheel and buttons, pan by drag, click to move the cross-hair.
- 56 outline switches: never more than one decoded outline (largest 25 MB); outline off frees it.
- 200 window changes, 570 scroll steps, 30 view switches without error.
- Grading on all 20 conditions, CSV export, staged page (step 1 lock, then reveal), pages pre-filled with
  earlier grades.
- CSV round trip: grades exported by the page were built back into a page and exported again; 21 rows x 31
  columns identical (save time excluded).
- A deliberately damaged copy shows "The scan could not be loaded"; grading, the Numbers buttons and CSV export
  still work.

## Revision 2 of the pages (same day)

Changes after review: a "1:1 pixels" button, a "Smooth" on / off toggle, the zoom read-out given as a percentage of
fit with screen pixels per image pixel, and the lobe choices in the grading form shown before an answer is given
(greyed until the answer is Present or Can't tell). All eight pages were rebuilt and re-tested in Chrome and Edge on
integrated graphics: 17 runs, 462 checks, all passed (`runs/V2_canvas_test_r3_*`). Memory was in the same range as
in the table above.

One build round in between was broken: an unescaped apostrophe in the new form hint stopped the grading script, so
the grading column was missing from every page. The browser tests caught it, the line was fixed, and a unit test
now syntax-checks every script embedded in the pages. The table above and the first-round totals are from before
this revision.

## Limits

- **No person has used the page yet.** An automated session in a visible window stalled on this desktop (the
  browser stopped answering input, most likely because the window was not in the foreground), so the
  interactive session is still to be done by a person. All other runs were headless.
- Tested on one workstation (128 GB RAM) on its integrated AMD adapter. The clinician reviewer's laptop has Intel graphics
  and much less memory. Compatibility with his laptop, or with laptops in general, is not claimed.
- Touchpad scrolling and pinch-zoom are handled in the code but were tested only through synthetic wheel events.
- Voxel values are the prepared display array used by every earlier page: the original HU clipped to
  -1024..3071. This clip existed before; values outside it (dense metal) show at the limit.
- No linked three-pane view in this version. Axial, coronal and sagittal are separate views.
- Zoom is shown as a percentage of fit-to-window, with the screen pixels per image pixel next to it. "1:1 pixels" draws one image
  pixel per screen pixel; "Smooth" off draws unsmoothed pixels for inspection. The stored voxels are never changed.
- The four case pages are open pages (tool answers visible from the start): for usability and discussion only. Independent
  clinician grading must use the staged build (`--blinded`), as in `staged_valid_832_v4.html`.
- Research measurements, checklist rules, grading form, staging and existing grades are unchanged.

## Files

- Pages: `runs/M3_canvas_viewer_20261122/` (open `index.html`): four cases and one staged example.
- Size-test pages: `runs/V2_canvas_size_pages/`.
- Code: `scripts/viewer_v4.py`, `scripts/build_viewer_page.py --viewer v4`, `scripts/test_canvas_viewer.py`,
  `tests/test_viewer_v4.py`.
- Earlier pages are kept: `runs/M1_meeting_pack_20261119/`, `runs/M2_location_first_20261122/`.

## Check to do on the clinician reviewer's laptop (about 5 minutes)

1. Open `index.html`, then valid_832. Does the scan appear, and after how many seconds?
2. Scroll through the images; switch Lung / Soft tissue / Bone several times; change contrast with Shift+drag.
3. Switch to Coronal and Sagittal and back.
4. Click three or four outlines one after another, then "Show in scan" on one row.
5. Enter two grades and press "Save grades (CSV)".
6. Report anything that is slow, stops or looks wrong, and whether the R / L labels match his expectation.
