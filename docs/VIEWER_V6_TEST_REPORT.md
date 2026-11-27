# Viewer v6: Axial 2.5 mm and MIP 10 mm for nodule review - test report

Build `checklist_viewer_v6_canvas2d_axial25_mip10` (`scripts/viewer_v6.py`), pack `runs/M12_policy_v3_10_axial25_nvreason_20261125` (2026-11-25).
The pack has not been sent. Viewer v4 and v5 and packs M1 to M11 are unchanged.

This report has two parts that must not be mixed: Part A is what was checked by program, Part B is what the clinician reviewer thinks of it.

## What the page does

| Control | What is shown |
|---|---|
| **Axial 2.5 mm** (opening view) | Average (mean) of neighbouring native slices, about 2.5 mm, lung window, fitted, no outlines |
| **MIP 10 mm** | Brightest value through about 10 mm, lung window |
| **Details** | Exact slab figures, how they are calculated, and a button for the **native thin slices** |

The 5 mm and 8 mm MIP options are gone. Outlines are shown on native slices only; "Show in scan" switches to native slices by itself.

## Source images: spacing and thickness are different things

| Scan | Slice spacing (from the file) | Slice thickness | Other reconstruction in the data set |
|---|---|---|---|
| valid_301_a_2 | 0.75 mm | not in the file or metadata; report text states 1.5 mm | 1.5 mm apart |
| valid_541_a_2 | 1.25 mm | not in the file or metadata; report text and series name state 1.5 mm | 1.25 mm apart |
| valid_832_a_2 | 0.75 mm | not recorded anywhere we hold | 1.5 mm apart |
| valid_988_a_2 | 0.75 mm | not in the file or metadata; report text states 1.5 mm | 1.5 mm apart |

- The CT-RATE metadata table has `ZSpacing` only, no slice-thickness column. The NIfTI header does not carry thickness.
- No scan has a 2.5 mm reconstruction, so the 2.5 mm view is **calculated on the page**. It is not a scanner reconstruction and the page says so.
- The page counts slabs from spacing (always known). It shows the report's thickness figure only as "stated in the radiology report".

## Convention

- `n = max(1, floor(requested_mm / spacing + 0.5))` whole slices, halves up. Coverage is quoted as `n x spacing`.
- If a thickness `t` is stated, Details also shows the tissue covered: `(n - 1) x spacing + t`.
- Mean: exact whole-number sum of the stored CT values (HU), one division, rounded to a whole HU with halves up, **before** the window is applied.
- MIP: maximum of the stored CT values, before the window.
- Slab centred on the current native slice; with an even count the extra slice is on the head side.
- At the scan edge the slab is shortened, never shifted or padded, and the label says so.
- Scrolling moves one native slice per step in every view, so slabs overlap and no slice is skipped.

| Spacing | Axial 2.5 mm | If slices are 1.5 mm thick | MIP 10 mm | If slices are 1.5 mm thick |
|---|---|---|---|---|
| 0.75 mm | 3 slices = 2.25 mm | 3.0 mm | 13 slices = 9.75 mm | 10.5 mm |
| 1.25 mm | 2 slices = 2.5 mm | 2.75 mm | 8 slices = 10.0 mm | 10.25 mm |

Thicker sources: from 1.67 mm spacing one slice is the nearest whole number, so nothing is averaged. The native slice is shown, the button reads
"Axial <spacing> mm", and the label says a 2.5 mm view is not available. If the native slices are themselves 2.25 to 2.75 mm apart they are the regular view.
Slices are never skipped and never relabelled.

## Part A - technical checks (all passed)

| Check | Result |
|---|---|
| Synthetic volumes with known values: spacings 0.75, 1.25, 2.0, 2.5, 6 mm; six storage orientations | 182 of 182 |
| Real pages: four cases in Edge 158, one case in Chrome 157, integrated AMD Radeon graphics | 16 of 16 each |
| Mean and MIP slabs against a direct calculation over the original NIfTI read with SimpleITK, 9 positions per view per case including both scan edges | 0 HU difference in 90 slabs |
| Opening state: Axial 2.5 mm, Fit, lung window W 1500 / L -600, no outline, outline bar absent | pass |
| Orientation: a left, anterior marker voxel appears on screen right and top in both slab views | pass |
| Averaging is real: marker voxel value equals the hand calculation; a multi-slice slab differs from its centre slice | pass |
| Scan edges: shortened slab, label says so | pass |
| Switching between Axial 2.5 mm, MIP 10 mm and native slices: position unchanged; native equals the stored image | pass |
| Outlines: none over a slab; active outline released when a slab view is chosen; "Show in scan" goes to native slices | pass |
| Reader's zoom kept across views (for example 3.102 screen pixels per image pixel in both), restored exactly; Fit stays Fit | pass |
| Scrolling: 300 wheel notches per view, then 600 steps with 200 window changes and view switches | no error; 12.1 to 13.4 ms per notch including the test driver |
| Slowest slab calculation plus colouring | mean 8 to 9 ms, MIP 11 to 14 ms |
| Memory after the stress block against start-up | -183 to -103 MB in Edge, +58 MB in Chrome; cache inside its 24 MB limit |
| Grading and CSV export; rows name the v6 build; the AI reader's answers are not in the export | pass |
| Pack M12 against M11: the 20 tool answers per case identical; embedded scan and outline data identical; checklist and evidence files identical | pass |
| Earlier output unchanged: M11 files match their recorded hashes; an M11 page rebuilt with the changed builder is byte-identical | pass |
| Unit tests, whole project | 298 passed |

Not tested: the reviewer's own laptop. The v4 browser suite was not rerun on a v6 page (it expects the page to open on native slices);
the v6 checks compare native slices with the stored image instead.

## AI reader on these pages

NV-Reason-CT replaces any other AI reader on the pages (earlier packs showed none). A small line under each of the 18 reader conditions gives its answer,
marked "shown only, not counted"; its own report text is in a closed card. No tool answer changes. See `docs/STATUS.md` for how it was run and its limits.

## Part B - clinician usability feedback

None yet. The pack has not been shown. Open questions for the reviewer: is 2.25 mm (three 0.75 mm slices) acceptable as "2.5 mm"; is one native slice per
wheel step the right pace in the 2.5 mm view; is the AI reader line helpful or distracting.
