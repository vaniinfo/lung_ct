# Viewer v5: axial sliding-slab MIP - test report

Research use only. Three builds exist; the numbers below are for the newest unless a row says otherwise.

| Build | Pack | Date | Layout |
|---|---|---|---|
| `checklist_viewer_v5_canvas2d_mip` | M9 | 2026-11-24 | first layout (kept, not to be sent) |
| `checklist_viewer_v5_1_canvas2d_mip` | M10 | 2026-11-25 | one-line banner with help, fixed-height bars (kept, not to be sent) |
| `checklist_viewer_v5_2_canvas2d_mip` | M11 | 2026-11-25 | short label, outline bar removed in MIP: **the one to show** |

This report has two parts that must not be mixed:

- **Part A - technical checks.** Done on the development workstation. They show that the page computes and draws
  what it says it does.
- **Part B - clinician usability feedback.** Not collected yet. The clinician reviewer has not opened these pages.
  Nothing in Part A says the mode is useful for reading, or that it helps find nodules.

## What was added

The clinician reviewer said that MIP (maximum intensity projection) helps him see nodules. Viewer v5 is viewer v4
with one optional display mode added:

| Control | What it does |
|---|---|
| `Single slice` / `MIP` | Switches between one axial image and a slab projection. The page opens on `Single slice`. |
| `5 mm` / `8 mm` / `10 mm` | Slab thickness asked for. 10 mm is selected at the start. Choosing a thickness does not switch MIP on. |
| Wheel, arrow keys | Move the slab one image at a time, so neighbouring slabs overlap. |
| Green `Single image N` | Leaves MIP and shows the image at the centre of the slab. `Single slice` does the same. |
| One-line label | `MIP ≈10 mm · 13 slices · Images 183–195 · Centre 189`: approximate thickness, slice count, first and last image, centre image. |
| `? MIP help` | Opens the longer explanation on request, with the spacing, exact coverage and centre distance of the current slab. Closed at first. |
| `W ... L ...` next to the window presets | Window width and level (centre) in HU, for presets and free changes. |

Unchanged: one standalone HTML file opened from `file://`; single axial image and no outline at start-up; scan
coverage and voxel resolution; grading form, autosave and CSV export; all earlier pages and packs. Model inputs,
predictions, measurements and checklist answers are not read or changed by MIP. These are display changes only.

### Refinements of 2026-11-25 (pack M10)

| Request | What was done |
|---|---|
| Collapse the long explanation | The three-line note is gone. MIP shows one line; the full text is behind `? MIP help`, as a panel that floats over the scan and does not move it. |
| Hide inactive outline controls in MIP | The outline buttons are not shown while MIP is on. Their bar keeps its height and says "Outlines are not shown in MIP view". |
| Fit at first opening; keep the reader's zoom when toggling | The page opens at Fit (zoom 100 % of fit, no pan; no image zoom is stored between visits). MIP on / off does not touch zoom or pan, and the bars above and below the scan keep the same height in both modes, so the scan area keeps its size. Zoom, pan, screen scale and image position are identical before and after. |
| Show window width and level | `W 1500 · L -600` (lung preset) next to the presets; follows presets and right-drag changes. |

### Second round of 2026-11-25 (pack M11)

| Request | What was done |
|---|---|
| Simplify the slab label | `MIP ≈10 mm · 13 slices · Images 183–195 · Centre 189`. The figure after ≈ is the thickness asked for. If the slab is shortened at the scan edge, or is a single slice, the figure is slices x spacing and the label adds "shortened at scan edge" or "same as single image". |
| Spacing and coverage convention in the help | The help panel has a line for the current slab, for example "This slab: 13 slices × 0.75 mm spacing = 9.75 mm coverage (asked for 10 mm). Centres of the first and last slice: 9 mm apart.", and explains ≈ and that spacing is not slice thickness. |
| Remove the empty space under the outline notice | The outline bar is removed in MIP. "Outlines are not shown in MIP view" is a small notice on the label line. The scan area is 87 to 125 px taller in MIP on the test window. |

Because the scan area now changes height with the mode, the reader's zoom is kept in a different way than in pack M10:
zoom is stored relative to Fit, so on a mode change it is rescaled to keep the same number of screen pixels per image
pixel and the same image point at the centre of the scan area. Leaving MIP restores the earlier value exactly. From
Fit, the image stays at Fit and uses the taller area. The read-out under the scan still reserves three lines.

## How the slab is defined

| Item | Rule |
|---|---|
| Values used | The stored CT values (HU), **before** brightness / contrast. The window only re-colours the maxima. |
| Slice spacing `dz` | Distance between neighbouring image centres, from the scan's affine, in mm. Never a fixed number of slices. |
| Images in the slab | `n = max(1, floor(requested mm / dz + 0.5))`: the nearest whole number, halves rounded up, at least 1. |
| Placement | Centred on the image the reader is on. If `n` is even, the extra image is on the head side. |
| Top and bottom of the scan | Images outside the scan are left out. The slab is shortened, never shifted or padded, and the page says "shortened at the edge of the scan". |
| Spacing larger than the request | `n = 1`. The slab equals the single image and the page says so. |

### Thickness convention (checked 2026-11-25)

The stated thickness uses a **coverage convention: thickness = n x spacing.** Each image is taken to stand for one
spacing of tissue. The other common figure is the distance from the centre of the first image to the centre of the
last, `(n - 1) x spacing`. Both describe the same images:

| Slice spacing | Asked | Images `n` | Stated thickness `n x dz` | First-to-last image centres `(n - 1) x dz` | Cases |
|---|---|---|---|---|---|
| 0.75 mm | 5 mm | 7 | 5.25 mm | 4.5 mm | valid_301, valid_832, valid_988 |
| 0.75 mm | 8 mm | 11 | 8.25 mm | 7.5 mm | |
| 0.75 mm | 10 mm | 13 | 9.75 mm | 9.0 mm | |
| 1.25 mm | 5 mm | 4 | 5.0 mm | 3.75 mm | valid_541 |
| 1.25 mm | 8 mm | 6 | 7.5 mm | 6.25 mm | |
| 1.25 mm | 10 mm | 8 | 10.0 mm | 8.75 mm | |

- The statement in the request is confirmed: 13 images at 0.75 mm are 9.75 mm under the coverage convention and
  9 mm between the outer image centres.
- The label shows the thickness asked for, marked as approximate (`MIP ≈10 mm · 13 slices`). Both exact figures for the
  current slab are in the help panel: "13 slices × 0.75 mm spacing = 9.75 mm coverage ... centres ... 9 mm apart".
- **The projection was not changed.** The same 13 images are used as before; only the wording was extended.
  Choosing `n` so that `(n - 1) x dz` is nearest to the request would give 14 images at 0.75 mm, a different picture.
  That is a design choice for the reviewer (Part B, question 3), not a labelling fix.

**Slice spacing is not slice thickness.**

- *Slice spacing* is how far apart the image centres are. It is what the NIfTI affine stores and what the page uses.
- *Acquisition / reconstruction slice thickness* is how much tissue one image averages. It is not in the NIfTI
  affine, and the CT-RATE metadata we hold lists z spacing only (`ZSpacing`; no slice-thickness column). So the page
  does not know it.
- If images are thicker than their spacing (overlapping reconstruction), the tissue really covered is
  `(n - 1) x dz + thickness`, more than the stated figure; if thinner, less.
- The page therefore says "spacing" and never "slice thickness", and the help text says the scanner's slice
  thickness is not known to the page.

## Limits of this version

1. **Axial only.** Choosing Coronal, Sagittal or 3 planes leaves MIP; its button is disabled there.
2. **No outlines in MIP.** An outline is drawn for one image. Over a projection of several images it would mark
   tissue that is not at that depth. An active outline is switched off when MIP starts and is not switched back on
   automatically. "Show in scan" first returns to the single image, then shows the outline there.
3. **MIP hides things too.** A bright structure anywhere in the slab covers darker tissue in front of or behind it,
   and ground-glass or low-density findings can disappear. It is an addition to the single image, not a replacement.
4. **Thickness is a count of images times spacing** (see above).
5. **Display range.** As in every viewer version, values are clipped to -1024..3071 HU before embedding.
6. On a narrow scan pane the one-line label is cut with "..."; the full line is in its tooltip.
7. The label's millimetre figure is the thickness asked for, not the exact coverage; the exact figures are one click away in the help.

## Part A - technical checks

Run by `scripts/test_mip_viewer.py` and `tests/test_viewer_v5.py` on the unmodified HTML opened from `file://`.
Edge 158 and Chrome 157 on this workstation.

**Graphics adapter.** The real-page runs force the low-power adapter. The browser reported its renderer as
`AMD Radeon(TM) Graphics` (the integrated adapter of this workstation), with the RTX 3090 present but not used.
This is integrated graphics, but it is not the reviewer's laptop.

| Requirement | How it was checked | Result |
|---|---|---|
| Known synthetic voxel maxima | Phantom with isolated bright voxels of known value. 33 slabs per page (5 / 8 / 10 mm x 11 centre images) compared with numpy; a marker voxel must appear in exactly the slabs that contain its image | equal on all 12 synthetic pages |
| Slice spacing | Phantoms with 0.75, 1.25, 2.5 and 6 mm spacing; image count, thickness and centre distance compared with the rule written out independently in the test | pass; 6 mm spacing gives a one-image slab that equals the single image |
| Boundary handling | Centre images at both ends of the scan and next to them; page text | shortened as defined; the page states it |
| Orientation | The phantom stored in 6 axis orders / flips; a patient-left, anterior marker must show on the screen right and top in MIP | pass in all 6 |
| Maxima before windowing | The drawn pixels equal the window applied to the HU maxima for 5 windows; stored maxima unchanged by the window | pass |
| Back to the original slice | Green button and `Single slice`: same image index and the same pixels as before MIP | pass |
| **Direct maximum on real scans** | 21 slabs per case (5 / 8 / 10 mm x 7 centre images: first image, third image, quarter, middle, three-quarter, third-last, last). Each compared with the maximum over the same images of the original NIfTI read with SimpleITK, independent of the page builder | largest difference 0 HU in all 84 slabs (Edge) and 21 (Chrome) |
| Scan boundaries on real scans | 10 mm slab at the first image: 7 of 13 images (0.75 mm) or 5 of 8 (1.25 mm); at the third image: 9 of 13 or 7 of 8 | as defined; equal to the direct maximum |
| Fit at first opening | Zoom 100 % of fit, no pan, whole image inside the scan area, nothing stored between visits | pass, all pages |
| Reader's zoom kept | Zoom and pan set by the reader; MIP on, slab changed, MIP left by each button. In MIP: screen pixels per image pixel and the image point at the centre of the scan area. Back on the single image: zoom, pan, screen scale and image rectangle compared exactly. From Fit: stays at Fit in both modes | same scale and centre in MIP (for example 3.102 and 3.102 screen pixels per image pixel); identical after return; all pages |
| Label and help | Label text compared character by character with the rule; one line high; help closed at first, opens and closes, does not resize the scan, and states spacing, coverage and centre distance of the current slab | pass |
| Outline controls in MIP | Outline bar removed, one-line notice on the label line, no empty strip (scan area taller in MIP); no coloured pixel over a slab; none can be switched on; bar and outlines work again in the single image | pass |
| Window width and level | Read-out for the three presets and free changes, in both modes | pass (`W 1500 · L -600`, `W 400 · L 40`, `W 1800 · L 400`) |
| No graphics texture, no WebGL | Page source searched; a slab is a typed array of one axial image | none in the page |
| Grading and CSV export | Grades entered and exported with MIP switched on; rows name the build. Pack check (40 checks) on M11 | pass; 40 of 40 |
| Older pages | A v4 page rebuilt before and after the first change | byte-identical (same SHA-256) |
| Everything viewer v4 did | The existing v4 page test run on a v5.2 page (valid_832, the largest) | 36 of 36 |

### Scrolling and memory on the integrated adapter (pack M11)

Per case: 300 real mouse-wheel notches in MIP drawn on screen, then 600 scroll steps with 200 window changes and
repeated thickness and on / off switches. Memory is private bytes of the whole browser, in MB.

| Page | Browser | Slowest slab | Per wheel notch (incl. test driver) | At start-up | In MIP after the block | After leaving MIP | Graphics process, start / in MIP |
|---|---|---|---|---|---|---|---|
| valid_301 | Edge | 12 ms | 12.8 ms | 846 | 778 | 771 | 157 / 167 |
| valid_541 | Edge | 12 ms | 12.6 ms | 913 | 746 | 754 | 159 / 162 |
| valid_832 | Edge | 16 ms | 15.2 ms | 1075 | 908 | 919 | 159 / 163 |
| valid_988 | Edge | 16 ms | 16.2 ms | 991 | 777 | 727 | 160 / 161 |
| valid_301 | Chrome | 13 ms | 14.5 ms | 671 | 644 | 621 | 173 / 180 |

- No error in any run. The slab cache never exceeded its 24 MB limit and was empty after leaving MIP.
- Memory did not grow: it was lower after the block than at start-up in every run. The graphics process changed by
  about 10 MB.
- Start-up figures include memory the browser has not yet released after decoding the scan, which is why later
  figures are lower. These are whole-browser figures and move by tens of MB between runs.

Totals: synthetic 194 of 194; real pages 17 of 17 for each of the four cases in Edge and for valid_301 in Chrome;
unit tests 22 (247 in the whole project).

What these checks do not cover:

- The reviewer's own laptop, other browsers than Edge and Chrome, and touch screens.
- A brightness window is a rising curve, so taking the maximum before or after windowing gives the same picture.
  The check shows that the stored maxima are HU and that the drawn image is the window applied to them; it cannot
  show a visible difference between the two orders.
- Per-notch time includes the test's own messaging to the browser, so it is an upper bound, not a frame time.

Run folders: `runs/V7_mip_test_*` (results, screenshots; `*_M9_*`, `*_M10_*`, `*_M11_*` by pack),
`runs/V8_M9_viewer_check_20261124`, `runs/V9_M10_viewer_check_20261125`, `runs/V10_M11_viewer_check_20261125`,
`runs/V2_canvas_test_v5*regression_*`.

## Part B - clinician usability feedback

**None yet.** Pack `runs/M11_policy_v3_10_mip_20261125/` (open `index.html`) is ready for the clinician reviewer
to try. It is the same content as pack M8 (policy checklist_v3_10, earlier grades filled in, report-label column)
with the newest MIP viewer. It has not been sent. Packs M9 and M10 (earlier layouts) are kept and should not be sent.

Questions to ask, with answers recorded here as the reviewer's opinion, not as a test result:

1. Is the MIP view useful for looking for nodules on these four scans? Which slab thickness does he use?
2. Is 10 mm the right default, and are 5 / 8 / 10 mm the right choices?
3. Thickness wording: is "≈10 mm · 13 slices" enough, with the exact figures in the help? Would he rather have the
   slab chosen so that the outer slice centres are closest to the requested thickness (14 slices, a different picture)?
4. Does he want an outline option in MIP, and if so which outline? (Not built: see limit 2.)
5. Is axial-only enough, or is a coronal MIP needed?
6. Is the one-line label enough, and did he need the help panel?
7. Is the window width / level read-out what he expects to see?
8. Did the page stay responsive on his laptop while scrolling in MIP?

The four cases are development cases he has already graded on open pages. Anything he reports is usability
feedback. It is not evidence that MIP improves nodule detection.
