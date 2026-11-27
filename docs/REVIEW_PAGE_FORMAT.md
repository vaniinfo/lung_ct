# ThoraxCheck review page format (standard for every page sent to a reader)

Status: **fixed 2026-11-27** by the project lead. Every review page is built with `scripts/build_viewer_page.py`
(through `scripts/build_policy_pack.py` for packs) and must look and behave as described here.
Reference page: pack M14 (`runs/M14_policy_v3_11_two_segmentations_20261126`), viewer build
`checklist_viewer_v7_canvas2d_axial25_mip10_two_segmentations`, grading form `grades_v2`.

**Rule:** a change to anything in this file needs the project lead's approval first, then a new version line in
"Change log" below, then the change in code. A new viewer build may add features; it may not move, rename or remove
what is listed here without that approval. One-off page builders that bypass `build_viewer_page.py` are not allowed for
pages that go to readers.

## 1. Layout

- One self-contained HTML file per scan, opened from a folder with `file://` (no internet, nothing sent anywhere).
- **Scan on the left, panel on the right.** The bar between them can be dragged (25 % to 75 % of the width);
  double-click returns to half and half; the choice is remembered in that browser.
- The panel scrolls on its own; the scan stays in view.

## 2. Scan viewer (left)

- Top bar, in this order: **Window** Lung / Soft tissue / Bone with the current W / L shown; **View** Axial / Coronal /
  Sagittal / 3 planes; **Zoom** - / + / Fit / 1:1 pixels / Smooth; **Cross-hair**; **Text size** A- / % / A+.
- Window is also adjustable by right-drag or Shift+drag (right = wider, down = darker).
- Second bar: **View** Axial 2.5 mm (opening view) / MIP 10 mm / Details (native slices).
- Line under the bars: slab description (mean of n slices, image range, centre); "Outlines: native slices only (Details)".
- Opens on Axial 2.5 mm, lung window, no outline switched on.
- Status line at the bottom: image number, value at the cross-hair, window, zoom, position in mm.
- Mouse: wheel or arrow keys = next / previous image; click = move cross-hair; drag = move image; Ctrl+wheel = zoom.
- Orientation letters on the image (R / L / A / P ...); patient's right is on the image left.

## 3. Panel (right), top to bottom

1. **Title**: `<scan> — ThoraxCheck condition checklist` (pages built before 2026-11-27 read `<scan> — condition checklist`); grey line: research use only, scan type, checklist version, report source,
   "patient's right is on the image left", viewer build, form version.
2. **Answer policy box** (blue): which checklist version the page shows and what its rules mean, in plain words.
3. **How to read this page** (yellow): where-first, regions are HU-selected, Numbers buttons, Measurement panel,
   blue "Show in scan" buttons; button "Open measurement panel".
4. **Your grades** box: Reviewer name field, **Save grades (CSV)** (green), **Clear**; "n of N graded (autosaved in this browser)";
   overall comment box; a note on which checklist version the grades are recorded against.
5. **Checklist table**, button "Show what it means / why" above it. Columns:
   | condition | agent answer and location | CT-RATE report label | Your grade and location |
   - *condition*: name; blue **Show in scan** button where an outline exists.
   - *agent answer and location*: status chip (e.g. "none measured", "cannot tell", "suggested by segmentation",
     "present"), **Numbers** button; below a dotted line: "AI reader NV-Reason-CT: present / absent (shown only, not counted)"
     or, where the checklist lets the reader decide, that basis.
   - *CT-RATE report label*: "mentioned" / "not mentioned" / "no label for this item", with a chip comparing it with the tool
     ("same as tool", "tool undecided", ...). Header note: "not mentioned" is not "absent".
   - *Your grade and location*: segmented buttons **Present / Absent / Can't tell**; **severity** and **confidence** drop-downs;
     **Where:** lobe chips RUL RML RLL LUL LLL (Right / Left for side-only findings), Indeterminate, Not assessed;
     "Location does not apply to an absent finding" note; **comment** field; agreement chip ("agrees with agent", "open").
6. Collapsible sections, in this order:
   - **Radiologist report (CT-RATE), not shown to the agent** (with the licence line: keep within the research group);
   - **Earlier review of this case** (only when earlier grades exist);
   - **Measurement panel** (numbers behind the answers, E-numbers);
   - **Report** (the tool's rule-based report: confirmed / not detected by measurement / suggested by segmentation / below the
     reporting limit), always open;
   - **AI reader report (NV-Reason-CT)**, shown only, not counted.
7. Viewer v7 adds the **Segmentation models** bar (TotalSegmentator / NV-Segment-CT label maps), display only.

## 4. Staged (blinded) variant

Used when a reader must answer before seeing any machine output (`build_viewer_page.py --blinded`).

- Same layout, viewer, panel order and grading controls as above.
- Step 1: the scan and the **Your grade and location** column only. The reader locks step 1.
- Step 2 (after locking): the agent answer column, AI reader lines, Measurement panel, Report and AI reader report appear in
  their usual places. Step-1 grades are kept as locked; any change is stored next to them.
- The CT-RATE report and its labels are not in staged pages.
- The hidden content is inside the file: this is a reading procedure, not a security barrier.

## 5. Grades CSV

One row per checklist item; columns of form `grades_v2` (case, reviewer, item, grade, severity, confidence, location,
comment, checklist version, viewer build, timestamp). The file name carries case, reviewer and time.

## 6. Reader names

Reader names are typed by the reader or passed on the command line. They are never written into code, configs or docs
in the repository (public); pages live in `runs/` (not in the repository).

## Change log

- 2026-11-27 v1: written from pack M14 after a one-off page (RS01 reading pages v1, `scripts/build_reading_pages.py`)
  departed from this format; those pages are withdrawn and not to be sent.
- 2026-11-27 v1.1 (approved with checklist_v3_12): the policy box and the pack index describe the v3_12 rule (two sources for 'present';
  no 'suggested by segmentation'); the AI reader line reads "used for this answer" where the two-source rule used it. Layout, viewer,
  columns and grading form unchanged.
- 2026-11-27 v1.2 (project lead): public name ThoraxCheck in the page title and the pack index title. Nothing else changed. Packs built earlier keep their old titles.
