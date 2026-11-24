@echo off
REM Location-first viewer pages with the lazy lossless loader (viewer v3 = v2 layout; the v2 files of 2026-11-22 are kept;
REM use --viewer v2 to rebuild those). Location-first viewer pages (viewer v3, grading form grades_v2, checklist_v3_4) for the four meeting cases.
REM These are DEVELOPMENT / USABILITY cases: nothing graded on them is evaluation data.
REM Needs the per-lobe rows (scripts\add_lobe_distribution.py) and checklist_checklist_v3_4.json
REM (scripts\rescore_checklist.py --config configs\checklist_v3_4.json) in each run. CPU, about 1-2 min per page.
REM The earlier pack runs\M1_meeting_pack_20261119 (viewer v1) is not touched: build_meeting_viewers.bat rebuilds it.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set M=runs/M2_location_first_20261122
set G=runs/grades_received
set B=.venv\Scripts\python.exe scripts\build_viewer_page.py --checklist-file checklist_checklist_v3_4.json --rule-based --viewer v3
REM Three cases already graded with form v1: grades shown, location marked "not recorded" (never as assessed).
%B% --run runs/A1_agent_valid_301_a_2_20261118_152058 --grades-csv %G%/grades_valid_301_a_2_*_2026-11-20.csv --out %M%/viewer_valid_301_v3.html
%B% --run runs/A1_agent_valid_541_a_2_20261118_151640 --grades-csv %G%/grades_valid_541_a_2_*_2026-11-21.csv --out %M%/viewer_valid_541_v3.html
%B% --run runs/A1_agent_valid_988_a_2_20261118_153023 --grades-csv %G%/grades_valid_988_a_2_*_2026-11-22.csv --out %M%/viewer_valid_988_v3.html
%B% --run runs/A1_agent_valid_832_a_2_20261118_094216 --out %M%/viewer_valid_832_v3.html
REM Staged example (step 1 without the tool's output): to try the procedure only.
%B% --run runs/A1_agent_valid_832_a_2_20261118_094216 --blinded --out %M%/staged_valid_832_v3.html
REM Finding-mask audit review pages with the stage-1 finding / location fields (blinded_viewer_v2).
set A=runs/A2_audit_inventory_usability_20261120_190509
for %%C in (valid_301_a_2 valid_541_a_2 valid_832_a_2 valid_988_a_2) do .venv\Scripts\python.exe scripts\build_blinded_viewer.py --inventory %A%/inventory.json --case %%C --out %A%/review_v2_%%C.html
echo Finished.
pause
