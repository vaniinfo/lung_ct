@echo off
REM 2D canvas reviewer pages (viewer v4: no WebGL / NiiVue; grading form grades_v2; checklist_v3_4) for the four meeting cases.
REM For laptops with integrated graphics. DEVELOPMENT / USABILITY cases: nothing graded on them is evaluation data.
REM Earlier packs are not touched: runs\M1_meeting_pack_20261119 (viewer v1), runs\M2_location_first_20261122 (v2 and v3).
REM v3 is NOT a confirmed fix for integrated graphics (docs\STATUS.md).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set M=runs/M3_canvas_viewer_20261122
set G=runs/grades_received
set B=.venv\Scripts\python.exe scripts\build_viewer_page.py --checklist-file checklist_checklist_v3_4.json --rule-based --viewer v4
%B% --run runs/A1_agent_valid_301_a_2_20261118_152058 --grades-csv %G%/grades_valid_301_a_2_*_2026-11-20.csv --out %M%/viewer_valid_301_v4.html
%B% --run runs/A1_agent_valid_541_a_2_20261118_151640 --grades-csv %G%/grades_valid_541_a_2_*_2026-11-21.csv --out %M%/viewer_valid_541_v4.html
%B% --run runs/A1_agent_valid_988_a_2_20261118_153023 --grades-csv %G%/grades_valid_988_a_2_*_2026-11-22.csv --out %M%/viewer_valid_988_v4.html
%B% --run runs/A1_agent_valid_832_a_2_20261118_094216 --out %M%/viewer_valid_832_v4.html
REM Two-step (staged) example, not part of the pack; remove REM to build it:
REM %B% --run runs/A1_agent_valid_832_a_2_20261118_094216 --blinded --out %M%/staged_valid_832_v4.html
echo Finished. Test with: .venv\Scripts\python.exe scripts\test_canvas_viewer.py --page %M%/viewer_valid_832_v4.html --browser edge --gpu-mode igpu
pause
