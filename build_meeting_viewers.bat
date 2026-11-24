@echo off
REM Build the 4 interactive viewer pages for the meeting pack (CPU, about 1-2 min per case, needs ~6 GB RAM).
REM Cases 301 and 541 were graded on v3_2, 988 on the v3_3 candidate; all three show the grades received. 832 (v3_3) is still to grade.
REM Output: runs\M1_meeting_pack_20261119\viewer_*.html  (open index.html in Chrome or Edge)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set M=runs/M1_meeting_pack_20261119
.venv\Scripts\python.exe scripts\build_viewer_page.py --run runs/A1_agent_valid_301_a_2_20261118_152058 --checklist-file checklist_checklist_v3_2.json --rule-based --out %M%/viewer_1_effusion_success_valid_301.html --grades-csv runs/grades_received/grades_valid_301_a_2_*_2026-11-20.csv
.venv\Scripts\python.exe scripts\build_viewer_page.py --run runs/A1_agent_valid_541_a_2_20261118_151640 --checklist-file checklist_checklist_v3_2.json --rule-based --out %M%/viewer_2_minimal_effusion_miss_valid_541.html --grades-csv runs/grades_received/grades_valid_541_a_2_*_2026-11-21.csv
.venv\Scripts\python.exe scripts\build_viewer_page.py --run runs/A1_agent_valid_832_a_2_20261118_094216 --checklist-file checklist_checklist_v3_3.json --rule-based --out %M%/viewer_3_calcium_extra_valid_832.html
.venv\Scripts\python.exe scripts\build_viewer_page.py --run runs/A1_agent_valid_988_a_2_20261118_153023 --checklist-file checklist_checklist_v3_3.json --rule-based --out %M%/viewer_4_nodule_cannot_tell_valid_988.html --grades-csv runs/grades_received/grades_valid_988_a_2_*_2026-11-22.csv
echo Finished.
pause
