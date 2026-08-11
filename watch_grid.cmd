@echo off
REM Health watchdog for the laptop Phase 1 grid. Independent of any Claude session.
REM Appends one line every 10 min to runs\health.log; see scripts\watch_grid.py.
REM
REM Start this alongside run_tail_laptop.cmd and leave it running. To review:
REM     type runs\health.log                     (everything)
REM     findstr /V " OK " runs\health.log        (only problems)
cd /d A:\Uncertain
python scripts\watch_grid.py
