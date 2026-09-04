@echo off
REM Work the Phase 2 training queue. Long-running: leave this window open.
REM Resume-safe — if it is closed, killed or crashes, run it again and each run
REM continues from its own weights\last.pt. Pause/Resume from the dashboard.
cd /d "%~dp0"
python scripts\run_queue.py run %*
echo.
echo Queue runner exited. Rerun this file to continue any unfinished run.
pause
