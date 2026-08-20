@echo off
REM Dashboard for STAGE 3 (runs\queue_screen3 + runs\screen3).
REM screen_dashboard.cmd watches stage 1, screen2_dashboard.cmd stage 2,
REM dashboard.cmd the Phase 2 queue. Port 8774 so all four coexist.
cd /d "%~dp0"
python scripts\dashboard.py --queue-dir runs\queue_screen3 --port 8774 --open %*
if errorlevel 1 (
  echo.
  echo Dashboard exited with an error ^(port in use -^> add --port 8775^).
  pause
)
