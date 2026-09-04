@echo off
REM Dashboard for STAGE 2 (runs\queue_screen2 + runs\screen2).
REM screen_dashboard.cmd watches stage 1; dashboard.cmd watches the Phase 2 queue.
REM Uses port 8772 so all three can run side by side.
cd /d "%~dp0"
python scripts\dashboard.py --queue-dir runs\queue_screen2 --port 8772 --open %*
if errorlevel 1 (
  echo.
  echo Dashboard exited with an error ^(port in use -^> add --port 8773^).
  pause
)
