@echo off
REM Dashboard for the STAGE-1 SMALL-OBJECT SCREEN (runs\queue_screen + runs\screen1).
REM The plain dashboard.cmd points at runs\queue\ (the Phase 2 queue) — this one points
REM at the screen's own queue dir, so the two never touch each other's state.
REM
REM Safe to start, stop and restart at any time: it only reads the queue json files
REM and, for Pause/Resume, flips one boolean. Closing this window does NOT stop training.
cd /d "%~dp0"
python scripts\dashboard.py --queue-dir runs\queue_screen --open %*
if errorlevel 1 (
  echo.
  echo Dashboard exited with an error. Common causes:
  echo   - the port is already in use  ^-^-^>  screen_dashboard.cmd --port 8771
  echo   - 'python' is not on PATH in this window
  pause
)
