@echo off
REM Launch the queue dashboard and open it in a browser.
REM Safe to start, stop and restart at any time — it only reads runs\queue\
REM and, for Pause/Resume, flips one boolean. Closing it does not stop training.
cd /d "%~dp0"
python scripts\dashboard.py --open %*
REM Keep the window up if python is missing or the port is already taken —
REM otherwise the window closes instantly and the error is unreadable.
if errorlevel 1 (
  echo.
  echo Dashboard exited with an error. Common causes:
  echo   - the port is already in use  ^-^-^>  dashboard.cmd --port 8771
  echo   - 'python' is not on PATH in this window
  pause
)
