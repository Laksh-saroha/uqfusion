@echo off
REM Dashboard for the IR BENCHMARK queue (runs\queue_ir_benchmark_stride4 +
REM runs\ir_benchmark_stride4). 31-variant x 3-seed IR grid @ train_stride=4.
REM The plain dashboard.cmd points at runs\queue\ (the Phase 2 queue) -- this one
REM points at this queue's own dir, so the two never touch each other's state.
REM
REM Safe to start, stop and restart at any time: it only reads the queue json
REM files and, for Pause/Resume, flips one boolean. Closing this window does
REM NOT stop training.
cd /d "%~dp0"
python scripts\dashboard.py --queue-dir runs\queue_ir_benchmark_stride4 --open %*
if errorlevel 1 (
  echo.
  echo Dashboard exited with an error. Common causes:
  echo   - the port is already in use  ^-^-^>  ir_benchmark_dashboard.cmd --port 8771
  echo   - 'python' is not on PATH in this window
  pause
)
