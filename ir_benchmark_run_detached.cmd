@echo off
REM Detached variant of ir_benchmark_run.cmd -- launched via Task Scheduler so it
REM keeps running independent of any particular terminal/session. No `pause` (no
REM console to read a keypress from when launched this way) and appends to the
REM log instead of inheriting a console's stdout, so a relaunch after a crash
REM doesn't lose the previous tail.
cd /d A:\Uncertain
"C:/Users/lasa2/AppData/Local/Programs/Python/Python313/python.exe" scripts\run_queue.py --queue-dir runs\queue_ir_benchmark_stride4 run >> runs\queue_ir_benchmark_stride4\queue_stdout.log 2>&1
