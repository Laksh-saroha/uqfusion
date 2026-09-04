@echo off
REM Work the IR benchmark queue: 31 variants x 3 seeds on IR @ train_stride=4
REM (93 runs total, runs\queue_ir_benchmark_stride4). Local RTX 4080 only.
REM Long-running by design -- do not expect this to finish quickly.
REM
REM Resume-safe -- if this window is closed, killed or crashes, just run this
REM file again: the queue picks up at whichever run/epoch it stopped on
REM (weights\last.pt), same mechanism as every other queue in this repo.
REM Pause/Resume from ir_benchmark_dashboard.cmd, or:
REM   python scripts\run_queue.py --queue-dir runs\queue_ir_benchmark_stride4 pause
REM   python scripts\run_queue.py --queue-dir runs\queue_ir_benchmark_stride4 resume
REM Uses the pinned GPU interpreter directly (config.yaml gpu_python) rather than
REM whatever 'python' resolves to on PATH -- this launches real training (D30).
cd /d A:\Uncertain
"C:/Users/lasa2/AppData/Local/Programs/Python/Python313/python.exe" scripts\run_queue.py --queue-dir runs\queue_ir_benchmark_stride4 run %*
echo.
echo Queue runner exited. Rerun this file to continue any unfinished run.
pause
