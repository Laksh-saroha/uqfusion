@echo off
REM Detached variant of ir_benchmark_dashboard.cmd -- launched via Task Scheduler.
REM No --open (no desktop session to open a browser tab in when launched this
REM way, and webbrowser.open() isn't wrapped in a try/except in dashboard.py --
REM letting it raise here would take the server down before it ever binds the
REM port). Open http://127.0.0.1:8770 yourself once it's up.
cd /d A:\Uncertain
"C:/Users/lasa2/AppData/Local/Programs/Python/Python313/python.exe" scripts\dashboard.py --queue-dir runs\queue_ir_benchmark_stride4 >> runs\queue_ir_benchmark_stride4\dashboard_stdout.log 2>&1
