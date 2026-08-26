@echo off
REM Windows twin of scripts/supervise_queue.sh: keep one queue working, restarting
REM its runner if it dies, without ever starting a second one against the same GPU.
REM
REM   scripts\supervise_queue_win.cmd runs\queue_ir_benchmark_stride4
REM
REM Launch it detached so it outlives any console:
REM   powershell -c "Start-Process -FilePath A:\Uncertain\scripts\supervise_queue_win.cmd -ArgumentList 'runs\queue_ir_benchmark_stride4' -WindowStyle Hidden"
REM
REM To stop it: close the window, or `taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq
REM uqfusion-supervisor*"`. Stopping the supervisor does NOT stop a run in
REM progress; pause that with `run_queue.py ... pause` first if that is the intent.
REM
REM It never kills anything, and it starts a runner only when
REM queue_runner_alive.py says none is working the queue. Uses the pinned GPU
REM interpreter (config.yaml gpu_python), not PATH python, which is CPU torch.
title uqfusion-supervisor %1
setlocal
set "PY=C:/Users/lasa2/AppData/Local/Programs/Python/Python313/python.exe"
set "Q=%~1"
if "%Q%"=="" (echo usage: supervise_queue_win.cmd ^<queue-dir^> & exit /b 2)
cd /d A:\Uncertain

echo [%date% %time%] supervisor start for %Q% >> "%Q%\supervisor.log"

:loop
"%PY%" scripts\queue_remaining.py --queue-dir "%Q%" --quiet
if not errorlevel 1 goto done

"%PY%" scripts\queue_runner_alive.py --queue-dir "%Q%" --quiet
if not errorlevel 1 goto wait

echo [%date% %time%] no runner working %Q% - starting one >> "%Q%\supervisor.log"
"%PY%" scripts\run_queue.py --queue-dir "%Q%" run >> "%Q%\queue_stdout.log" 2>&1
echo [%date% %time%] runner exited (code %errorlevel%) >> "%Q%\supervisor.log"

:wait
REM 60 s between checks: long enough that a restarting runner is never raced,
REM short enough that a crash costs a minute rather than a night.
timeout /t 60 /nobreak >nul
goto loop

:done
echo [%date% %time%] queue complete - supervisor exiting >> "%Q%\supervisor.log"
endlocal
