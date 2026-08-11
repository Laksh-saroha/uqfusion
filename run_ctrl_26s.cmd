@echo off
REM ---------------------------------------------------------------------------
REM CONTROL EXPERIMENT: yolo26s seed 0 under laptop settings.
REM
REM Why: laptop yolo26m seed0 scored mAP50-95 0.3452, but the best server row was
REM yolo12x at 0.2965 (seed-mean) and server yolo26s at 0.2864. A +0.06 step from
REM 26s to 26m is far outside the 0.005-0.011 seed sd, so the laptop arm may not be
REM comparable to the server arm at all - which would invalidate Table 1 as a
REM ranking. Three things changed at once between the arms:
REM     batch        24        -> 8      (effective 72 -> 64; BN micro-batch differs)
REM     ultralytics  8.4.90    -> 8.4.7
REM     torch        2.12.1+cu126 -> 2.7.1+cu118
REM
REM This run holds the model, split, seed and epoch budget fixed and changes only
REM that bundle, so it reads out the total laptop-vs-server delta for one cell:
REM     server yolo26s seed0 mAP50-95 = 0.28635   <- compare against this
REM     ~0.286  => arms comparable, yolo26m 0.3452 is a real result
REM     ~0.33+  => the seam is an artifact, every laptop row needs re-scoping
REM
REM Separate --out-csv and --run-prefix are REQUIRED: yolo26s seeds 0/1/2 are
REM already in benchmark_results_tail.csv from the server, so reusing that CSV
REM would make the grid skip this run as already done.
REM
REM Logs to the same grid_laptop.log so scripts\watch_grid.py keeps working
REM unmodified (its rows=N/27 counter tracks the main CSV and will sit still).
REM
REM Never put a REM line between the ^-continued lines below - the caret continues
REM onto the comment, the command ends there, and the next flag is parsed as its
REM own command ("'--workers' is not recognized"). Comments go here, above.
REM ---------------------------------------------------------------------------
setlocal enabledelayedexpansion
cd /d A:\Uncertain
set PYTHONPATH=A:\Uncertain\src
set LOG=runs\grid_laptop.log
set /a TRIES=0

:loop
set /a TRIES+=1
echo.>> %LOG%
echo [wrapper] CONTROL yolo26s seed0 attempt !TRIES! at %DATE% %TIME% >> %LOG%
python scripts\run_benchmark.py ^
    --data runs\derived\data_vis_stride2.yaml ^
    --classes 0 ^
    --seeds 0 ^
    --variants yolo26s ^
    --batch 8 ^
    --workers 8 ^
    --out-csv runs\benchmark\benchmark_ctrl_26s.csv ^
    --run-prefix ship_ctrl >> %LOG% 2>&1
if not errorlevel 1 goto done

if !TRIES! GEQ 20 (
    echo [wrapper] CONTROL: 20 consecutive failures - giving up, read %LOG% >> %LOG%
    echo [wrapper] CONTROL: 20 consecutive failures - read %LOG%
    exit /b 1
)
echo [wrapper] CONTROL exited with an error, retrying in 60s >> %LOG%
timeout /t 60 /nobreak >nul
goto loop

:done
echo [wrapper] CONTROL complete at %DATE% %TIME% >> %LOG%
echo [wrapper] CONTROL complete. Main grid is PAUSED - restart it with run_tail_laptop.cmd
