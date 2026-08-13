@echo off
REM ---------------------------------------------------------------------------
REM Phase 1 tail on the laptop: yolo26l/26x x seeds 0,1,2 (6 runs).
REM
REM 2026-08-12: yolo26m dropped from --variants. Seeds 0,1 are already in the CSV;
REM seed 2 was abandoned after five pause/resume cycles reset ultralytics'
REM EarlyStopping counter (the stopper is rebuilt fresh on resume and is NOT
REM restored from the checkpoint, so patience restarts and the run never stops).
REM Its fragmented run dir is in runs\benchmark\_discarded\. yolo26m stays n=2.
REM
REM Crash/shutdown safe in two layers:
REM   1. grid.py resumes an interrupted run from its own weights/last.pt
REM      (epoch-level), so a power cut costs minutes, not a whole run.
REM   2. this loop restarts the grid if the process dies for any other reason.
REM After a reboot, just double-click this file again - it picks up where it
REM stopped. Nothing is recomputed: finished runs are skipped via the CSV.
REM
REM batch 8 fits all three variants inside 12 GB (peaks 4.3 / 5.3 / 7.9 GB).
REM Do NOT change --data, --classes or --out-csv: the grid refuses to mix splits.
REM
REM workers 8 is measured, not guessed: 8 beats 16 (4.00 vs 2.80 it/s median). 16
REM workers cost ~10 GB RAM (~1.27 GB per process under Windows spawn) and leave
REM 1.6 GB free; CPU sits at ~14% either way, so the loader was never the limit.
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
echo [wrapper] attempt !TRIES! at %DATE% %TIME% >> %LOG%
python scripts\run_benchmark.py ^
    --data runs\derived\data_vis_stride2.yaml ^
    --classes 0 ^
    --seeds 0 1 2 ^
    --variants yolo26l yolo26x ^
    --batch 8 ^
    --workers 8 ^
    --out-csv runs\benchmark\benchmark_results_tail.csv ^
    --run-prefix ship >> %LOG% 2>&1
if not errorlevel 1 goto done

if !TRIES! GEQ 20 (
    echo [wrapper] 20 consecutive failures - giving up, read %LOG% >> %LOG%
    echo [wrapper] 20 consecutive failures - read %LOG%
    exit /b 1
)
echo [wrapper] exited with an error, retrying in 60s >> %LOG%
timeout /t 60 /nobreak >nul
goto loop

:done
echo [wrapper] grid complete at %DATE% %TIME% >> %LOG%
echo [wrapper] grid complete.
