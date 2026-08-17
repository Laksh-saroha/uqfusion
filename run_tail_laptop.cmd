@echo off
REM ---------------------------------------------------------------------------
REM Phase 1 tail on the laptop: yolo26x x seeds 0,1,2 (3 runs).
REM
REM 2026-08-12: yolo26m dropped from --variants. Seeds 0,1 are already in the CSV;
REM seed 2 was abandoned after five pause/resume cycles reset ultralytics'
REM EarlyStopping counter (the stopper is rebuilt fresh on resume and is NOT
REM restored from the checkpoint, so patience restarts and the run never stops).
REM Its fragmented run dir is in runs\benchmark\_discarded\. yolo26m stays n=2.
REM
REM 2026-08-14: yolo26l dropped too, same defect, one resume instead of five.
REM Seeds 0,1 are in the CSV (both stopped at ep 28, best at ep 8). Seed 2 peaked
REM at ep 6 but was resumed at ep 16; the fresh stopper re-anchored on ep 19 and
REM pushed the stop from ep 26 out to ep 39, so the run would have trained ~40%
REM longer than its siblings for a best.pt that was already fixed at ep 6. Stopped
REM at ep 26 and discarded rather than banked. yolo26l stays n=2.
REM
REM Both variants are n=2 by the same root cause: the ONLY safe time to pause a
REM run is before its fitness peak, where the rebuilt stopper re-converges on the
REM same window. After the peak, every resume extends the run and breaks
REM comparability with the other seeds. Prefer letting a run finish.
REM
REM 2026-08-17 CORRECTION: both seed-2 runs were RECOVERED and both variants are
REM n=3. The notes above were written using the wrong fitness definition (8.4.90
REM ranks epochs on mAP50-95 alone, not 0.1*mAP50 + 0.9*mAP50-95), which put
REM yolo26l seed 2's peak at the wrong epoch. Re-audited: it peaked at ep 6 and
REM ran to ep 26, i.e. exactly peak+patience, so best.pt holds the weights an
REM uninterrupted run would have kept. Banked via scripts\recover_row.py.
REM The pause-after-peak advice still stands; the n=2 conclusions do not.
REM See docs\phase1-experimental-record.md sections 6 and 15.
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
    --variants yolo26x ^
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
