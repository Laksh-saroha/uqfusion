@echo off
REM Detached day/night slice, queued behind the fine-tune. Waits on the training
REM command line, then re-validates the 27 Phase 1 checkpoints.
cd /d A:\Uncertain
python -u scripts\queue_phase1_slice.py --poll 180 --control 3 --out runs/eval/phase1_day_night_slice.md >> runs\logs_cache_draws\queue_phase1_slice.log 2>&1
