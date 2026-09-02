@echo off
REM Detached relaunch of the night-restore fine-tune (resumes from weights/last.pt).
REM Started via Win32_Process.Create so its parent is the WMI host, not a Claude
REM Code shell -- it survives the session that launched it.
cd /d A:\Uncertain
python -u scripts\train_night_restore.py --resume >> runs\logs_cache_draws\train_night_restore.log 2>&1
