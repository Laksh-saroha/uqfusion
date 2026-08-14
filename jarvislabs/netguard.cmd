@echo off
REM Network-loss guard for the upload phase. Launch DETACHED, same as the
REM supervisor (see jarvislabs\README.md) so a closing console cannot take it
REM down. It exits by itself once training is confirmed.
cd /d A:\Uncertain
jarvislabs\.venv-jl\Scripts\python.exe jarvislabs\netguard.py >> runs\netguard_stdout.log 2>&1
