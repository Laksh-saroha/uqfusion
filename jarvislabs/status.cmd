@echo off
REM One-shot progress + spend snapshot for the rented L4. Safe to run any time,
REM as often as you like: it only reads (SDK get + one ssh probe), never controls
REM the instance. The supervisor keeps running independently.
REM
REM   A:\Uncertain\jarvislabs\status.cmd
cd /d A:\Uncertain
jarvislabs\.venv-jl\Scripts\python.exe jarvislabs\watch_l4.py --once
