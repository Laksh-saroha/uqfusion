@echo off
REM Bring the run back after a halt (top-up, spot capacity, anything that needed a
REM human). Resumes the instance as SPOT, relaunches the grid, and supervises it.
REM
REM One command, because the launch/confirm logic now lives in watch_l4.py: the
REM bash launcher it replaced had its own probe regex and would have paused a
REM healthy epoch-1 run when that regex missed.
cd /d A:\Uncertain
jarvislabs\.venv-jl\Scripts\jl.exe resume %1 --spot --yes
jarvislabs\.venv-jl\Scripts\python.exe jarvislabs\watch_l4.py --machine-id %1 --launch --clear-halt --interval 300 >> runs\watch_l4_stdout.log 2>&1
