@echo off
REM Supervisor for the rented-L4 yolo26x runs. Launch DETACHED (see
REM jarvislabs\README.md) so a console teardown cannot take it down with it.
REM
REM It reads the machine id from jarvislabs\l4_state.json, so this file needs no
REM edit between instances. JL points at the isolated venv the jl CLI lives in --
REM never the project venv, whose package set is the running experiment.
cd /d A:\Uncertain
jarvislabs\.venv-jl\Scripts\python.exe jarvislabs\watch_l4.py --interval 300 >> runs\watch_l4_stdout.log 2>&1
