@echo off
rem Double-click to install (first time), check the data and start training + dashboard.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1" %*
pause
