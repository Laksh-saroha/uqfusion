@echo off
REM Store the JarvisLabs API key without typing it into a masked prompt.
REM
REM   1. paste the key into a plain text file saved as:
REM          A:\Uncertain\jarvislabs\jlkey.txt
REM      (Notepad accepts a normal Ctrl+V; the CLI's hidden prompt often does not)
REM   2. double-click this file, or run it from any terminal
REM
REM It reads the file, hands the value straight to `jl setup --token`, deletes the
REM file, and prints the account status. The key is never echoed and never appears
REM in your shell history — the command line here holds a variable, not the value.
REM `jl setup` writes it to %LOCALAPPDATA%\jl\config.toml with 0600 permissions;
REM jlkey.txt is gitignored either way.
setlocal
set KEYFILE=%~dp0jlkey.txt
set JL=%~dp0.venv-jl\Scripts\jl.exe

if not exist "%KEYFILE%" (
    echo [set_key] no key file found.
    echo [set_key] paste your API key into: %KEYFILE%
    echo [set_key] then run this again.
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$k = (Get-Content -Raw '%KEYFILE%').Trim(); if (-not $k) { Write-Host '[set_key] key file is empty'; exit 1 }; & '%JL%' setup --token $k --yes"
if errorlevel 1 (
    echo [set_key] setup failed - key file kept so you can retry.
    exit /b 1
)

del /f /q "%KEYFILE%"
echo [set_key] key stored, %KEYFILE% deleted.
"%JL%" status
