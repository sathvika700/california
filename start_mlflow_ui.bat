@echo off
cd /d "%~dp0"
set "PORT=%~1"
if not defined PORT set "PORT=5000"

if exist "%~dp0myenv\Scripts\mlflow.exe" (
    "%~dp0myenv\Scripts\mlflow.exe" ui --host 127.0.0.1 --port %PORT% --workers 1
) else (
    mlflow ui --host 127.0.0.1 --port %PORT% --workers 1
)
