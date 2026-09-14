@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Venv not found. Running install first...
  call install.bat
  if errorlevel 1 exit /b 1
)
set "FLY_GENERATION_BACKEND=mock"
start "" http://127.0.0.1:8951
"%PY%" app.py
if errorlevel 1 (
  echo.
  echo ERROR: app.py failed. If port 8951 is busy, close the other loop console first.
  pause
  exit /b 1
)
