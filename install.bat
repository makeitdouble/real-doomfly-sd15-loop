@echo off
setlocal
cd /d "%~dp0"

echo === FLY SD1.5 LOOP installer ===

if exist ".venv\Scripts\python.exe" goto install_deps

set "PY_CMD="
py -3.11 -c "import sys" >nul 2>&1 && set "PY_CMD=py -3.11"
if not defined PY_CMD py -3.10 -c "import sys" >nul 2>&1 && set "PY_CMD=py -3.10"
if not defined PY_CMD python -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1 && set "PY_CMD=python"

if not defined PY_CMD (
  echo ERROR: Python 3.10+ not found.
  echo Install Python 3.10 or newer and run this file again.
  pause
  exit /b 1
)

echo Using: %PY_CMD%
echo Creating .venv...
%PY_CMD% -m venv .venv
if errorlevel 1 goto fail

:install_deps
set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" goto fail

echo Installing dependencies...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto fail
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
echo Install complete.
echo Next: run_mock.bat first, then run_server.bat.
pause
exit /b 0

:fail
echo.
echo ERROR: installation failed. The loop was NOT installed.
pause
exit /b 1
