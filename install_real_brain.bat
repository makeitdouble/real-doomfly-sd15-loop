@echo off
setlocal
cd /d "%~dp0"

set "DOOMROOT=C:\doomfly-main"
set "PY=%DOOMROOT%\.venv-neural\Scripts\python.exe"
set "GRAPH=%DOOMROOT%\outputs\doom\malecns_v1\graph.npz"

if not exist "%PY%" (
  echo ERROR: Missing %PY%
  echo Run SETUP_REAL_CONNECTOME.bat in C:\doomfly-main first.
  pause
  exit /b 1
)
if not exist "%GRAPH%" (
  echo ERROR: Missing real graph: %GRAPH%
  echo Run SETUP_REAL_CONNECTOME.bat in C:\doomfly-main first.
  pause
  exit /b 1
)

echo Installing web/ComfyUI dependencies into the EXISTING DoomFly Python 3.11 venv...
"%PY%" -m pip install fastapi==0.116.1 uvicorn==0.35.0 requests==2.32.4 Jinja2==3.1.6 || goto :fail

echo.
echo OK. Real-brain runtime is ready.
pause
exit /b 0

:fail
echo.
echo INSTALL FAILED. Scroll to the first ERROR above.
pause
exit /b 1
