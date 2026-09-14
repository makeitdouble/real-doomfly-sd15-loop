@echo off
setlocal
cd /d "%~dp0"

set "DOOMROOT=C:\doomfly-main"
set "PY=%DOOMROOT%\.venv-neural\Scripts\python.exe"
set "GRAPH=%DOOMROOT%\outputs\doom\malecns_v1\graph.npz"
set "MANIFEST=%DOOMROOT%\outputs\doom\malecns_v1\manifest.json"

if not exist "%PY%" (
  echo ERROR: Missing DoomFly Python: %PY%
  echo Run install_real_brain.bat after preparing the connectome.
  pause
  exit /b 1
)
if not exist "%GRAPH%" (
  echo ERROR: Missing real graph: %GRAPH%
  pause
  exit /b 1
)
if not exist "%MANIFEST%" (
  echo ERROR: Missing manifest: %MANIFEST%
  pause
  exit /b 1
)

set "PYTHONPATH=%DOOMROOT%;%PYTHONPATH%"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "FLY_GENERATION_BACKEND=comfyui"

echo REAL DOOMFLY MaleCNS v1.0 + ComfyUI
 echo Graph: %GRAPH%
 echo UI: http://127.0.0.1:8951
 echo.
"%PY%" app.py
set CODE=%ERRORLEVEL%
echo.
echo app.py exited with code %CODE%.
pause
exit /b %CODE%
