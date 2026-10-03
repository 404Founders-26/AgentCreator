@echo off
REM ============================================================
REM  AgentCreator - one-time setup on native Windows (Python 3.11-3.14; 3.13 recommended)
REM  Usage:  setup_windows.bat            (normally started for you by AgentCreator.bat)
REM          setup_windows.bat cu126      (pick another CUDA wheel if needed)
REM ============================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "CUDA_TAG=%~1"
if "%CUDA_TAG%"=="" set "CUDA_TAG=cu128"

echo.
echo === AgentCreator setup (native Windows, PyTorch %CUDA_TAG%) ===
echo.
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo WARNING: nvidia-smi not found. Install the latest NVIDIA Game Ready / Studio driver first.
) else (
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
)

REM ---- find Python. Unsloth supports 3.11-3.13; Python 3.14 works but without Unsloth (slower training).
REM      If several versions are installed, the best one for training is picked automatically.
set "PY="
set "NO_UNSLOTH="
py -3.13 -c "import sys" >nul 2>nul && set "PY=py -3.13"
if not defined PY py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
if not defined PY py -3.11 -c "import sys" >nul 2>nul && set "PY=py -3.11"
if not defined PY python -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul && set "PY=python"
if not defined PY py -3.14 -c "import sys" >nul 2>nul && set "PY=py -3.14" && set "NO_UNSLOTH=1"
if not defined PY python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,14) else 1)" >nul 2>nul && set "PY=python" && set "NO_UNSLOTH=1"
if not defined PY (
  echo.
  echo ERROR: no supported Python found ^(3.11 - 3.14^).
  echo        Install Python 3.13 from https://www.python.org/downloads/windows/
  echo        ^(tick "Add python.exe to PATH"^), then run this file again.
  if not defined AC_NOPAUSE pause
  exit /b 1
)
echo Using: %PY%
if defined NO_UNSLOTH (
  echo.
  echo NOTE: only Python 3.14 was found. Unsloth does not support 3.14 yet, so training will use
  echo       transformers + peft: it works, but is slower and uses more VRAM.
  echo       For the fast path, also install Python 3.13 ^(it can live next to 3.14^), delete the .venv
  echo       folder and run this file again - the 3.13 install will be picked automatically.
  echo.
)

REM ---- virtual environment
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv
  if errorlevel 1 ( echo ERROR: could not create .venv & (if not defined AC_NOPAUSE pause) & exit /b 1 )
)
set "VPY=%CD%\.venv\Scripts\python.exe"
set "PYTHONUTF8=1"

"%VPY%" -m pip install --upgrade pip wheel setuptools
if errorlevel 1 ( echo ERROR: pip upgrade failed & (if not defined AC_NOPAUSE pause) & exit /b 1 )

REM ---- app + training packages FIRST (these may pull a CPU-only torch from PyPI - that's fine,
REM      it is small; we swap in the GPU build of the exact same version afterwards so nothing downloads twice)
echo.
echo === [1/3] Installing AgentCreator packages ===
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 ( echo ERROR: core packages failed & (if not defined AC_NOPAUSE pause) & exit /b 1 )
if not defined NO_UNSLOTH (
  "%VPY%" -m pip install unsloth
  if errorlevel 1 echo WARNING: Unsloth did not install - training will use transformers + peft instead.
)
"%VPY%" -m pip install -r requirements-windows.txt
if errorlevel 1 ( echo ERROR: training packages failed & (if not defined AC_NOPAUSE pause) & exit /b 1 )

REM ---- GPU build of PyTorch, same version the packages above asked for
echo.
echo === [2/3] Installing the GPU (CUDA) build of PyTorch - about 3 GB ===
"%VPY%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if not errorlevel 1 goto torch_ok
set "TV="
set "TVV="
for /f "usebackq delims=" %%v in (`"%VPY%" -c "import torch; print(torch.__version__.split('+')[0])"`) do set "TV=%%v"
REM torchvision must match torch exactly (torch 2.N pairs with torchvision 0.(N+15)), otherwise
REM transformers fails with "operator torchvision::nms does not exist"
for /f "usebackq delims=" %%v in (`"%VPY%" -c "import torch; v=torch.__version__.split('+')[0].split('.'); print('0.%%d.%%s' %% (int(v[1])+15, v[2]))"`) do set "TVV=%%v"
set "PKGS=torch==%TV%"
if defined TVV set "PKGS=%PKGS% torchvision==%TVV%"
if not defined TV set "PKGS=torch"
echo Need: %PKGS%
for %%c in (%CUDA_TAG% cu130 cu128 cu126) do (
  echo Trying CUDA wheels %%c ...
  "%VPY%" -m pip install --force-reinstall --no-deps %PKGS% --index-url https://download.pytorch.org/whl/%%c
  "%VPY%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul && goto torch_ok
)
echo WARNING: no CUDA build of %PKGS% found - installing the newest CUDA PyTorch instead.
"%VPY%" -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/%CUDA_TAG%
:torch_ok
REM torchvision built for a different torch breaks transformers - repair it if needed
"%VPY%" -c "import torchvision" >nul 2>nul
if not errorlevel 1 goto tv_ok
for /f "usebackq delims=" %%v in (`"%VPY%" -c "import torch; v=torch.__version__.split('+')[0].split('.'); print('0.%%d.%%s' %% (int(v[1])+15, v[2]))"`) do set "TVV=%%v"
for /f "usebackq delims=" %%v in (`"%VPY%" -c "import torch; print('cu'+(torch.version.cuda or '').replace('.',''))"`) do set "TCU=%%v"
echo Repairing torchvision (installing %TVV% for %TCU%) ...
"%VPY%" -m pip install --no-deps --force-reinstall torchvision==%TVV% --index-url https://download.pytorch.org/whl/%TCU%
:tv_ok
"%VPY%" -c "import torch; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')"

echo ok> ".venv\.setup_ok"
echo.
echo === [3/3] System check ===
"%VPY%" -m distiller doctor
echo.
echo Setup finished.
if not defined AC_NOPAUSE pause
