@echo off
REM ============================================================
REM  AgentCreator self-test for native Windows.
REM  Runs setup (first time only), a system check and the test suite,
REM  and writes everything to the logs\ folder.
REM  Usage: windows_selftest.bat          (setup only if .venv is missing)
REM         windows_selftest.bat force    (re-run setup)
REM ============================================================
setlocal
cd /d "%~dp0"
if not exist logs mkdir logs
del /q logs\selftest.done 2>nul
set "AC_NOPAUSE=1"
set "PYTHONUTF8=1"

echo [1/4] Setup - log: logs\setup.log
echo       First run downloads about 3 GB ^(PyTorch^) and can take 10-30 minutes...
if /i "%~1"=="force" goto do_setup
if exist ".venv\Scripts\python.exe" (
  echo .venv already exists - setup skipped ^(run "windows_selftest.bat force" to redo it^) > logs\setup.log
  goto after_setup
)
:do_setup
call setup_windows.bat > logs\setup.log 2>&1
echo setup exit code: %errorlevel% >> logs\setup.log
:after_setup

echo [2/4] Environment info - log: logs\env.log
(
  echo === where python
  where python
  echo === py -0p
  py -0p
  echo === nvidia-smi
  nvidia-smi
  echo === ollama
  where ollama
  ollama list
) > logs\env.log 2>&1

if not exist ".venv\Scripts\python.exe" (
  echo Setup failed - see logs\setup.log
  echo SETUP_FAILED > logs\selftest.done
  pause
  exit /b 1
)

echo [3/4] System check - log: logs\doctor.log
".venv\Scripts\python.exe" -m distiller doctor > logs\doctor.log 2>&1

echo [4/4] Test suite - log: logs\pytest.log
".venv\Scripts\python.exe" -m pip install -q pytest > logs\pytest.log 2>&1
".venv\Scripts\python.exe" -m pytest -v -p no:cacheprovider tests >> logs\pytest.log 2>&1
echo pytest exit code: %errorlevel% >> logs\pytest.log

echo DONE > logs\selftest.done
echo.
echo Finished. Results are in the logs folder - you can close this window.
pause
