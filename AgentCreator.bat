@echo off
REM ==================================================================
REM  AgentCreator - double-click to run.
REM  First run: installs everything automatically (one time).
REM  Every run: makes sure Ollama + the teacher model are ready,
REM             starts the app and opens it in your browser.
REM ==================================================================
setlocal
title AgentCreator
cd /d "%~dp0"
set "HERE=%~dp0"
if not exist logs mkdir logs
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PORT=7860"
set "TEACHER=qwen3.5:9b"
set "URL=http://127.0.0.1:%PORT%"

echo.
echo   =============================================
echo     AgentCreator - big LLM in, fast specialist out
echo   =============================================
echo.

REM ---- already running? just open the browser
curl -s -o nul -m 2 %URL%/api/status >nul 2>nul
if not errorlevel 1 (
  echo AgentCreator is already running - opening %URL%
  start "" %URL%
  exit /b 0
)

REM ---- 1. first run: install Python packages, PyTorch GPU build, Unsloth
if exist ".venv\.setup_ok" goto setup_done
echo [1/4] First run - installing everything. This happens only once and takes
echo       10-30 minutes depending on your internet. Progress is shown below
echo       and saved to logs\setup.log.
echo.
set "AC_NOPAUSE=1"
if exist logs\setup.log del /q logs\setup.log
powershell -NoProfile -ExecutionPolicy Bypass -Command "cmd /c 'setup_windows.bat 2>&1' | ForEach-Object { $_; Add-Content -Path 'logs\setup.log' -Value $_ -Encoding utf8 }"
set "AC_NOPAUSE="
if not exist ".venv\.setup_ok" (
  echo.
  echo Setup did not finish - the reason is in logs\setup.log.
  echo Fix it, or just run AgentCreator.bat again to retry - finished steps are skipped.
  pause
  exit /b 1
)
call :make_shortcut
:setup_done
echo [1/4] Python packages ........ ready

REM ---- 2. Ollama (serves the teacher model)
REM (PATH is only changed outside parenthesised blocks: "Program Files (x86)" in PATH would break them)
where ollama >nul 2>nul
if not errorlevel 1 goto ollama_installed
if not exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" goto ollama_missing
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
goto ollama_installed
:ollama_missing
echo [2/4] Ollama is not installed - installing it with winget ...
winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
where ollama >nul 2>nul
if errorlevel 1 (
  echo Could not install Ollama automatically. Opening the download page -
  echo install it, then run AgentCreator.bat again.
  start "" https://ollama.com/download/windows
  pause
  exit /b 1
)
:ollama_installed
ollama list >nul 2>nul
if errorlevel 1 (
  echo       starting the Ollama server ...
  start "Ollama" /min ollama serve
  ping -n 6 127.0.0.1 >nul
)
echo [2/4] Ollama ................. ready

REM ---- 3. teacher model (downloads in its own window so the app can open meanwhile)
ollama list 2>nul | findstr /i /l /c:"%TEACHER%" >nul
if errorlevel 1 (
  echo [3/4] Teacher model %TEACHER% is not downloaded yet - downloading it in a
  echo       separate window, about 6.6 GB. You can set up a run while it downloads.
  start "Downloading teacher model %TEACHER%" cmd /c "ollama pull %TEACHER% && echo. && echo Download finished - you can close this window. && pause"
) else (
  echo [3/4] Teacher model .......... %TEACHER% ready
)

REM ---- 4. start the web app and open the browser
echo [4/4] Starting the app at %URL%
echo.
echo       Keep this window open while you use AgentCreator.
echo       To stop it: click Quit in the app, or close this window.
echo.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process '%URL%'"
".venv\Scripts\python.exe" -m distiller ui --port %PORT%
if errorlevel 1 (
  echo.
  echo The app stopped with an error - the message is shown above.
  pause
)
exit /b 0

:make_shortcut
REM Desktop shortcut so next time it's one double-click from the desktop
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$d=[Environment]::GetFolderPath('Desktop'); $s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'AgentCreator.lnk'));" ^
  "$s.TargetPath='%HERE%AgentCreator.bat'; $s.WorkingDirectory='%HERE%'; $s.IconLocation='%SystemRoot%\System32\imageres.dll,109'; $s.Save()" >nul 2>nul
exit /b 0
