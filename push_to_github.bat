@echo off
REM ============================================================
REM  Uploads this folder to https://github.com/404Founders-26/AgentCreator
REM  (code only - .venv, runs, models, logs and downloaded tools are ignored)
REM ============================================================
setlocal
title Upload AgentCreator to GitHub
cd /d "%~dp0"
set "HTTPS_URL=https://github.com/404Founders-26/AgentCreator.git"

where git >nul 2>nul
if errorlevel 1 goto nogit

REM ---- make sure git knows who is committing
for /f "delims=" %%a in ('git config user.email 2^>nul') do set "GEMAIL=%%a"
if defined GEMAIL goto have_identity
echo Git needs your name and email for the commit (use the email of your GitHub account).
set /p "GNAME=Your name: "
set /p "GEMAIL=Your email: "
git config --global user.name "%GNAME%"
git config --global user.email "%GEMAIL%"
:have_identity

echo.
echo === Committing ===
git add -A
git commit -q -m "AgentCreator: distil a big open LLM into a small, fast specialist" -m "Web app + CLI: pick teacher, student and specialty (built-in or custom); teacher generates verified training data, LoRA distillation with Unsloth, eval vs the untrained student, GGUF export to Ollama and a speed benchmark. Runs natively on Windows (AgentCreator.bat)." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01HJtWjyzNMnsZzNexTReWZC"
if errorlevel 1 echo (nothing new to commit)
git branch -M main

echo.
echo === Uploading ===
git push -u origin main
if not errorlevel 1 goto done
echo.
echo SSH upload failed - trying HTTPS instead (a GitHub sign-in window may open)...
git remote set-url origin %HTTPS_URL%
git push -u origin main
if not errorlevel 1 goto done
echo.
echo Upload failed. Check that your GitHub account can push to 404Founders-26/AgentCreator,
echo then run this file again.
pause
exit /b 1

:done
echo.
echo Uploaded: https://github.com/404Founders-26/AgentCreator
start "" https://github.com/404Founders-26/AgentCreator
pause
exit /b 0

:nogit
echo Git is not installed - installing it with winget, then run this file again.
winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements
pause
exit /b 1
