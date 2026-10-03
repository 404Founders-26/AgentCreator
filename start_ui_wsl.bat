@echo off
REM Starts the AgentCreator UI inside WSL and opens the browser.
start "" http://127.0.0.1:7860
wsl -e bash -lc "cd \"$(wslpath '%~dp0')\" && source .venv/bin/activate && python -m distiller ui --host 0.0.0.0"
