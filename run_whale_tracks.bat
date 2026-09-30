@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo =======================================================
echo Starting Whale Tracks Presentation Desktop App...
echo =======================================================
call .venv\Scripts\python.exe whale_tracks_ui.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while running Whale Tracks.
    pause
)
