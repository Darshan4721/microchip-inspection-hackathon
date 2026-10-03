@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo =======================================================
echo Starting VELTRAXX Presentation Desktop App...
echo =======================================================
call .venv\Scripts\python.exe veltraxx_ui.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while running VELTRAXX.
    pause
)
