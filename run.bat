@echo off
title InsightAnalyst AI - Streamlit Launcher
cd /d "%~dp0"

echo =======================================================
echo   Starting InsightAnalyst AI
echo =======================================================
echo.

IF EXIST "venv\Scripts\streamlit.exe" (
    echo [INFO] Detected virtual environment. Launching Streamlit...
    "venv\Scripts\streamlit.exe" run app.py
) ELSE IF EXIST "venv\Scripts\python.exe" (
    echo [INFO] Launching via venv python module...
    "venv\Scripts\python.exe" -m streamlit run app.py
) ELSE (
    echo [INFO] Using system Streamlit...
    streamlit run app.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application failed to start.
    echo Press any key to exit...
    pause >nul
)
