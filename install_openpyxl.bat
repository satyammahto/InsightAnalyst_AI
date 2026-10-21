@echo off
title Install Dependencies - InsightAnalyst AI
cd /d "%~dp0"

echo =======================================================
echo   Installing Dependencies (openpyxl, etc.)
echo =======================================================
echo.

IF EXIST "venv\Scripts\pip.exe" (
    echo [INFO] Detected virtual environment. Installing openpyxl...
    "venv\Scripts\pip.exe" install openpyxl
) ELSE (
    echo [INFO] Installing openpyxl via system pip...
    pip install openpyxl
)

echo.
echo =======================================================
echo [SUCCESS] openpyxl has been installed!
echo Return to your browser (http://localhost:8501) and re-upload your Excel file.
echo =======================================================
pause
