# PowerShell Launcher for InsightAnalyst AI
Set-Location -Path $PSScriptRoot

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Starting InsightAnalyst AI Streamlit Application     " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path ".\venv\Scripts\streamlit.exe") {
    Write-Host "[INFO] Using virtual environment Streamlit..." -ForegroundColor Green
    & ".\venv\Scripts\streamlit.exe" run app.py
} elseif (Test-Path ".\venv\Scripts\python.exe") {
    Write-Host "[INFO] Using venv python -m streamlit..." -ForegroundColor Green
    & ".\venv\Scripts\python.exe" -m streamlit run app.py
} else {
    Write-Host "[INFO] Using system streamlit..." -ForegroundColor Yellow
    streamlit run app.py
}
