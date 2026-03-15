@echo off
title Resume-JD Matcher
color 0B

echo.
echo ============================================================
echo    Resume - JD Smart Matcher
echo    Starting Streamlit Dashboard...
echo ============================================================
echo.

:: Check if venv exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Please run setup.bat first!
    echo.
    pause
    exit /b 1
)

:: Start Ollama in background (if installed)
ollama --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting Ollama server in background...
    start /min "" ollama serve 2>nul
    timeout /t 2 /nobreak >nul
)

echo.
echo Opening browser at http://localhost:8501 ...
echo Press Ctrl+C in this window to stop the server.
echo.

:: Run Streamlit
.venv\Scripts\python.exe -m streamlit run app_ui.py --server.headless true --browser.gatherUsageStats false

pause
