@echo off
title Resume-JD Matcher Setup
color 0A

echo.
echo ============================================================
echo    Resume - JD Smart Matcher :: One-Time Setup
echo ============================================================
echo.

:: -----------------------------------------------------------
:: 1. Check Python
:: -----------------------------------------------------------
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
python --version
echo       Python found!
echo.

:: -----------------------------------------------------------
:: 2. Create virtual environment
:: -----------------------------------------------------------
echo [2/5] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo       Virtual environment created.
) else (
    echo       Virtual environment already exists. Skipping.
)
echo.

:: -----------------------------------------------------------
:: 3. Install Python dependencies
:: -----------------------------------------------------------
echo [3/5] Installing Python dependencies...
echo       This may take a few minutes on first run...
echo.
.venv\Scripts\pip.exe install --upgrade pip >nul 2>&1
.venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies.
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)
echo.
echo       Dependencies installed successfully!
echo.

:: -----------------------------------------------------------
:: 4. Install additional packages not in requirements.txt
:: -----------------------------------------------------------
echo [4/5] Installing additional packages...
.venv\Scripts\pip.exe install keybert trafilatura pdf2image opencv-python-headless pdfplumber 2>nul
echo       Additional packages installed!
echo.

:: -----------------------------------------------------------
:: 5. Check Ollama
:: -----------------------------------------------------------
echo [5/5] Checking Ollama (for LLM features)...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Ollama is not installed.
    echo LLM features (match explanations, improvement suggestions) will not work.
    echo.
    echo To install Ollama:
    echo   1. Visit https://ollama.com/download
    echo   2. Download and install for Windows
    echo   3. Open a terminal and run:  ollama pull qwen2.5:1.5b
    echo.
    echo The app still works without Ollama -- just toggle off LLM in the sidebar.
    echo.
) else (
    echo       Ollama found! Pulling model...
    ollama pull qwen2.5:1.5b
    echo       Model ready!
    echo.
)

:: -----------------------------------------------------------
:: Done
:: -----------------------------------------------------------
echo.
echo ============================================================
echo    SETUP COMPLETE!
echo ============================================================
echo.
echo    To run the app, double-click:  run.bat
echo    Or run:  .venv\Scripts\python.exe -m streamlit run app_ui.py
echo.
echo ============================================================
echo.
pause
