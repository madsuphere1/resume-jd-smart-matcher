@echo off
title Resume-JD Matcher (CLI Mode)
color 0E

echo.
echo ============================================================
echo    Resume - JD Matcher :: CLI Mode
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Please run setup.bat first!
    pause
    exit /b 1
)

echo Running pipeline on sample data...
echo.

.venv\Scripts\python.exe main.py

echo.
pause
