@echo off
title Investment Engine Dashboard - Auto Launcher
color 0A

echo ================================================
echo         INVESTMENT ENGINE DASHBOARD
echo         Fully Automated - Zero Config!
echo ================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

:: Run the launcher
echo Starting dashboard...
python launcher.py

:: Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Dashboard stopped with an error.
    pause
)
