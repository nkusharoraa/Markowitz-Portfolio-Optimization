@echo off
REM Markowitz Portfolio Optimization - Virtual Environment Setup Script
REM This script creates a Python virtual environment and installs all dependencies

echo ========================================
echo Markowitz Portfolio Optimization Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version

echo.
echo [2/4] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Removing old version...
    rmdir /s /q venv
)
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [4/4] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To activate the virtual environment, run:
echo    venv\Scripts\activate
echo.
echo To run the portfolio optimization script:
echo    python markowitz_portfolio.py
echo.
echo To deactivate the virtual environment:
echo    deactivate
echo.
pause
