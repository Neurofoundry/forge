@echo off
echo ============================================
echo   AI Image Fusion - Quick Start
echo ============================================
echo.
echo Starting composition service...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

REM Check dependencies
python -c "import flask, rembg" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Service starting on http://localhost:5001
echo Browser will open automatically...
echo.
echo Press Ctrl+C to stop
echo ============================================
echo.

python subject_extractor.py
pause
