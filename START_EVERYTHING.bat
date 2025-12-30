@echo off
echo ============================================
echo   Starting AI Composition Pipeline
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

REM Check if dependencies are installed
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies for first time...
    pip install -r requirements.txt
    echo.
)

echo Starting server on http://localhost:5001
echo Opening test interface...
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

REM Open browser after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5001/static/test_composition_pipeline.html"

REM Start server (this blocks)
python subject_extractor.py
