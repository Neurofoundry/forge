@echo off
echo ============================================
echo   Starting ALL Services
echo ============================================
echo.
echo Starting ControlNet service on port 5002...
start "ControlNet Service" cmd /k "python controlnet_service.py"

timeout /t 3 /nobreak >nul

echo Starting Composition service on port 5001...
start "Composition Service" cmd /k "python subject_extractor.py"

echo.
echo ============================================
echo   All services starting!
echo ============================================
echo.
echo Port 5001: Composition + Background Removal
echo Port 5002: ControlNet Style Transform
echo.
echo Browser will open automatically...
echo Close this window to keep services running
echo.
pause
