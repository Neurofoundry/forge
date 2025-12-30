@echo off
setlocal

echo ============================================
echo   Starting Forge Test Stack
echo ============================================
echo.

set "PYTHON_EXE=C:\Users\Neurofoundry\AppData\Local\Programs\Python\Python310\python.exe"
set "FORGE_DIR=D:\0___TESTZONE\_theneurofoundry\import_scripts\Forge"
set "AI_APP_DIR=D:\0___TESTZONE\AI_BasedApp"
set "TEST_PAGE=%FORGE_DIR%\SUBJECT_MERGE_TEST_copy.html"

echo Starting Rembg/Composer service (port 5001)...
start "Forge Extractor" cmd /k "cd /d %FORGE_DIR% && %PYTHON_EXE% subject_extractor.py"

timeout /t 3 /nobreak >nul

echo Starting Core API stack (port 8080)...
start "Neuroforge Core" cmd /k "cd /d %AI_APP_DIR% && %PYTHON_EXE% unified_launcher.py"

timeout /t 3 /nobreak >nul

echo Opening test page...
start "" "%TEST_PAGE%"

echo.
echo ============================================
echo   Stack launched.
echo   Close service windows to stop.
echo ============================================
echo.
pause
