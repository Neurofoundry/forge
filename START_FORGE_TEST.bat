@echo off
setlocal

echo ============================================
echo   Starting Forge Test Stack
echo ============================================
echo.

set "FORGE_DIR=D:\0___TESTZONE\_theneurofoundry\import_scripts\Forge"
set "PYTHON_EXE=C:\Users\Neurofoundry\AppData\Local\Programs\Python\Python310\python.exe"
set "VENV_PY=%FORGE_DIR%\.venv\Scripts\python.exe"
if exist "%VENV_PY%" set "PYTHON_EXE=%VENV_PY%"
set "TEST_URL=http://127.0.0.1:8000/SUBJECT_MERGE_TEST_copy.html"

echo Starting Rembg/Composer service (port 5001)...
start "Forge Extractor" cmd /k "cd /d %FORGE_DIR% && set PORT=5001 && %PYTHON_EXE% subject_extractor.py"

timeout /t 3 /nobreak >nul

echo Starting local web server (port 8000)...
start "Forge Web" cmd /k "cd /d %FORGE_DIR% && %PYTHON_EXE% -m http.server 8000"

timeout /t 3 /nobreak >nul

echo Opening test page...
start "" "%TEST_URL%"

echo.
echo ============================================
echo   Stack launched.
echo   Close service windows to stop.
echo ============================================
echo.
pause
