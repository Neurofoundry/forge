@echo off
echo Starting ControlNet + SDXL Service...
echo.
echo This will download ~6GB of models on first run
echo Make sure you have a GPU with at least 8GB VRAM
echo.
python controlnet_service.py
pause
