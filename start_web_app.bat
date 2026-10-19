@echo off
cd /d "%~dp0\WasteDetector"
echo ===================================================================
echo   Starting ReciBot Tri-Model E-Waste AI Web Dashboard
echo   Models: YOLOv8 SOTA + YOLOv5 Baseline + Faster R-CNN (Parallel)
echo   Report Engine: Google Gemini LLM
echo ===================================================================
echo.
echo Opening browser to: http://localhost:5000
start http://localhost:5000
echo.
if exist ".\venv\Scripts\python.exe" (
    .\venv\Scripts\python.exe web_app.py
) else (
    python web_app.py
)
pause
