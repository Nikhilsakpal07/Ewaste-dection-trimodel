@echo off
cd /d "%~dp0\WasteDetector"
echo ===================================================
echo Starting YOLOv5 (Single-stage CSPDarknet53) Training...
echo ===================================================
.\venv\Scripts\python.exe Train_YOLOv5.py --epochs 5 --batch-size 16 --weights yolov5nu.pt
pause
