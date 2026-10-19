@echo off
cd /d "%~dp0"
echo ===================================================
echo Resuming YOLOv5 Training from last checkpoint...
echo ===================================================
.\venv\Scripts\python.exe Train_YOLOv5.py --resume --epochs 5 --batch-size 16
pause
