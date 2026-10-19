@echo off
cd /d "%~dp0"
echo ===================================================
echo Starting Faster R-CNN (Two-stage RPN) Training...
echo ===================================================
.\venv\Scripts\python.exe Train_FasterRCNN.py --epochs 5 --batch-size 4 --grad-accum 4 --lr 0.0005
pause
