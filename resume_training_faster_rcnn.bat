@echo off
cd /d "%~dp0\WasteDetector"
echo ===================================================
echo Resuming Faster R-CNN Training from last.pth...
echo ===================================================
.\venv\Scripts\python.exe Train_FasterRCNN.py --resume runs/faster_rcnn/weights/last.pth --epochs 5 --batch-size 4 --grad-accum 4
pause
