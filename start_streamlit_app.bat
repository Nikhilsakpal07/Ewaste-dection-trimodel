@echo off
cd /d "%~dp0\WasteDetector"
echo ===================================================================
echo   Starting ReciBot Tri-Model Streamlit Dashboard
echo ===================================================================
echo.
if exist ".\venv\Scripts\streamlit.exe" (
    .\venv\Scripts\streamlit.exe run streamlit_app.py
) else (
    streamlit run streamlit_app.py
)
pause
