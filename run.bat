@echo off
chcp 65001 >nul
title Cyberbullying Detector App
echo ==========================================
echo  Starting Cyberbullying Detector + Safe Reply
echo ==========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\streamlit.exe" (
    echo [ERROR] Virtual environment not found or incomplete.
    echo         Please run setup.bat first.
    pause
    exit /b 1
)

REM Launch Streamlit
echo Launching app in your browser...
echo.
venv\Scripts\streamlit.exe run app.py

pause
