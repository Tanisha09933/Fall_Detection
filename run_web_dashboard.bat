@echo off
title Fall Detection Web Dashboard
color 0E
echo ===============================================================================
echo       STARTING AI FALL DETECTION STREAMLIT WEB DASHBOARD
echo ===============================================================================
echo  Opening browser console at: http://localhost:8501
echo  Press Ctrl+C to terminate the dashboard.
echo ===============================================================================
echo.

streamlit run app.py

pause
