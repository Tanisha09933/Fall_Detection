@echo off
title Fall Detection System - Live Webcam Demo
color 0A
echo ===============================================================================
echo        REAL-TIME PERSON & ELDERLY FALL DETECTION SYSTEM (LIVE WEBCAM)
echo ===============================================================================
echo  Hardware Acceleration : NVIDIA GeForce RTX 4060 (CUDA)
echo  Deep Learning Model   : YOLOv8-Pose (17 Human Keypoints)
echo  Kinematics Engine     : Real-time Torso Inclination & Vertical Velocity (v_y)
echo  Alert System          : Instant Visual Banner + Audible Frequency Siren
echo -------------------------------------------------------------------------------
echo  Key Controls:
echo    [Q] or [ESC] : Quit application
echo    [R]          : Reset all alerts
echo    [M]          : Toggle audio alarm (Mute / Unmute)
echo    [S]          : Save high-resolution event snapshot
echo    [V]          : Toggle between Webcam (0) and Sample Demo Video
echo    [SPACE]      : Pause / Resume live feed
echo ===============================================================================
echo.

python fall_detection_live.py --source 0

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Webcam 0 could not be opened or encountered an error.
    echo [!] Attempting fallback to pre-recorded test fall video...
    echo.
    python fall_detection_live.py --source sample_videos/fall_example_1.mp4
)

pause
