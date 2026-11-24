@echo off
title Fall Detection System - Pre-recorded Video Demo
color 0B
echo ===============================================================================
echo       REAL-TIME FALL DETECTION SYSTEM (SAMPLE VIDEO DEMO)
echo ===============================================================================
echo  Running on real pre-recorded fall incident video: fall_example_1.mp4
echo  Press [V] during playback to switch to Webcam or other demo videos!
echo ===============================================================================
echo.

python fall_detection_live.py --source sample_videos/fall_example_1.mp4

pause
