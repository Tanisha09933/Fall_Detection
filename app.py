"""
========================================================================================
  STREAMLIT WEB DASHBOARD: REAL-TIME AI FALL DETECTION & MONITORING CONSOLE
========================================================================================
"""

import os
import sys
import time
import tempfile
import cv2
import numpy as np
import streamlit as st

# Import the core pipeline
from fall_detection_live import FallDetectionPipeline

st.set_page_config(
    page_title="Elderly Fall Detection AI Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #78909C;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1A2130;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
    }
    .status-normal {
        color: #00E676;
        font-weight: bold;
    }
    .status-alert {
        color: #FF1744;
        font-weight: bold;
        animation: blinker 1s linear infinite;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛡️ Real-Time Person & Elderly Fall Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">YOLOv8-Pose Landmark Estimation & Biomechanical State Machine Architecture</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("⚙️ System Configuration")
source_mode = st.sidebar.radio(
    "Select Video Feed Source:",
    ["Sample Test Video", "Upload Video File", "Live Webcam"]
)

conf_threshold = st.sidebar.slider("Pose Confidence Threshold", 0.15, 0.80, 0.35, 0.05)
fall_velocity_threshold = st.sidebar.slider("Fall Velocity Trigger (v_y h/s)", 0.5, 2.5, 1.0, 0.1)
flat_angle_threshold = st.sidebar.slider("Flat Torso Angle Trigger (°)", 30, 75, 55, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖥️ Hardware Acceleration")
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"GPU Active: **{gpu_name}**")
else:
    st.sidebar.info("Running on CPU Mode")

# Main Layout
col_vid, col_stats = st.columns([3, 1])

with col_stats:
    st.subheader("📊 Live Telemetry")
    fps_display = st.empty()
    status_display = st.empty()
    angle_display = st.empty()
    velocity_display = st.empty()
    aspect_display = st.empty()
    st.markdown("---")
    alert_box = st.empty()

with col_vid:
    video_placeholder = st.empty()
    stop_button = st.button("⏹️ Stop Stream / Video")

# Video Source Resolution
video_path = None
is_webcam = False

if source_mode == "Sample Test Video":
    samples = {
        "Sample 1: Forward Fall & Inactivity": "sample_videos/fall_example_1.mp4",
        "Sample 2: Rapid Collapse Fall": "sample_videos/fall_example_2.mp4",
        "Sample 3: Quick Sudden Slip": "sample_videos/fall_example_3.mp4",
        "Sample 4: Room Walk": "sample_videos/fall_example_4.mp4"
    }
    selected_sample = st.sidebar.selectbox("Choose Sample Incident:", list(samples.keys()))
    video_path = samples[selected_sample]

elif source_mode == "Upload Video File":
    uploaded = st.sidebar.file_uploader("Upload an MP4 or AVI video", type=["mp4", "avi", "mov"])
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        video_path = tfile.name

elif source_mode == "Live Webcam":
    cam_index = st.sidebar.number_input("Webcam Device Index", min_value=0, max_value=5, value=0)
    video_path = cam_index
    is_webcam = True

# Pipeline Execution
if video_path is not None and not stop_button:
    if is_webcam:
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(int(video_path), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error(f"Could not open source: {video_path}")
    else:
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        pipeline = FallDetectionPipeline(
            model_name="yolov8n-pose.pt",
            conf_thresh=conf_threshold,
            fps=fps_in
        )
        pipeline.state_machine.fall_velocity_thresh = fall_velocity_threshold
        pipeline.state_machine.flat_angle_thresh = flat_angle_threshold
        pipeline.alarm.enabled = False  # Mute web audio by default

        prev_time = time.time()
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret or frame is None:
                if not is_webcam:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    break

            vis_frame, alerts = pipeline.process_frame(frame)

            curr_time = time.time()
            fps_val = 1.0 / max(curr_time - prev_time, 1e-4)
            prev_time = curr_time

            # Update Metrics Panel
            fps_display.metric("Stream FPS", f"{fps_val:.1f}")

            top_state = "MONITORING"
            top_angle = 90.0
            top_vy = 0.0
            top_ar = 0.45

            if pipeline.tracker.tracklets:
                for t in pipeline.tracker.tracklets.values():
                    top_state = t.state
                    if len(t.bbox_history) > 0 and len(t.keypoints_history) > 0:
                        f = pipeline.kinematics.extract_features(t.keypoints_history, t.bbox_history, {})
                        top_angle = f["torso_angle"]
                        top_vy = f["vertical_velocity"]
                        top_ar = f["aspect_ratio"]
                    break

            if top_state in ["FALL_DETECTED", "CONFIRMED_FALL"]:
                status_display.markdown(f"Status: <span class='status-alert'>🚨 {top_state}</span>", unsafe_allow_html=True)
                alert_box.error(f"🚨 **EMERGENCY DETECTED!** Fall triggered (Tilt: {top_angle:.1f}°, v_y: {top_vy:+.2f} h/s)")
            else:
                status_display.markdown(f"Status: <span class='status-normal'>🟢 {top_state}</span>", unsafe_allow_html=True)
                alert_box.success("✅ Subject In Normal Posture")

            angle_display.metric("Torso Inclination", f"{top_angle:.1f}°")
            velocity_display.metric("Vertical Descent", f"{top_vy:+.2f} h/s")
            aspect_display.metric("Aspect Ratio (W/H)", f"{top_ar:.2f}")

            # Stream video frame in RGB
            frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
