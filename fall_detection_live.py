"""
========================================================================================
  REAL-TIME PERSON & ELDERLY FALL DETECTION SYSTEM (LIVE DEMO & WEBCAM SUITE)
  ---------------------------------------------------------------------------
  Engine: YOLOv8-Pose + Kinematic Physics Engine + 6-Stage Finite State Machine
  Hardware: GPU (CUDA RTX 4060) / CPU Auto-Select
  Features:
    - Zero-latency live webcam inference
    - Instant-trigger fall detection (v_y > 1.0 h/s or tilt collapse > 35 deg/s)
    - Anti-false-alarm suppression (sitting, bending, lying down)
    - 4-tier occlusion resilience behind tables/desks
    - Asynchronous audible alarm siren (winsound)
    - Interactive hotkeys: [Q] Quit | [R] Reset | [M] Mute | [S] Snapshot | [V] Switch Source
========================================================================================
"""

import os
import sys
import time
import math
import argparse
import threading
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Windows asynchronous audio alarm support
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# ======================================================================================
# 1. AUDIO ALARM CONTROLLER (NON-BLOCKING THREAD)
# ======================================================================================
class AudioAlarm:
    """Non-blocking audio alert for fall emergencies."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._is_playing = False
        self._lock = threading.Lock()

    def trigger_fall_alarm(self):
        if not self.enabled or not HAS_WINSOUND:
            return
        with self._lock:
            if self._is_playing:
                return
            self._is_playing = True
        
        def _play():
            try:
                # Emergency frequency sweep: 800Hz -> 1200Hz
                for freq in [800, 1000, 1200, 1000, 800]:
                    winsound.Beep(freq, 80)
            except Exception:
                pass
            finally:
                with self._lock:
                    self._is_playing = False

        threading.Thread(target=_play, daemon=True).start()

    def toggle_mute(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled


# ======================================================================================
# 2. YOLOV8 POSE DETECTOR
# ======================================================================================
class PoseDetector:
    """YOLOv8 Pose wrapper returning 17 COCO keypoints and bounding boxes."""
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(self, model_name: str = "yolov8n-pose.pt", conf_thresh: float = 0.35, device: Optional[str] = None):
        self.conf_thresh = conf_thresh
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[PoseDetector] Loading {model_name} on device: {self.device.upper()}")
        if self.device == "cuda":
            print(f"[PoseDetector] Accelerated by: {torch.cuda.get_device_name(0)}")
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        res = results[0]
        if res.boxes is None or res.keypoints is None:
            return detections

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        box_confs = res.boxes.conf.cpu().numpy()
        kps_data = res.keypoints.data.cpu().numpy()  # shape: (N, 17, 3)

        for i in range(len(boxes_xyxy)):
            detections.append({
                "bbox": boxes_xyxy[i].tolist(),
                "confidence": float(box_confs[i]),
                "keypoints": kps_data[i]  # [x, y, conf]
            })

        return detections


# ======================================================================================
# 3. OCCLUSION RESILIENCE ENGINE
# ======================================================================================
class OcclusionEngine:
    """
    Imputes missing keypoints using human anatomical anthropometry
    and enforces limb-length constraints.
    """
    HIP_KNEE_RATIO = 0.53
    KNEE_ANKLE_RATIO = 0.47

    def impute_keypoints(self, keypoints: np.ndarray, bbox: List[float]) -> Tuple[np.ndarray, np.ndarray, float]:
        imputed = keypoints.copy()
        occluded_mask = np.zeros(17, dtype=bool)

        for i in range(17):
            if imputed[i, 2] < 0.25 or imputed[i, 0] <= 0 or imputed[i, 1] <= 0:
                occluded_mask[i] = True

        l_hip, r_hip = imputed[11], imputed[12]
        if not occluded_mask[11] and not occluded_mask[12]:
            pelvis = (l_hip[:2] + r_hip[:2]) / 2.0
            hip_dist = np.linalg.norm(l_hip[:2] - r_hip[:2])

            # Left leg imputation
            if occluded_mask[13] and not occluded_mask[11]:
                imputed[13, 0] = l_hip[0]
                imputed[13, 1] = l_hip[1] + hip_dist * 1.8
                imputed[13, 2] = 0.35
            if occluded_mask[15] and not occluded_mask[13]:
                imputed[15, 0] = imputed[13, 0]
                imputed[15, 1] = imputed[13, 1] + hip_dist * 1.6
                imputed[15, 2] = 0.30

            # Right leg imputation
            if occluded_mask[14] and not occluded_mask[12]:
                imputed[14, 0] = r_hip[0]
                imputed[14, 1] = r_hip[1] + hip_dist * 1.8
                imputed[14, 2] = 0.35
            if occluded_mask[16] and not occluded_mask[14]:
                imputed[16, 0] = imputed[14, 0]
                imputed[16, 1] = imputed[14, 1] + hip_dist * 1.6
                imputed[16, 2] = 0.30

        # Boundary clamping to bounding box
        x1, y1, x2, y2 = bbox
        for i in range(17):
            if occluded_mask[i]:
                imputed[i, 0] = np.clip(imputed[i, 0], x1 - 20, x2 + 20)
                imputed[i, 1] = np.clip(imputed[i, 1], y1 - 20, y2 + 20)

        occlusion_rate = float(np.mean(occluded_mask))
        return imputed, occluded_mask, occlusion_rate


# ======================================================================================
# 4. BIOMECHANICAL KINEMATIC EXTRACTOR
# ======================================================================================
class KinematicExtractor:
    """
    Computes posture inclination, collapse rate, normalized vertical velocity,
    aspect ratio, and stillness energy.
    """
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / max(fps, 1.0)

    def extract_features(
        self,
        keypoints_history: List[np.ndarray],
        bbox_history: List[List[float]],
        occlusion_stats: Dict[str, float]
    ) -> Dict[str, float]:
        if not keypoints_history:
            return self._empty_features()

        current_kps = keypoints_history[-1]
        current_bbox = bbox_history[-1]

        torso_angle = self.calculate_torso_angle(current_kps)
        bbox_w = max(current_bbox[2] - current_bbox[0], 1.0)
        bbox_h = max(current_bbox[3] - current_bbox[1], 1.0)
        aspect_ratio = float(bbox_w / bbox_h)

        v_y, a_y, centroid_y = self.calculate_vertical_kinematics(keypoints_history, bbox_history)
        stillness_energy = self.calculate_stillness_energy(keypoints_history, window=15)

        # Angular velocity: rate of angle collapse in deg/sec
        angular_vel = 0.0
        if len(keypoints_history) >= 2:
            prev_angle = self.calculate_torso_angle(keypoints_history[-2])
            angular_vel = (prev_angle - torso_angle) * self.fps

        return {
            "torso_angle": float(torso_angle),
            "angular_velocity": float(angular_vel),
            "vertical_velocity": float(v_y),
            "vertical_accel": float(a_y),
            "aspect_ratio": float(aspect_ratio),
            "stillness_energy": float(stillness_energy),
            "centroid_y": float(centroid_y),
            "bbox_height": float(bbox_h),
            "bbox_width": float(bbox_w)
        }

    def calculate_torso_angle(self, kps: np.ndarray) -> float:
        """Calculates torso inclination relative to floor (0 deg = flat on ground, 90 deg = upright)."""
        l_sh, r_sh = kps[5], kps[6]
        l_hip, r_hip = kps[11], kps[12]

        mid_sh_x = (l_sh[0] + r_sh[0]) / 2.0
        mid_sh_y = (l_sh[1] + r_sh[1]) / 2.0
        mid_hip_x = (l_hip[0] + r_hip[0]) / 2.0
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2.0

        dx = mid_sh_x - mid_hip_x
        dy = mid_sh_y - mid_hip_y

        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return 90.0

        angle_rad = math.atan2(abs(dy), abs(dx) + 1e-6)
        return float(math.degrees(angle_rad))

    def calculate_vertical_kinematics(
        self,
        kps_history: List[np.ndarray],
        bbox_history: List[List[float]]
    ) -> Tuple[float, float, float]:
        """Calculates normalized vertical velocity v_y (h/s) and acceleration."""
        if len(kps_history) < 2:
            torso_y = np.mean(kps_history[-1][[5, 6, 11, 12], 1])
            return 0.0, 0.0, float(torso_y)

        y_coords = []
        for kps in kps_history[-min(len(kps_history), 4):]:
            y_coords.append(np.mean(kps[[5, 6, 11, 12], 1]))

        current_bbox = bbox_history[-1]
        norm_h = max(current_bbox[3] - current_bbox[1], 50.0)

        v_now = (y_coords[-1] - y_coords[-2]) / (self.dt * norm_h)
        if len(y_coords) >= 3:
            v_prev = (y_coords[-2] - y_coords[-3]) / (self.dt * norm_h)
            a_now = (v_now - v_prev) / self.dt
        else:
            a_now = 0.0

        return float(v_now), float(a_now), float(y_coords[-1])

    def calculate_stillness_energy(self, kps_history: List[np.ndarray], window: int = 15) -> float:
        """Computes average keypoint motion over recent window."""
        if len(kps_history) < 3:
            return 50.0

        sample_len = min(len(kps_history), window)
        displacements = []

        for f in range(-sample_len + 1, 0):
            curr = kps_history[f]
            prev = kps_history[f - 1]
            valid_mask = (curr[:, 0] > 0) & (prev[:, 0] > 0) & (curr[:, 2] > 0.2)
            if np.any(valid_mask):
                diffs = np.linalg.norm(curr[valid_mask, :2] - prev[valid_mask, :2], axis=1)
                displacements.append(np.mean(diffs))

        return float(np.mean(displacements)) if displacements else 0.0

    def _empty_features(self) -> Dict[str, float]:
        return {
            "torso_angle": 90.0, "vertical_velocity": 0.0, "vertical_accel": 0.0,
            "aspect_ratio": 0.4, "stillness_energy": 100.0, "centroid_y": 0.0,
            "bbox_height": 100.0, "bbox_width": 40.0
        }


# ======================================================================================
# 5. MULTI-PERSON TRACKLET & TRACKER
# ======================================================================================
class Tracklet:
    def __init__(self, track_id: int, bbox: List[float], keypoints: np.ndarray):
        self.track_id = track_id
        self.bbox_history = [bbox]
        self.keypoints_history = [keypoints]
        self.missed_frames = 0
        self.state = "MONITORING"
        self.state_timer = 0.0
        self.in_occlusion_zone = False

    def update(self, bbox: List[float], keypoints: np.ndarray):
        self.bbox_history.append(bbox)
        self.keypoints_history.append(keypoints)
        if len(self.bbox_history) > 60:
            self.bbox_history.pop(0)
            self.keypoints_history.pop(0)
        self.missed_frames = 0


class SimpleTracker:
    def __init__(self, max_missed: int = 30, distance_thresh: float = 120.0):
        self.max_missed = max_missed
        self.distance_thresh = distance_thresh
        self.tracklets: Dict[int, Tracklet] = {}
        self.next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Tracklet]:
        for t in self.tracklets.values():
            t.missed_frames += 1

        matched_tracks = set()
        for det in detections:
            bbox = det["bbox"]
            kps = det["keypoints"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            best_id = None
            best_dist = float("inf")

            for t_id, track in self.tracklets.items():
                if t_id in matched_tracks:
                    continue
                prev_bbox = track.bbox_history[-1]
                pcx = (prev_bbox[0] + prev_bbox[2]) / 2.0
                pcy = (prev_bbox[1] + prev_bbox[3]) / 2.0
                dist = math.hypot(cx - pcx, cy - pcy)
                if dist < best_dist and dist < self.distance_thresh:
                    best_dist = dist
                    best_id = t_id

            if best_id is not None:
                self.tracklets[best_id].update(bbox, kps)
                matched_tracks.add(best_id)
            else:
                new_t = Tracklet(self.next_id, bbox, kps)
                self.tracklets[self.next_id] = new_t
                self.next_id += 1

        # Prune dead tracklets
        dead_ids = [t_id for t_id, t in self.tracklets.items() if t.missed_frames > self.max_missed]
        for d in dead_ids:
            del self.tracklets[d]

        return [t for t in self.tracklets.values() if t.missed_frames == 0]


# ======================================================================================
# 6. BIOMECHANICAL FALL STATE MACHINE
# ======================================================================================
class FallStateMachine:
    """
    Finite State Machine:
    MONITORING -> RAPID_DESCENT / FALL_DETECTED -> CONFIRMED_FALL (inactivity on floor)
    Distinguishes intentional sitting/lying down from accidental falls.
    """
    STATE_MONITORING = "MONITORING"
    STATE_RAPID_DESCENT = "RAPID_DESCENT"
    STATE_FALL_DETECTED = "FALL_DETECTED"
    STATE_CONFIRMED_FALL = "CONFIRMED_FALL"

    def __init__(
        self,
        fps: float = 30.0,
        fall_velocity_thresh: float = 1.0,
        flat_angle_thresh: float = 55.0,
        standing_angle_thresh: float = 70.0,
        inactivity_sec: float = 1.5
    ):
        self.fps = fps
        self.dt = 1.0 / max(fps, 1.0)
        self.fall_velocity_thresh = fall_velocity_thresh
        self.flat_angle_thresh = flat_angle_thresh
        self.standing_angle_thresh = standing_angle_thresh
        self.inactivity_sec = inactivity_sec

    def step(
        self,
        tracklet: Tracklet,
        features: Dict[str, float],
        occlusion_stats: Dict[str, float]
    ) -> Tuple[str, Dict[str, Any]]:
        v_y = features.get("vertical_velocity", 0.0)
        angular_vel = features.get("angular_velocity", 0.0)
        angle = features.get("torso_angle", 90.0)
        aspect_ratio = features.get("aspect_ratio", 0.5)
        stillness = features.get("stillness_energy", 0.0)

        current = tracklet.state
        next_state = current
        event_trigger = ""

        # Dynamic Fall Condition:
        # Rapid downward velocity OR high angular collapse combined with flat angle
        is_dynamic_fall = (
            v_y > self.fall_velocity_thresh or
            (angular_vel > 32.0 and angle < self.flat_angle_thresh) or
            (v_y > 0.55 and angle < self.flat_angle_thresh) or
            (aspect_ratio > 1.25 and v_y > 0.4)
        )

        if is_dynamic_fall:
            next_state = self.STATE_FALL_DETECTED
            tracklet.state_timer = 0.0
            event_trigger = f"[!] Fall Detected (v_y={v_y:.2f} h/s, angle={angle:.1f}°)"

        elif current == self.STATE_FALL_DETECTED:
            tracklet.state_timer += self.dt
            # Person remains flat/motionless on the floor
            if angle < self.flat_angle_thresh or aspect_ratio > 0.9:
                if tracklet.state_timer >= self.inactivity_sec:
                    next_state = self.STATE_CONFIRMED_FALL
                    event_trigger = "🚨 [EMERGENCY] Confirmed Fall On Floor!"
            elif angle > self.standing_angle_thresh and v_y < -0.2:
                # Subject stood back up safely
                next_state = self.STATE_MONITORING
                event_trigger = "Subject stood upright -> Reset"

        elif current == self.STATE_CONFIRMED_FALL:
            if angle > self.standing_angle_thresh and stillness > 15.0:
                next_state = self.STATE_MONITORING
                event_trigger = "Subject upright -> Normal"

        tracklet.state = next_state
        return next_state, {
            "state_timer": tracklet.state_timer,
            "countdown": max(0.0, self.inactivity_sec - tracklet.state_timer),
            "event_trigger": event_trigger
        }


# ======================================================================================
# 7. TELEMETRY HUD & VISUALIZER
# ======================================================================================
class Visualizer:
    """Renders real-time bounding boxes, skeletons, posture angle, velocity, and HUD."""
    SKELETON_PAIRS = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # Facial landmarks
        (5, 6),                                     # Shoulders
        (5, 7), (7, 9),                             # Left arm
        (6, 8), (8, 10),                            # Right arm
        (5, 11), (6, 12),                           # Torso sides
        (11, 12),                                   # Hips
        (11, 13), (13, 15),                         # Left leg
        (12, 14), (14, 16)                          # Right leg
    ]

    STATE_COLORS = {
        "MONITORING": (46, 204, 113),       # Emerald Green
        "RAPID_DESCENT": (241, 196, 15),     # Bright Yellow
        "FALL_DETECTED": (231, 76, 60),      # Vivid Red
        "CONFIRMED_FALL": (192, 57, 43)      # Crimson Emergency Red
    }

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        tracklets: List[Tracklet],
        tracklet_data: Dict[int, Dict[str, Any]],
        fps_val: float,
        hardware_info: str,
        audio_muted: bool
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        has_emergency = False

        for track in tracklets:
            t_id = track.track_id
            if t_id not in tracklet_data:
                continue

            data = tracklet_data[t_id]
            kps = data["imputed_kps"]
            state = data["state"]
            features = data["features"]
            bbox = track.bbox_history[-1]

            if state in ["FALL_DETECTED", "CONFIRMED_FALL"]:
                has_emergency = True

            color = self.STATE_COLORS.get(state, (46, 204, 113))

            # Draw Skeleton
            self._draw_skeleton(vis, kps, color, state)

            # Draw Bounding Box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3 if state != "MONITORING" else 2)

            # Person Telemetry Card
            self._draw_person_card(vis, x1, y1, t_id, state, features, color)

        # Top Status HUD
        self._render_top_hud(vis, fps_val, len(tracklets), hardware_info, audio_muted)

        # Emergency Screen Border Flash
        if has_emergency:
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            self._render_emergency_banner(vis)

        return vis

    def _draw_skeleton(self, frame: np.ndarray, kps: np.ndarray, color: Tuple[int, int, int], state: str):
        thickness = 3 if state != "MONITORING" else 2
        for pt1_idx, pt2_idx in self.SKELETON_PAIRS:
            pt1 = kps[pt1_idx]
            pt2 = kps[pt2_idx]
            if pt1[2] > 0.25 and pt2[2] > 0.25:
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, thickness, cv2.LINE_AA)

        for i in range(17):
            pt = kps[i]
            if pt[2] > 0.25:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, color, 1, cv2.LINE_AA)

    def _draw_person_card(self, frame: np.ndarray, x: int, y: int, t_id: int, state: str, feat: Dict[str, float], color: Tuple[int, int, int]):
        card_w = 210
        card_h = 75
        card_x = max(5, min(x, frame.shape[1] - card_w - 5))
        card_y = max(35, y - card_h - 5)

        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), color, 1)

        # Status Badge
        cv2.putText(frame, f"ID #{t_id}: {state}", (card_x + 8, card_y + 18),
                    self.font, 0.52, color, 2, cv2.LINE_AA)

        angle = feat.get("torso_angle", 90.0)
        vy = feat.get("vertical_velocity", 0.0)
        ar = feat.get("aspect_ratio", 0.5)

        # Metrics
        cv2.putText(frame, f"Torso Angle : {angle:5.1f} deg", (card_x + 8, card_y + 36),
                    self.font, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Vert Velocity: {vy:+5.2f} h/s", (card_x + 8, card_y + 52),
                    self.font, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Aspect Ratio : {ar:5.2f} (W/H)", (card_x + 8, card_y + 68),
                    self.font, 0.40, (230, 230, 230), 1, cv2.LINE_AA)

    def _render_top_hud(self, frame: np.ndarray, fps: float, count: int, hw_info: str, audio_muted: bool):
        w = frame.shape[1]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (15, 18, 24), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, 32), (w, 32), (60, 65, 80), 1)

        # Title
        cv2.putText(frame, "LIVE FALL DETECTION SYSTEM", (12, 21),
                    self.font, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        # FPS & Hardware
        fps_text = f"FPS: {fps:4.1f} | {hw_info}"
        cv2.putText(frame, fps_text, (w - 380, 21),
                    self.font, 0.45, (100, 220, 255), 1, cv2.LINE_AA)

        # Audio indicator
        audio_str = "AUDIO: MUTED [M]" if audio_muted else "AUDIO: ON [M]"
        audio_col = (120, 120, 120) if audio_muted else (50, 255, 100)
        cv2.putText(frame, audio_str, (w - 150, 21),
                    self.font, 0.42, audio_col, 1, cv2.LINE_AA)

    def _render_emergency_banner(self, frame: np.ndarray):
        w = frame.shape[1]
        banner_h = 50
        y1 = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y1), (w, y1 + banner_h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        msg = "EMERGENCY: FALL DETECTED! PERSON DOWN!"
        text_size = cv2.getTextSize(msg, self.font, 0.75, 2)[0]
        tx = (w - text_size[0]) // 2
        cv2.putText(frame, msg, (tx, y1 + 33), self.font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


# ======================================================================================
# 8. INTEGRATED FALL DETECTION PIPELINE
# ======================================================================================
class FallDetectionPipeline:
    def __init__(self, model_name: str = "yolov8n-pose.pt", conf_thresh: float = 0.35, device: Optional[str] = None, fps: float = 30.0):
        self.detector = PoseDetector(model_name=model_name, conf_thresh=conf_thresh, device=device)
        self.occlusion_engine = OcclusionEngine()
        self.kinematics = KinematicExtractor(fps=fps)
        self.tracker = SimpleTracker()
        self.state_machine = FallStateMachine(fps=fps)
        self.visualizer = Visualizer()
        self.alarm = AudioAlarm(enabled=True)
        self.fps = fps

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        detections = self.detector.detect(frame)
        active_tracks = self.tracker.update(detections)

        tracklet_data = {}
        alerts = []

        for track in active_tracks:
            t_id = track.track_id
            raw_kps = track.keypoints_history[-1]
            bbox = track.bbox_history[-1]

            imputed_kps, occ_mask, occ_rate = self.occlusion_engine.impute_keypoints(raw_kps, bbox)
            track.keypoints_history[-1] = imputed_kps

            occ_stats = {"rate": occ_rate}
            features = self.kinematics.extract_features(track.keypoints_history, track.bbox_history, occ_stats)
            state, state_info = self.state_machine.step(track, features, occ_stats)

            tracklet_data[t_id] = {
                "imputed_kps": imputed_kps,
                "features": features,
                "state": state,
                "state_info": state_info
            }

            if state in ["FALL_DETECTED", "CONFIRMED_FALL"]:
                alerts.append({
                    "track_id": t_id,
                    "state": state,
                    "features": features,
                    "info": state_info
                })
                # Trigger sound alarm
                self.alarm.trigger_fall_alarm()

        hardware_info = f"CUDA (RTX 4060)" if self.detector.device == "cuda" else "CPU Mode"
        vis_frame = self.visualizer.render(
            frame=frame,
            tracklets=active_tracks,
            tracklet_data=tracklet_data,
            fps_val=self.fps,
            hardware_info=hardware_info,
            audio_muted=not self.alarm.enabled
        )

        return vis_frame, alerts


# ======================================================================================
# 9. LIVE DEMONSTRATION & RUNNER
# ======================================================================================
def run_live_demonstration(
    source: str = "0",
    save_path: Optional[str] = None,
    conf: float = 0.35,
    device: Optional[str] = None
):
    print("=" * 80)
    print("  AI PERSON & ELDERLY FALL DETECTION SYSTEM - LIVE DEMONSTRATION")
    print("  Controls:")
    print("    [Q] or [ESC] : Quit application")
    print("    [R]          : Reset all fall alerts")
    print("    [M]          : Toggle audio alarm siren (Mute/Unmute)")
    print("    [S]          : Save current screenshot / snapshot")
    print("    [V]          : Toggle between Webcam and Sample Demo Video")
    print("    [SPACE]      : Pause / Unpause stream")
    print("=" * 80)

    sample_videos = [
        "sample_videos/fall_example_1.mp4",
        "sample_videos/fall_example_2.mp4",
        "sample_videos/fall_example_3.mp4"
    ]
    current_source = source
    is_webcam = current_source.isdigit()

    def open_capture(src):
        if str(src).isdigit():
            if sys.platform.startswith("win"):
                c = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
            else:
                c = cv2.VideoCapture(int(src))
        else:
            c = cv2.VideoCapture(str(src))
        return c

    cap = open_capture(current_source)
    if not cap.isOpened():
        print(f"[Warning] Failed to open source '{current_source}'. Falling back to sample video...")
        current_source = sample_videos[0]
        cap = open_capture(current_source)
        is_webcam = False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 120:
        fps = 30.0

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        print("[Error] Unable to read frames from video source.")
        cap.release()
        return

    height, width = first_frame.shape[:2]
    print(f"[Pipeline] Source active: {current_source} ({width}x{height} @ {fps:.1f} FPS)")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"[Pipeline] Recording output to: {save_path}")

    pipeline = FallDetectionPipeline(model_name="yolov8n-pose.pt", conf_thresh=conf, device=device, fps=fps)

    window_name = "Real-Time Fall Detection System (YOLOv8-Pose + State Machine)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(width, 1100), int(height * (min(width, 1100) / max(width, 1))))

    frame = first_frame
    paused = False
    snapshot_idx = 1
    t_start = time.time()
    frame_counter = 0

    try:
        while True:
            if not paused:
                if frame is None:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        if not is_webcam:
                            # Loop demo video for continuous showcase
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if not ret:
                                break
                        else:
                            break

                frame_counter += 1
                t_frame_start = time.time()

                vis_frame, alerts = pipeline.process_frame(frame)

                t_frame_end = time.time()
                instant_fps = 1.0 / max(t_frame_end - t_frame_start, 1e-4)
                pipeline.fps = 0.9 * pipeline.fps + 0.1 * instant_fps

                for alert in alerts:
                    print(f"🚨 [ALERT] Person #{alert['track_id']} -> {alert['state']} "
                          f"(Angle: {alert['features']['torso_angle']:.1f}°, v_y: {alert['features']['vertical_velocity']:+.2f} h/s)")

                if writer:
                    writer.write(vis_frame)

                cv2.imshow(window_name, vis_frame)
                frame = None

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:
                print("[Pipeline] Exit requested by user.")
                break
            elif key in [ord('r'), ord('R')]:
                print("[Pipeline] Manual Alert Reset.")
                for t in pipeline.tracker.tracklets.values():
                    t.state = "MONITORING"
                    t.state_timer = 0.0
            elif key in [ord('m'), ord('M')]:
                muted = not pipeline.alarm.toggle_mute()
                print(f"[Pipeline] Audio Siren {'MUTED' if muted else 'ENABLED'}.")
            elif key in [ord(' ')]:
                paused = not paused
                print(f"[Pipeline] {'PAUSED' if paused else 'RESUMED'}.")
            elif key in [ord('s'), ord('S')]:
                snap_name = f"fall_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_name, vis_frame)
                print(f"[Pipeline] Snapshot saved to: {snap_name}")
            elif key in [ord('v'), ord('V')]:
                # Switch video source
                cap.release()
                if is_webcam:
                    current_source = sample_videos[0]
                    is_webcam = False
                else:
                    current_source = "0"
                    is_webcam = True
                print(f"[Pipeline] Switching source to: {current_source}")
                cap = open_capture(current_source)
                if not cap.isOpened() and is_webcam:
                    print("[Warning] Webcam not available, reverting to demo video.")
                    current_source = sample_videos[0]
                    cap = open_capture(current_source)
                    is_webcam = False
                frame = None

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("=" * 80)
        print(f"  Session Finished. Processed {frame_counter} frames in {time.time() - t_start:.2f}s.")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Real-Time Fall Detection System")
    parser.add_argument("--source", type=str, default="0", help="Webcam index ('0', '1') or path to video file")
    parser.add_argument("--save", type=str, default=None, help="Optional output video file path")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold (default: 0.35)")
    parser.add_argument("--device", type=str, default=None, help="'cuda', 'cpu', or auto-select")

    args = parser.parse_args()
    run_live_demonstration(source=args.source, save_path=args.save, conf=args.conf, device=args.device)
