import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import time
import collections
import os
import csv
from datetime import datetime

# --- ASSET LOADING ---
def load_assets():
    files = ['first.png', 'second.png', 'third.png']
    imgs = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # Resize for web visibility
            img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_NEAREST)
            imgs.append(img)
    return imgs

ASSETS = load_assets()

class TaiyakiGuardTransformer(VideoTransformerBase):
    def __init__(self, picker_type="skin picking", tracking_enabled=False, user_id=""):
        # Initialize MediaPipe
        self.mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5)

        # Config
        self.picker_type = picker_type
        self.tracking_enabled = tracking_enabled
        self.user_id = user_id or "anonymous"

        # State tracking
        self.finger_history = collections.deque(maxlen=180)
        self.continuous_touch_start = None
        self.alert_until = 0
        self.TIP_IDS = [4, 8, 12, 16, 20] # Fingertips
        self.last_logged_alert = 0
        self.data_dir = "data"
        self.log_file = os.path.join(self.data_dir, "picking_log.csv")
        if self.tracking_enabled and not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir, exist_ok=True)
            except Exception:
                pass

    def is_active_motion(self, history, current_time):
        recent = [e for e in history if current_time - e[0] <= 6.0]
        if len(recent) < 16: return False
        xs = np.array([p[1] for p in recent])
        ys = np.array([p[2] for p in recent])
        if np.ptp(xs) < 80 and np.ptp(ys) < 80: return False
        dx, dy = np.diff(xs), np.diff(ys)
        x_dir = np.sign(dx) * (np.abs(dx) > 10)
        x_changes = np.sum((x_dir[1:] != x_dir[:-1]) & (x_dir[1:] != 0) & (x_dir[:-1] != 0))
        return x_changes >= 3

    def draw_hud(self, frame, duration, motion, alerting):
        ih, iw, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, ih - 100), (iw, ih), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Colors
        SEAFOAM = (180, 255, 180); CORAL = (128, 128, 255); WHITE = (240, 240, 240)
        
        # UI Text
        cv2.putText(frame, "CONTACT:", (30, ih - 65), cv2.FONT_HERSHEY_DUPLEX, 0.5, WHITE, 1)
        cv2.putText(frame, f"{duration:.1f}s", (30, ih - 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, CORAL if duration > 3 else SEAFOAM, 2)
        
        status_text = "PICKING DETECTED" if motion else "IDLE"
        cv2.putText(frame, status_text, (200, ih - 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, CORAL if motion else SEAFOAM, 2)

        if alerting:
            cv2.circle(frame, (iw - 40, ih - 50), 12, CORAL, -1)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        current_time = time.time()
        
        # Detection
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_res = self.mp_face.process(rgb_img)
        hand_res = self.mp_hands.process(rgb_img)
        
        face_box = None
        if face_res.detections:
            b = face_res.detections[0].location_data.relative_bounding_box
            face_box = (int(b.xmin * w), int(b.ymin * h), int(b.width * w), int(b.height * h))

        touching = False
        finger_pos = []
        if hand_res.multi_hand_landmarks and face_box:
            fx, fy, fw, fh = face_box
            for hand in hand_res.multi_hand_landmarks:
                for tid in self.TIP_IDS:
                    tip = hand.landmark[tid]
                    tx, ty = int(tip.x * w), int(tip.y * h)
                    finger_pos.append((tx, ty))
                    if fx <= tx <= fx+fw and fy <= ty <= fy+fh: touching = True

        # Logic
        if touching:
            if self.continuous_touch_start is None: self.continuous_touch_start = current_time
            avg_x = int(np.mean([p[0] for p in finger_pos]))
            avg_y = int(np.mean([p[1] for p in finger_pos]))
            self.finger_history.append((current_time, avg_x, avg_y))
        else:
            self.continuous_touch_start = None
            self.finger_history.clear()

        motion = self.is_active_motion(self.finger_history, current_time)
        duration = current_time - self.continuous_touch_start if self.continuous_touch_start else 0
        
        if duration >= 5.0 and motion:
            # Trigger alert
            if current_time >= self.alert_until:
                # only log when new alert occurs
                if self.tracking_enabled:
                    try:
                        self.log_event(event_type="alert", duration=duration, motion=motion)
                    except Exception:
                        pass
            self.alert_until = current_time + 3.0

        # Rendering
        alerting = current_time < self.alert_until
        self.draw_hud(img, duration, motion, alerting)
        
        if alerting and ASSETS:
            # Slower cycle (0.7 multiplier)
            idx = int((current_time * 0.7) % len(ASSETS))
            overlay = ASSETS[idx]
            oh, ow = overlay.shape[:2]
            # Overlay logic for web (top right)
            x_off, y_off = w - ow - 20, 20
            if overlay.shape[2] == 4:
                alpha = overlay[:,:,3] / 255.0
                for c in range(3):
                    img[y_off:y_off+oh, x_off:x_off+ow, c] = (1.0 - alpha) * img[y_off:y_off+oh, x_off:x_off+ow, c] + alpha * overlay[:,:,c]
        
        return img

    def log_event(self, event_type, duration, motion):
        # Append a CSV row with timestamp, picker type, user id, event, duration, motion
        if not self.tracking_enabled:
            return
        header = ["timestamp", "user_id", "picker_type", "event", "duration_s", "motion"]
        row = [datetime.utcnow().isoformat(), self.user_id, self.picker_type, event_type, f"{duration:.2f}", str(bool(motion))]
        write_header = not os.path.exists(self.log_file)
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="Taiyaki Guard", page_icon="🐟")
st.title("🐟 Taiyaki Guard")
st.markdown("### Protect your skin with computer vision.")

# --- UI: Picker type + Tracking consent ---
picker_options = ["skin picking", "nail biting", "hair picking", "lip picking", "other"]
picker_choice = st.sidebar.selectbox("Choose picking type:", picker_options)
track_choice = st.sidebar.radio("Allow anonymous tracking to understand your picking?", ("No", "Yes"))
tracking_enabled = True if track_choice == "Yes" else False
user_id = st.sidebar.text_input("Optional user id (or leave blank for anonymous)")

if tracking_enabled:
    st.sidebar.success("Tracking enabled — events will be saved locally to data/picking_log.csv")
else:
    st.sidebar.info("Tracking disabled — no data will be saved")

# Start camera with transformer configured by UI
webrtc_streamer(key="taiyaki", video_transformer_factory=lambda: TaiyakiGuardTransformer(picker_type=picker_choice, tracking_enabled=tracking_enabled, user_id=user_id))

st.sidebar.info("This app monitors for repetitive face-touching. If detected for 5 seconds, Taiyaki will intervene!")
