import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- MediaPipe (Cloud-safe import) ---
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing


# -------------------------------
# Load ML model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("volleyball_model.pkl")


model = load_model()


# -------------------------------
# Video Processor
# -------------------------------
class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # --- Extract landmarks ---
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

            # --- Predict (if feature size matches) ---
            try:
                X = np.array(landmarks).reshape(1, -1)
                pred = model.predict(X)[0]

                cv2.putText(
                    img,
                    f"Action: {pred}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            except Exception:
                pass

        return img


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Volleyball Pose Detection", layout="centered")

st.title("🏐 Volleyball Pose Detection")
st.write("ใช้ MediaPipe + Machine Learning ตรวจจับท่าทางจากกล้อง")

webrtc_streamer(
    key="pose-detection",
    video_processor_factory=PoseVideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
