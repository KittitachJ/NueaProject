import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import joblib
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# โหลดโมเดล
model = joblib.load("volleyball_model.pkl")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# UI
st.title("🏐 Volleyball Pose Detection (Realtime Camera Switch)")

# เลือกกล้อง (frontend control)
camera_facing = st.radio("เลือกกล้อง", ("กล้องหน้า", "กล้องหลัง"))

facing_mode = "user" if camera_facing == "กล้องหน้า" else "environment"

class PoseDetector(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])

            columns = [f"{axis}{i}" for i in range(33) for axis in ["x","y"]]
            X_new = pd.DataFrame([keypoints], columns=columns)

            prediction = model.predict(X_new)[0]
            text = "Yes" if prediction == "under_correct" else "No"
            color = (0, 255, 0) if prediction == "under_correct" else (0, 0, 255)

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(img, text, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        return img

# ใช้ key ที่เปลี่ยนตามค่า facing_mode
webrtc_streamer(
    key=f"volleyball-{facing_mode}",  
    video_transformer_factory=PoseDetector,
    media_stream_constraints={
        "video": {"facingMode": facing_mode},
        "audio": False
    }
)



# streamlit run app.py
