import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import joblib
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

model = joblib.load("volleyball_model.pkl")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.title("🏐 Volleyball Pose Detection")

class PoseDetector(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])

            columns = [f"{axis}{i}" for i in range(33) for axis in ["x", "y"]]
            X_new = pd.DataFrame([keypoints], columns=columns)

            prediction = model.predict(X_new)[0]
            text = "Yes" if prediction == "under_correct" else "No"
            color = (0, 255, 0) if prediction == "under_correct" else (0, 0, 255)

            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            cv2.putText(
                img, text, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="volleyball",
    video_processor_factory=PoseDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]
    }
)


