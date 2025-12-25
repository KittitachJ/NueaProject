import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Pose Detection App",
    layout="centered"
)

st.title("🧍‍♂️ MediaPipe Pose + Streamlit")
st.write("ทดสอบ MediaPipe Pose บน Streamlit Cloud")

# =========================
# Load ML model (ถ้ามี)
# =========================
@st.cache_resource
def load_model():
    try:
        return joblib.load("volleyball_model.pkl")
    except Exception:
        return None

model = load_model()

# =========================
# MediaPipe setup (Cloud-safe)
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =========================
# Video Processor
# =========================
class PoseProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pose detection
        results = pose.process(img_rgb)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

            # Example: extract landmarks for ML
            if model is not None:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                X = np.array(landmarks).reshape(1, -1)

                try:
                    pred = model.predict(X)[0]
                    cv2.putText(
                        img,
                        f"Prediction: {pred}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                    )
                except Exception:
                    pass

        return img


# =========================
# WebRTC
# =========================
webrtc_streamer(
    key="pose-detection",
    video_processor_factory=PoseProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)

st.markdown("---")
st.caption("Powered by MediaPipe & Streamlit")
