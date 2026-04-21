import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json

st.title("Sign Language Recognition System")
st.write("Show your hand gesture in front of camera")

# Load model
model_data = joblib.load("gesture_model.pkl")
model = model_data["model"]
label_encoder = model_data["label_encoder"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

run = st.checkbox("Start Camera")
frame_window = st.image([])

cap = cv2.VideoCapture(0)

def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

def normalize_landmarks(coords):
    coords = np.array(coords).reshape(21, 3)
    wrist = coords[0]
    coords -= wrist
    scale = np.max(np.abs(coords)) + 1e-6
    coords /= scale
    return coords.flatten()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1
) as hands:

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Camera error")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                raw = extract_landmarks(hand)
                norm = normalize_landmarks(raw)

                pred = model.predict([norm])[0]
                gesture = label_encoder.inverse_transform([pred])[0]

                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        frame_window.image(frame, channels="BGR")
