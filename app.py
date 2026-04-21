import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp

st.set_page_config(page_title="Sign Language Recognition", layout="centered")

st.title("🤟 Sign Language Recognition System")

# Load trained model
model_data = joblib.load("gesture_model.pkl")
model = model_data["model"]
label_encoder = model_data["label_encoder"]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Function to extract landmarks
def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

# Normalize landmarks
def normalize_landmarks(coords):
    coords = np.array(coords).reshape(21, 3)
    wrist = coords[0]
    coords -= wrist
    scale = np.max(np.abs(coords)) + 1e-6
    coords /= scale
    return coords.flatten()

# Camera input from browser
img_file = st.camera_input("📷 Show your hand gesture")

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                raw = extract_landmarks(hand_landmarks)
                norm = normalize_landmarks(raw)

                prediction = model.predict([norm])[0]
                gesture = label_encoder.inverse_transform([prediction])[0]

                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                st.success(f"Predicted Gesture: {gesture}")

        else:
            st.warning("No hand detected")

    st.image(frame, channels="BGR")