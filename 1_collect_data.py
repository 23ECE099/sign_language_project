"""
STEP 1: Collect custom gesture data
------------------------------------
Controls:
  Press a number key (0-9) or letter key to set the gesture label
  Press SPACE to capture a sample
  Press 'q' to quit
  Press 's' to save collected data

Each sample = 63 floats (21 landmarks × x,y,z)
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
GESTURES = {
    '0': 'thumbs_up',
    '1': 'peace',
    '2': 'fist',
    '3': 'open_hand',
    '4': 'point',
    # Add your own custom gestures here
}
SAMPLES_PER_GESTURE_TARGET = 200   # Aim for 200 samples per gesture
DATA_FILE = 'gesture_data.json'
# ────────────────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

def extract_landmarks(hand_landmarks):
    """Flatten 21 landmarks into 63 normalized values (x, y, z)."""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

def normalize_landmarks(coords):
    """
    Normalize relative to wrist (landmark 0).
    Makes the model translation-invariant.
    """
    coords = np.array(coords).reshape(21, 3)
    wrist   = coords[0]
    coords -= wrist                          # center on wrist
    scale   = np.max(np.abs(coords)) + 1e-6 # scale to [-1, 1]
    coords /= scale
    return coords.flatten().tolist()

def load_existing_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {'samples': [], 'labels': [], 'gesture_map': GESTURES}

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)
    counts = {}
    for lbl in data['labels']:
        counts[lbl] = counts.get(lbl, 0) + 1
    print(f"\n✅ Saved {len(data['samples'])} total samples:")
    for g, c in counts.items():
        print(f"   {g}: {c} samples")

def main():
    data          = load_existing_data()
    current_label = None
    cap           = cv2.VideoCapture(0)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # ── Draw landmarks ──────────────────────────────────────────────
            landmarks_detected = False
            if result.multi_hand_landmarks:
                for hlm in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
                    landmarks_detected = True

            # ── HUD ─────────────────────────────────────────────────────────
            counts = {}
            for lbl in data['labels']:
                counts[lbl] = counts.get(lbl, 0) + 1

            y = 30
            for key, name in GESTURES.items():
                cnt   = counts.get(name, 0)
                color = (0, 255, 100) if name == current_label else (180, 180, 180)
                cv2.putText(frame, f"[{key}] {name}: {cnt}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                y += 24

            status_color = (0, 220, 0) if landmarks_detected else (0, 80, 200)
            status_text  = "Hand detected" if landmarks_detected else "No hand"
            cv2.putText(frame, status_text, (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            if current_label:
                cv2.putText(frame, f"Label: {current_label}  |  SPACE=capture  S=save  Q=quit",
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                cv2.putText(frame, "Press 0-4 to select gesture",
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

            cv2.imshow('Gesture Data Collector', frame)
            key = cv2.waitKey(1) & 0xFF

            # ── Key handling ────────────────────────────────────────────────
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_data(data)
            elif chr(key) in GESTURES:
                current_label = GESTURES[chr(key)]
                print(f"Selected: {current_label}")
            elif key == ord(' '):
                if current_label and result.multi_hand_landmarks:
                    raw  = extract_landmarks(result.multi_hand_landmarks[0])
                    norm = normalize_landmarks(raw)
                    data['samples'].append(norm)
                    data['labels'].append(current_label)
                    cnt = counts.get(current_label, 0) + 1
                    print(f"  Captured {current_label} [{cnt}/{SAMPLES_PER_GESTURE_TARGET}]")
                elif not current_label:
                    print("  ⚠ Select a gesture first (press 0-4)")
                else:
                    print("  ⚠ No hand detected")

    cap.release()
    cv2.destroyAllWindows()

    if data['samples']:
        save_data(data)

if __name__ == '__main__':
    main()
