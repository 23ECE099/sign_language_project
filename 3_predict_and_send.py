"""
STEP 3: Real-time prediction + send to ESP32 via Serial
---------------------------------------------------------
- Loads trained model
- Captures webcam frames
- Extracts MediaPipe landmarks
- Predicts gesture
- Sends predicted letter/label to ESP32 over USB serial
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import joblib
import serial
import serial.tools.list_ports
import time
import sys


# ── Config ──────────────────────────────────────────────────────────────────
MODEL_FILE     = 'gesture_model.pkl'
LABELS_FILE    = 'label_map.json'
SERIAL_BAUD    = 115200
CONFIDENCE_THR = 0.2    # Only send prediction if confidence ≥ this
SEND_INTERVAL  = 0.2     # Seconds between serial sends (avoid flooding)
# ────────────────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


def find_esp32_port():
    """Auto-detect ESP32 serial port."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or '').lower()
        if any(k in desc for k in ['cp210', 'ch340', 'esp', 'uart', 'usb serial']):
            print(f"   Auto-detected ESP32 on {p.device} ({p.description})")
            return p.device
    # Fallback — list all ports and let user pick
    print("Available serial ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} — {p.description}")
    if not ports:
        print("  ⚠ No serial ports found. Running without serial.")
        return None
    idx = input("Enter port number (or ENTER to skip serial): ").strip()
    if idx == '':
        return None
    return ports[int(idx)].device


def load_model():
    bundle = joblib.load(MODEL_FILE)
    model  = bundle['model']
    le     = bundle['label_encoder']
    with open(LABELS_FILE) as f:
        label_map = json.load(f)
    return model, le, label_map


def extract_and_normalize(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    coords = np.array(coords).reshape(21, 3)
    wrist   = coords[0]
    coords -= wrist
    scale   = np.max(np.abs(coords)) + 1e-6
    coords /= scale
    return coords.flatten()


def draw_hud(frame, label, confidence, serial_ok):
    h, w = frame.shape[:2]

    # Prediction box
    bar_w = int(confidence * 300)
    cv2.rectangle(frame, (10, h - 100), (310, h - 60), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, h - 100), (10 + bar_w, h - 60),
                  (0, 220, 100) if confidence >= 0.85 else (0, 140, 220), -1)
    cv2.putText(frame, f"{label}  {confidence*100:.0f}%",
                (16, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Serial status
    color = (0, 200, 80) if serial_ok else (60, 60, 200)
    cv2.putText(frame, "SERIAL OK" if serial_ok else "NO SERIAL",
                (w - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(frame, "Q = quit", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def main():
    print("── Loading model ────────────────────────────────")
    model, le, label_map = load_model()
    print(f"   Gestures: {list(label_map.values())}")

    print("\n── Connecting to ESP32 ──────────────────────────")
    port = find_esp32_port()
    ser  = None
    if port:
        try:
            ser = serial.Serial(port, SERIAL_BAUD, timeout=1)
            time.sleep(2)   # Allow ESP32 to reset
            print(f"   Connected on {port} @ {SERIAL_BAUD} baud")
        except Exception as e:
            print(f"   ⚠ Serial error: {e}  — continuing without serial")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_send  = 0
    last_label = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            predicted_label = "—"
            confidence      = 0.0

            if result.multi_hand_landmarks:
                hlm  = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

                features = extract_and_normalize(hlm).reshape(1, -1)
                probs    = model.predict_proba(features)[0]
                idx      = np.argmax(probs)
                confidence      = probs[idx]
                predicted_label = le.inverse_transform([idx])[0]

                # Send over serial if confident & interval elapsed
                now = time.time()
                if (confidence >= CONFIDENCE_THR
                        and (now - last_send) >= SEND_INTERVAL
                        and predicted_label != last_label):

                    short = predicted_label[:8].upper()   # Max 8 chars for OLED
                    if ser:
                        try:
                            print(">>> SENT:", short, "Confidence:", confidence)
                            ser.write((short + '\n').encode())
                            print(f"  → Sent: {short}  ({confidence*100:.0f}%)")
                        except Exception as e:
                            print(f"  ⚠ Serial write error: {e}")

                    last_send  = now
                    last_label = predicted_label

            draw_hud(frame, predicted_label, confidence, ser is not None and ser.is_open)
            cv2.imshow('Sign Language — Real-time', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()
