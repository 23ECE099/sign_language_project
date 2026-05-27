# Sign Language Recognition System using MediaPipe, Machine Learning & ESP32

A real-time **Hand Gesture / Sign Language Recognition System** built using:

* Python
* MediaPipe Hands
* OpenCV
* Scikit-learn
* Streamlit
* ESP32 Serial Communication

The system can:

✅ Collect custom hand gesture datasets
✅ Train a machine learning model
✅ Predict gestures in real-time using webcam
✅ Send predictions to ESP32 via Serial Communication
✅ Run as a web app using Streamlit

---

# 📌 Features

* Real-time hand tracking using **MediaPipe**
* Custom gesture dataset collection
* Landmark normalization for better accuracy
* Machine Learning classifier:

  * MLP Neural Network
  * Random Forest
* Live gesture prediction
* ESP32 integration through USB serial
* Streamlit GUI support
* CPU-friendly and lightweight

---

# 🛠 Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Main programming language |
| OpenCV       | Webcam & image processing |
| MediaPipe    | Hand landmark detection   |
| NumPy        | Numerical operations      |
| Scikit-learn | Machine learning          |
| Streamlit    | Web interface             |
| PySerial     | ESP32 communication       |
| Joblib       | Model saving/loading      |

---

# 📂 Project Structure

```bash
Sign-Language-Recognition/
│
├── collect_data.py          # Step 1 - Dataset collection
├── train_model.py           # Step 2 - Model training
├── realtime_predict.py      # Step 3 - Real-time prediction + ESP32
├── app.py                   # Streamlit web app
│
├── gesture_data.json        # Saved training samples
├── gesture_model.pkl        # Trained ML model
├── label_map.json           # Gesture labels
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/sign-language-recognition.git

cd sign-language-recognition
```

---

## 2️⃣ Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Required Packages

Create a `requirements.txt` file:

```txt
opencv-python
mediapipe
numpy
scikit-learn
joblib
pyserial
streamlit
```

---

# ✋ STEP 1 — Collect Gesture Data

Run:

```bash
python collect_data.py
```

---

## 🎮 Controls

| Key   | Action               |
| ----- | -------------------- |
| 0-9   | Select gesture label |
| SPACE | Capture sample       |
| S     | Save dataset         |
| Q     | Quit                 |

---

## 📌 Current Gestures

```python
GESTURES = {
    '0': 'thumbs_up',
    '1': 'peace',
    '2': 'fist',
    '3': 'open_hand',
    '4': 'point',
}
```

You can add your own gestures.

---

## 📊 Data Format

Each sample contains:

* 21 hand landmarks
* x, y, z coordinates

Total:

```text
21 × 3 = 63 values
```

Landmarks are normalized relative to wrist position.

---

# 🧠 STEP 2 — Train Model

Run:

```bash
python train_model.py
```

---

## 🔍 Training Pipeline

### 1. Load Dataset

Reads:

```text
gesture_data.json
```

---

### 2. Data Augmentation

Adds Gaussian noise for robustness.

---

### 3. Label Encoding

Converts gesture names into numeric labels.

Example:

```json
{
  "0": "fist",
  "1": "open_hand",
  "2": "peace"
}
```

---

### 4. Model Training

Supported models:

## ✅ MLP Neural Network

```python
hidden_layer_sizes=(256,128,64)
```

OR

## ✅ Random Forest

```python
n_estimators=200
```

---

### 5. Evaluation

Outputs:

* Accuracy
* Classification report
* Confusion matrix

---

## 💾 Saved Files

| File              | Description    |
| ----------------- | -------------- |
| gesture_model.pkl | Trained model  |
| label_map.json    | Gesture labels |

---

# 🎥 STEP 3 — Real-Time Prediction

Run:

```bash
python realtime_predict.py
```

---

## ✨ Features

* Live webcam prediction
* Confidence score
* Serial communication with ESP32
* Auto ESP32 port detection

---

## 🔌 ESP32 Serial Communication

Predicted gesture is sent through USB serial:

```text
THUMBSUP
PEACE
FIST
```

---

## ⚡ Serial Settings

```python
SERIAL_BAUD = 115200
```

---

## 📡 Prediction Logic

Gesture is sent only when:

* Confidence ≥ threshold
* Prediction changes
* Send interval elapsed

---

# 🌐 Streamlit Web Application

Run:

```bash
streamlit run app.py
```

---

## 🖥 Features

* Live webcam feed
* Real-time prediction
* Simple browser interface

---

# 🧮 Landmark Normalization

The system normalizes landmarks by:

1. Centering relative to wrist
2. Scaling coordinates

This improves:

✅ Translation invariance
✅ Scale invariance
✅ Robustness

---

# 📈 Accuracy Tips

For better accuracy:

* Collect 150–300 samples per gesture
* Use different lighting conditions
* Capture multiple hand angles
* Avoid background clutter

---

# 🔄 Future Improvements

* Sentence generation
* Full ASL alphabet recognition
* Deep learning with TensorFlow
* Mobile deployment
* Voice output
* Multi-hand recognition
* Gesture smoothing

---

# 🤖 ESP32 Integration Ideas

You can connect this system with:

* OLED display
* Smart home controls
* Robot arm
* Wheelchair control
* IoT automation
* Speech synthesizer

---

# 📷 Example Workflow

```text
Hand Gesture
      ↓
MediaPipe Detection
      ↓
Landmark Extraction
      ↓
Normalization
      ↓
ML Model Prediction
      ↓
Display / ESP32 Output
```

---

# 🚀 Performance

| Component | Performance              |
| --------- | ------------------------ |
| MediaPipe | Real-time                |
| Model     | CPU-friendly             |
| FPS       | ~20–30 FPS               |
| Accuracy  | High with proper dataset |

---

# 🧪 Example Commands

## Collect Data

```bash
python collect_data.py
```

## Train Model

```bash
python train_model.py
```

## Real-Time Prediction

```bash
python realtime_predict.py
```

## Launch Web App

```bash
streamlit run app.py
```

---

# 📝 Example Use Cases

* Sign language translation
* Gesture-controlled systems
* Human-computer interaction
* Smart home automation
* Assistive technology
* Educational projects

---

# 👨‍💻 Author

Developed using:

* Python
* MediaPipe
* OpenCV
* Scikit-learn
* Streamlit
* ESP32

---

# 📜 License

This project is open-source and available under the MIT License.

---

# ⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork the project
🛠 Contribute improvements

---

