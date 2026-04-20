"""
STEP 2: Train gesture classifier
----------------------------------
Uses a pre-trained backbone: MediaPipe already extracts features (landmarks).
We train a lightweight MLP on top — fast, accurate, runs on CPU.

Outputs:
  gesture_model.pkl   — sklearn model (use in Python)
  gesture_model.tflite — TFLite model (optional, for ESP32 / mobile)
  label_map.json      — gesture index ↔ name mapping
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.neural_network  import MLPClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import classification_report, confusion_matrix
import joblib
import os

# ── Config ──────────────────────────────────────────────────────────────────
DATA_FILE   = 'gesture_data.json'
MODEL_FILE  = 'gesture_model.pkl'
LABELS_FILE = 'label_map.json'
MODEL_TYPE  = 'mlp'   # 'mlp'  or  'rf' (random forest)
# ────────────────────────────────────────────────────────────────────────────


def load_data():
    with open(DATA_FILE, 'r') as f:
        d = json.load(f)
    X = np.array(d['samples'], dtype=np.float32)
    y = np.array(d['labels'])
    return X, y


def augment_data(X, y, factor=3):
    """
    Light augmentation: add small Gaussian noise to landmarks.
    Triples dataset size, improves robustness to hand position jitter.
    """
    X_aug, y_aug = [X], [y]
    for _ in range(factor - 1):
        noise = np.random.normal(0, 0.01, X.shape).astype(np.float32)
        X_aug.append(X + noise)
        y_aug.append(y)
    return np.vstack(X_aug), np.concatenate(y_aug)


def build_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True,
    )


def build_rf():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )


def train():
    print("── Loading data ─────────────────────────────────")
    X, y = load_data()
    print(f"   Samples: {len(X)}  |  Classes: {np.unique(y)}")

    # ── Label encode ────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    label_map = {int(i): name for i, name in enumerate(le.classes_)}
    with open(LABELS_FILE, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"   Label map: {label_map}")

    # ── Augment ─────────────────────────────────────────────────────────────
    print("\n── Augmenting data ──────────────────────────────")
    X_aug, y_aug = augment_data(X, y_enc, factor=3)
    print(f"   Augmented to {len(X_aug)} samples")

    # ── Train / test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\n── Training {MODEL_TYPE.upper()} ─────────────────────────────")
    model = build_mlp() if MODEL_TYPE == 'mlp' else build_rf()
    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n── Evaluation ───────────────────────────────────")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    acc = (y_pred == y_test).mean()
    print(f"\n✅ Test accuracy: {acc*100:.2f}%")

    # ── Save sklearn model ───────────────────────────────────────────────────
    joblib.dump({'model': model, 'label_encoder': le}, MODEL_FILE)
    print(f"✅ Model saved to {MODEL_FILE}")

    # ── Optional: export TFLite (uncomment if you have tensorflow) ───────────
    # export_tflite(model, X_train, le)

    return model, le


def export_tflite(sklearn_model, X_sample, le):
    """
    Convert to TFLite for on-device inference.
    Requires: pip install tensorflow
    """
    try:
        import tensorflow as tf

        # Rebuild as TF Keras model mirroring MLP weights
        n_in  = X_sample.shape[1]
        n_out = len(le.classes_)

        inp = tf.keras.Input(shape=(n_in,))
        x   = tf.keras.layers.Dense(256, activation='relu')(inp)
        x   = tf.keras.layers.Dense(128, activation='relu')(x)
        x   = tf.keras.layers.Dense(64,  activation='relu')(x)
        out = tf.keras.layers.Dense(n_out, activation='softmax')(x)
        tf_model = tf.keras.Model(inp, out)

        # Copy weights from sklearn MLP
        for i, (W, b) in enumerate(zip(sklearn_model.coefs_, sklearn_model.intercepts_)):
            tf_model.layers[i+1].set_weights([W, b])

        converter   = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open('gesture_model.tflite', 'wb') as f:
            f.write(tflite_model)
        print("✅ TFLite model saved to gesture_model.tflite")

    except ImportError:
        print("⚠ TensorFlow not installed — skipping TFLite export")


if __name__ == '__main__':
    train()
