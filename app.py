from __future__ import annotations

import io
import os
import site
import sys
from functools import lru_cache
from pathlib import Path


USER_SITE = site.getusersitepackages()
if USER_SITE and USER_SITE not in sys.path:
    sys.path.append(USER_SITE)

LOCAL_PACKAGES = Path(__file__).resolve().parent / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.append(str(LOCAL_PACKAGES))

import librosa
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "CNN_model.h5"
CLASS_LABELS = ["Human", "Amazon Polly", "gTTS"]
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

app = Flask(__name__, static_folder=None)


@lru_cache(maxsize=1)
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def extract_melspectrogram(file_bytes: bytes) -> np.ndarray:
    audio_data, sample_rate = librosa.load(io.BytesIO(file_bytes), sr=None)

    mel = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_mels=128,
        hop_length=256,
        n_fft=1024,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)
    mel_db = mel_db[:128, :128].astype("float32")

    return mel_db


def predict_audio(file_bytes: bytes) -> dict:
    mel = extract_melspectrogram(file_bytes)
    model_input = np.expand_dims(mel, axis=(0, -1))

    model = load_model()
    probabilities = model.predict(model_input, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index]) * 100
    class_probabilities = {
        label: round(float(score) * 100, 2)
        for label, score in zip(CLASS_LABELS, probabilities.tolist())
    }

    return {
        "label": CLASS_LABELS[predicted_index],
        "classIndex": predicted_index,
        "confidence": round(confidence, 2),
        "probabilities": class_probabilities,
        "inputShape": [128, 128, 1],
    }


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/styles.css")
def styles():
    return send_from_directory(BASE_DIR, "styles.css")


@app.get("/script.js")
def script():
    return send_from_directory(BASE_DIR, "script.js")


@app.post("/predict")
def predict():
    uploaded_file = request.files.get("audio")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "No audio file was uploaded."}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Unsupported file type. Use WAV, MP3, FLAC, OGG, or M4A."}), 400

    file_bytes = uploaded_file.read()
    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        result = predict_audio(file_bytes)
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
