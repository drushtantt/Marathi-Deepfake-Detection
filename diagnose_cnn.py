from __future__ import annotations

import io
import math
import random
import site
import sys
import wave
from itertools import product
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


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "CNN_model.h5"
CLASS_LABELS = ["Human", "Amazon Polly", "gTTS"]


def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def wav_bytes(samples: list[float], sample_rate: int = 22050) -> bytes:
    clipped = [max(-1.0, min(1.0, sample)) for sample in samples]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = b"".join(
            int(sample * 32767).to_bytes(2, byteorder="little", signed=True)
            for sample in clipped
        )
        wav_file.writeframes(frames)
    return buffer.getvalue()


def build_signals(sample_rate: int = 22050, seconds: float = 1.0) -> dict[str, bytes]:
    count = int(sample_rate * seconds)
    time_axis = [i / sample_rate for i in range(count)]

    signals = {
        "silence": [0.0] * count,
        "sine_220": [0.35 * math.sin(2 * math.pi * 220 * t) for t in time_axis],
        "sine_440": [0.35 * math.sin(2 * math.pi * 440 * t) for t in time_axis],
        "dual_tone": [
            0.2 * math.sin(2 * math.pi * 220 * t) + 0.2 * math.sin(2 * math.pi * 660 * t)
            for t in time_axis
        ],
        "chirp": [
            0.25 * math.sin(2 * math.pi * (160 + 520 * t / seconds) * t)
            for t in time_axis
        ],
        "noise": [random.uniform(-0.35, 0.35) for _ in range(count)],
        "impulse_train": [0.9 if i % 1800 == 0 else 0.0 for i in range(count)],
    }

    return {name: wav_bytes(samples, sample_rate) for name, samples in signals.items()}


def preprocess_variant(
    file_bytes: bytes,
    *,
    target_sr: int | None,
    n_mels: int,
    hop_length: int,
    n_fft: int,
    use_db: bool,
    norm_mode: str,
    transpose: bool,
) -> np.ndarray:
    audio_data, sample_rate = librosa.load(io.BytesIO(file_bytes), sr=target_sr)

    mel = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
    )

    features = librosa.power_to_db(mel, ref=np.max) if use_db else mel

    if transpose:
        features = features.T

    features = librosa.util.fix_length(features, size=128, axis=0)
    features = librosa.util.fix_length(features, size=128, axis=1)
    features = features[:128, :128].astype("float32")

    if norm_mode == "minmax":
        min_value = float(np.min(features))
        max_value = float(np.max(features))
        if max_value > min_value:
            features = (features - min_value) / (max_value - min_value)
        else:
            features = np.zeros((128, 128), dtype="float32")
    elif norm_mode == "standard":
        mean_value = float(np.mean(features))
        std_value = float(np.std(features))
        if std_value > 0:
            features = (features - mean_value) / std_value
        else:
            features = np.zeros((128, 128), dtype="float32")
    elif norm_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported norm mode: {norm_mode}")

    return np.expand_dims(features, axis=(0, -1))


def variant_name(config: dict[str, object]) -> str:
    return (
        f"sr={config['target_sr']}_mels={config['n_mels']}_hop={config['hop_length']}"
        f"_fft={config['n_fft']}_db={config['use_db']}_norm={config['norm_mode']}"
        f"_transpose={config['transpose']}"
    )


def predict(model: tf.keras.Model, tensor: np.ndarray) -> tuple[int, list[float]]:
    probabilities = model.predict(tensor, verbose=0)[0]
    return int(np.argmax(probabilities)), [round(float(value), 6) for value in probabilities.tolist()]


def main() -> None:
    random.seed(7)
    np.random.seed(7)

    model = load_model()
    signals = build_signals()

    configs = []
    for target_sr, n_mels, hop_length, n_fft, use_db, norm_mode, transpose in product(
        [None, 22050],
        [64, 128],
        [256, 512],
        [1024, 2048],
        [False, True],
        ["none", "minmax", "standard"],
        [False, True],
    ):
        configs.append(
            {
                "target_sr": target_sr,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "n_fft": n_fft,
                "use_db": use_db,
                "norm_mode": norm_mode,
                "transpose": transpose,
            }
        )

    varying_variants = []
    constant_variants = []

    for config in configs:
        predictions = {}
        for signal_name, file_bytes in signals.items():
            tensor = preprocess_variant(file_bytes, **config)
            class_index, probabilities = predict(model, tensor)
            predictions[signal_name] = {
                "class_index": class_index,
                "label": CLASS_LABELS[class_index],
                "probabilities": probabilities,
            }

        predicted_indices = {result["class_index"] for result in predictions.values()}
        target_list = varying_variants if len(predicted_indices) > 1 else constant_variants
        target_list.append((config, predictions))

    print(f"tested_variants={len(configs)}")
    print(f"varying_variants={len(varying_variants)}")
    print(f"constant_variants={len(constant_variants)}")
    print()

    if varying_variants:
        print("Variants that changed class across inputs:")
        for config, predictions in varying_variants[:10]:
            print(variant_name(config))
            for signal_name, result in predictions.items():
                print(
                    f"  {signal_name:12s} -> {result['label']:12s} "
                    f"{result['probabilities']}"
                )
            print()
    else:
        print("No tested preprocessing variant changed the predicted class across inputs.")
        print()

    print("Sample constant variants:")
    for config, predictions in constant_variants[:5]:
        print(variant_name(config))
        for signal_name, result in predictions.items():
            print(
                f"  {signal_name:12s} -> {result['label']:12s} "
                f"{result['probabilities']}"
            )
        print()


if __name__ == "__main__":
    main()
