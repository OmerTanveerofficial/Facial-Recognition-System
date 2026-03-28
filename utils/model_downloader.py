import os
import urllib.request
import numpy as np
from utils.config import MODEL_URLS, MODEL_FILES, MODELS_DIR
from utils.logger import log


def download_file(url, dest_path, callback=None):
    try:
        log.info(f"Downloading: {os.path.basename(dest_path)}")

        def reporthook(block_num, block_size, total_size):
            if callback and total_size > 0:
                progress = min(block_num * block_size / total_size, 1.0)
                callback(os.path.basename(dest_path), progress)

        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        log.info(f"Downloaded: {os.path.basename(dest_path)}")
        return True
    except Exception as e:
        log.error(f"Failed to download {os.path.basename(dest_path)}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def create_emotion_model():
    try:
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            layers.Input(shape=(48, 48, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(7, activation="softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.save(MODEL_FILES["emotion_model"])
        log.info("Emotion model architecture created and saved")
        return True
    except Exception as e:
        log.error(f"Failed to create emotion model: {e}")
        return False


def check_models():
    missing = {}
    for key, path in MODEL_FILES.items():
        if not os.path.exists(path):
            if key in MODEL_URLS:
                missing[key] = MODEL_URLS[key]
            elif key == "emotion_model":
                missing[key] = "create"
    return missing


def download_all_models(callback=None):
    os.makedirs(MODELS_DIR, exist_ok=True)
    missing = check_models()

    if not missing:
        log.info("All models are already available")
        return True

    success = True
    total = len(missing)
    for i, (key, url_or_action) in enumerate(missing.items()):
        if callback:
            callback(f"Downloading model {i + 1}/{total}...", i / total)

        if url_or_action == "create":
            if not create_emotion_model():
                success = False
        else:
            dest = MODEL_FILES[key]
            if not download_file(url_or_action, dest, callback):
                success = False

    if callback:
        callback("All models ready!" if success else "Some models failed to download", 1.0)

    return success


def models_available():
    return all(os.path.exists(p) for p in MODEL_FILES.values())
