import cv2
import numpy as np
from dataclasses import dataclass
from utils.config import AGE_BUCKETS, GENDER_LIST, MODEL_FILES
from utils.logger import log


@dataclass
class AgeGenderResult:
    age_range: str
    gender: str
    age_confidence: float
    gender_confidence: float


class AgeGenderEstimator:

    def __init__(self):
        self._age_net = None
        self._gender_net = None
        self._initialized = False
        self._init_models()

    def _init_models(self):
        try:
            self._age_net = cv2.dnn.readNet(
                MODEL_FILES["age_model"],
                MODEL_FILES["age_prototxt"]
            )
            self._gender_net = cv2.dnn.readNet(
                MODEL_FILES["gender_model"],
                MODEL_FILES["gender_prototxt"]
            )
            self._initialized = True
            log.info("Age/gender estimator initialized")
        except Exception as e:
            log.warning(f"Age/gender estimator not available: {e}")

    def estimate(self, face_roi):
        if not self._initialized or face_roi is None or face_roi.size == 0:
            return None

        try:
            blob = cv2.dnn.blobFromImage(
                face_roi, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )

            self._gender_net.setInput(blob)
            gender_preds = self._gender_net.forward()
            gender_idx = gender_preds[0].argmax()
            gender = GENDER_LIST[gender_idx]
            gender_confidence = float(gender_preds[0][gender_idx])

            self._age_net.setInput(blob)
            age_preds = self._age_net.forward()
            age_idx = age_preds[0].argmax()
            age_range = AGE_BUCKETS[age_idx]
            age_confidence = float(age_preds[0][age_idx])

            return AgeGenderResult(
                age_range=age_range,
                gender=gender,
                age_confidence=age_confidence,
                gender_confidence=gender_confidence
            )

        except Exception as e:
            log.error(f"Age/gender estimation failed: {e}")
            return None

    @property
    def is_ready(self):
        return self._initialized
