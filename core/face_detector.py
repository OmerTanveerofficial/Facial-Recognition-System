import cv2
import numpy as np
from dataclasses import dataclass
from utils.config import DETECTION_CONFIDENCE, MODEL_FILES
from utils.logger import log


@dataclass
class FaceRegion:
    x: int
    y: int
    w: int
    h: int
    confidence: float

    @property
    def bbox(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self):
        return self.w * self.h

    def to_location(self):
        return (self.y, self.x + self.w, self.y + self.h, self.x)

    def get_roi(self, frame):
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]

    def get_padded_roi(self, frame, padding=0.2):
        fh, fw = frame.shape[:2]
        pad_w = int(self.w * padding)
        pad_h = int(self.h * padding)
        x1 = max(0, self.x - pad_w)
        y1 = max(0, self.y - pad_h)
        x2 = min(fw, self.x + self.w + pad_w)
        y2 = min(fh, self.y + self.h + pad_h)
        return frame[y1:y2, x1:x2]


class FaceDetector:

    def __init__(self, method="dnn", confidence_threshold=None):
        self.method = method
        self.confidence = confidence_threshold or DETECTION_CONFIDENCE
        self._net = None
        self._haar = None
        self._initialized = False
        self._init_detector()

    def _init_detector(self):
        if self.method == "dnn":
            try:
                prototxt = MODEL_FILES.get("face_detector_prototxt")
                model = MODEL_FILES.get("face_detector_model")
                self._net = cv2.dnn.readNetFromCaffe(prototxt, model)
                self._initialized = True
                log.info("DNN face detector initialized (SSD ResNet-10)")
            except Exception as e:
                log.warning(f"DNN detector failed, falling back to Haar: {e}")
                self.method = "haar"

        if self.method == "haar":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            if self._haar.empty():
                log.error("Failed to load Haar cascade classifier")
            else:
                self._initialized = True
                log.info("Haar cascade face detector initialized")

    def detect(self, frame):
        if not self._initialized:
            return []

        if self.method == "dnn":
            faces = self._detect_dnn(frame)
        else:
            faces = self._detect_haar(frame)

        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    def _detect_dnn(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            fw = x2 - x1
            fh = y2 - y1
            if fw > 20 and fh > 20:
                faces.append(FaceRegion(x1, y1, fw, fh, float(confidence)))

        return faces

    def _detect_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        faces = []
        for (x, y, w, h) in rects:
            faces.append(FaceRegion(x, y, w, h, 0.9))

        return faces

    def set_confidence(self, threshold):
        self.confidence = max(0.1, min(1.0, threshold))

    @property
    def is_ready(self):
        return self._initialized
