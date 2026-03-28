import cv2
import numpy as np


def resize_frame(frame, width=None, height=None):
    h, w = frame.shape[:2]
    if width and height:
        return cv2.resize(frame, (width, height))
    elif width:
        ratio = width / w
        return cv2.resize(frame, (width, int(h * ratio)))
    elif height:
        ratio = height / h
        return cv2.resize(frame, (int(w * ratio), height))
    return frame


def crop_face(frame, bbox, padding=0.2):
    h, w = frame.shape[:2]
    x, y, fw, fh = bbox

    pad_w = int(fw * padding)
    pad_h = int(fh * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + fw + pad_w)
    y2 = min(h, y + fh + pad_h)

    return frame[y1:y2, x1:x2]


def preprocess_face_for_emotion(face_roi):
    if face_roi is None or face_roi.size == 0:
        return None
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)


def preprocess_face_for_age_gender(face_roi):
    if face_roi is None or face_roi.size == 0:
        return None
    blob = cv2.dnn.blobFromImage(
        face_roi, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )
    return blob


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2

    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_label_with_background(img, text, position, font_scale=0.6, color=(255, 255, 255),
                                bg_color=(0, 0, 0), thickness=1, padding=5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position

    cv2.rectangle(
        img,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color, -1
    )

    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def overlay_transparent(background, overlay, x, y, alpha=0.7):
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    if x >= bg_w or y >= bg_h:
        return background
    if x + w > bg_w:
        w = bg_w - x
    if y + h > bg_h:
        h = bg_h - y

    roi = background[y:y + h, x:x + w]
    blended = cv2.addWeighted(roi, 1 - alpha, overlay[:h, :w], alpha, 0)
    background[y:y + h, x:x + w] = blended
    return background
