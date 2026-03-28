import sys
import os
import faulthandler
faulthandler.enable()

import cv2
import time
import json
import base64
import threading
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.analysis_pipeline import AnalysisPipeline
from core.face_detector import FaceDetector
from database.db_manager import DatabaseManager
from database.face_store import FaceStore
from utils.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, FACES_DIR
from utils.image_utils import draw_rounded_rect, draw_label_with_background
from utils.logger import log

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__, template_folder="templates", static_folder="static")


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


db_manager = DatabaseManager()
face_store = FaceStore(db_manager)
pipeline = AnalysisPipeline(face_store)

camera_lock = threading.Lock()
camera = None
camera_active = False
latest_results = []
latest_fps = 0.0


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return camera


def release_camera():
    global camera, camera_active
    camera_active = False
    if camera and camera.isOpened():
        camera.release()
    camera = None
    log.info("Camera released")


def draw_overlays(frame, results):
    annotated = frame.copy()

    for result in results:
        fr = result.face_region
        x, y, w, h = fr.bbox

        if result.is_known:
            color = (0, 212, 170)
        else:
            color = (38, 167, 255)

        draw_rounded_rect(annotated, (x, y), (x + w, y + h), color, 2, radius=8)

        corner_len = min(20, w // 4, h // 4)
        thickness = 3
        cv2.line(annotated, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(annotated, (x, y), (x, y + corner_len), color, thickness)
        cv2.line(annotated, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(annotated, (x + w, y), (x + w, y + corner_len), color, thickness)
        cv2.line(annotated, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(annotated, (x, y + h), (x, y + h - corner_len), color, thickness)
        cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

        info_lines = result.info_lines
        label_y = y - 8
        for line in reversed(info_lines):
            bg_color = (color[0] // 2, color[1] // 2, color[2] // 2)
            draw_label_with_background(
                annotated, line, (x, label_y),
                font_scale=0.55, color=(255, 255, 255),
                bg_color=bg_color, padding=4
            )
            label_y -= 24

        if result.recognition:
            conf = result.recognition.confidence
            bar_width = int(w * conf)
            bar_y = y + h + 4
            cv2.rectangle(annotated, (x, bar_y), (x + w, bar_y + 4), (50, 50, 50), -1)
            cv2.rectangle(annotated, (x, bar_y), (x + bar_width, bar_y + 4), color, -1)

    return annotated


def generate_frames():
    global latest_results, latest_fps, camera_active

    camera_active = True
    cap = get_camera()
    if not cap.isOpened():
        log.error("Cannot open camera")
        return

    log.info("Video stream started")

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        try:
            results = pipeline.analyze_frame(frame)
            latest_results = results
            latest_fps = pipeline.fps

            annotated = draw_overlays(frame, results)
        except Exception as e:
            log.error(f"Frame analysis error: {e}")
            annotated = frame

        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    release_camera()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    release_camera()
    return jsonify({"status": "stopped"})


@app.route("/api/status")
def get_status():
    result_data = []
    for r in latest_results:
        face_data = {
            "name": r.display_name,
            "is_known": r.is_known,
            "confidence": r.recognition.confidence if r.recognition else 0,
            "bbox": list(r.face_region.bbox),
        }
        if r.emotion:
            face_data["emotion"] = r.emotion.dominant_emotion
            face_data["emotion_emoji"] = r.emotion.emoji
            face_data["emotion_confidence"] = r.emotion.confidence
        if r.age_gender:
            face_data["age_range"] = r.age_gender.age_range
            face_data["gender"] = r.age_gender.gender
        result_data.append(face_data)

    return jsonify(sanitize_for_json({
        "camera_active": camera_active,
        "fps": round(float(latest_fps), 1),
        "face_count": len(latest_results),
        "faces": result_data,
        "registered_count": db_manager.get_person_count(),
        "pipeline_status": pipeline.status,
    }))


@app.route("/api/persons")
def get_persons():
    persons = db_manager.get_all_persons()
    for p in persons:
        p["recognition_count"] = db_manager.get_recognition_count(p["id"])
        thumb = face_store.get_person_thumbnail(p["id"])
        if thumb is not None:
            _, buf = cv2.imencode(".jpg", thumb)
            p["thumbnail"] = base64.b64encode(buf).decode("utf-8")
        else:
            p["thumbnail"] = None
    return jsonify(persons)


@app.route("/api/persons/search")
def search_persons():
    query = request.args.get("q", "")
    persons = db_manager.search_persons(query)
    for p in persons:
        p["recognition_count"] = db_manager.get_recognition_count(p["id"])
        thumb = face_store.get_person_thumbnail(p["id"])
        if thumb is not None:
            _, buf = cv2.imencode(".jpg", thumb)
            p["thumbnail"] = base64.b64encode(buf).decode("utf-8")
        else:
            p["thumbnail"] = None
    return jsonify(persons)


@app.route("/api/persons/<int:person_id>", methods=["PUT"])
def update_person(person_id):
    data = request.json
    db_manager.update_person(
        person_id,
        name=data.get("name"),
        notes=data.get("notes")
    )
    pipeline.reload_faces()
    return jsonify({"status": "updated"})


@app.route("/api/persons/<int:person_id>", methods=["DELETE"])
def delete_person(person_id):
    face_store.delete_person(person_id)
    pipeline.reload_faces()
    return jsonify({"status": "deleted"})


@app.route("/api/register/capture", methods=["POST"])
def capture_face():
    reg_cap = cv2.VideoCapture(CAMERA_INDEX)
    reg_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    reg_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not reg_cap.isOpened():
        return jsonify({"error": "Cannot open camera"}), 500

    for _ in range(5):
        reg_cap.read()

    ret, frame = reg_cap.read()
    reg_cap.release()

    if not ret:
        return jsonify({"error": "Cannot read from camera"}), 500

    faces = pipeline.detector.detect(frame)

    if not faces:
        return jsonify({"error": "No face detected — make sure your face is visible"}), 400

    face = faces[0]
    face_roi = face.get_padded_roi(frame)

    encoding = None
    try:
        encodings = pipeline.recognizer.get_encoding(frame, face.to_location())
        if encodings:
            encoding = encodings[0].tolist()
    except Exception as e:
        log.warning(f"Could not get face encoding: {e}")

    _, buf = cv2.imencode(".jpg", face_roi)
    face_b64 = base64.b64encode(buf).decode("utf-8")

    return jsonify(sanitize_for_json({
        "face_image": face_b64,
        "encoding": encoding,
        "bbox": list(face.bbox),
        "confidence": face.confidence,
    }))


@app.route("/api/register/save", methods=["POST"])
def save_registration():
    data = request.json
    name = data.get("name", "").strip()
    notes = data.get("notes", "").strip()
    captures = data.get("captures", [])

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not captures:
        return jsonify({"error": "No face captures provided"}), 400

    encodings = []
    face_images = []

    for capture in captures:
        img_data = base64.b64decode(capture["face_image"])
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        face_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        face_images.append(face_img)

        if capture.get("encoding"):
            encodings.append(np.array(capture["encoding"]))

    if encodings:
        person_id = face_store.register_face(name, encodings, face_images, notes)
    else:
        person_id = db_manager.add_person(name, notes)
        person_dir = os.path.join(FACES_DIR, str(person_id))
        os.makedirs(person_dir, exist_ok=True)
        for i, img in enumerate(face_images):
            cv2.imwrite(os.path.join(person_dir, f"face_{i}.jpg"), img)

    if person_id:
        pipeline.reload_faces()
        return jsonify({"status": "registered", "person_id": person_id})
    return jsonify({"error": "Registration failed"}), 500


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.json

    if "detection_confidence" in data:
        pipeline.set_detection_confidence(data["detection_confidence"])
    if "recognition_tolerance" in data:
        pipeline.set_recognition_tolerance(data["recognition_tolerance"])
    if "analysis_interval" in data:
        pipeline.set_analysis_interval(data["analysis_interval"])
    if "emotion_enabled" in data:
        pipeline.toggle_emotion(data["emotion_enabled"])
    if "age_gender_enabled" in data:
        pipeline.toggle_age_gender(data["age_gender_enabled"])

    return jsonify({"status": "updated", "settings": pipeline.status})


@app.route("/api/models/download", methods=["POST"])
def download_models():
    from utils.model_downloader import download_all_models
    success = download_all_models()
    return jsonify({"status": "success" if success else "partial_failure"})


@app.route("/api/logs/recent")
def recent_logs():
    logs = db_manager.get_recent_logs(limit=50)
    return jsonify(logs)


def run_server(host="127.0.0.1", port=5000, debug=False):
    print(f"\n  FaceVision Pro is running at: http://{host}:{port}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run_server(debug=True)
