import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "registered_faces")
DB_PATH = os.path.join(DATA_DIR, "faces.db")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

DETECTION_CONFIDENCE = 0.7
DETECTION_METHOD = "dnn"

RECOGNITION_TOLERANCE = 0.6
RECOGNITION_MODEL = "hog"

ANALYSIS_INTERVAL = 3
MAX_FACES = 5

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_EMOJIS = {
    "Angry": "\U0001f620",
    "Disgust": "\U0001f922",
    "Fear": "\U0001f628",
    "Happy": "\U0001f604",
    "Sad": "\U0001f622",
    "Surprise": "\U0001f632",
    "Neutral": "\U0001f610",
}

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LIST = ["Male", "Female"]

MODEL_URLS = {
    "face_detector_prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "face_detector_model": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "age_prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    "age_model": "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel",
    "gender_prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    "gender_model": "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel",
}

MODEL_FILES = {
    "face_detector_prototxt": os.path.join(MODELS_DIR, "deploy.prototxt"),
    "face_detector_model": os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel"),
    "age_prototxt": os.path.join(MODELS_DIR, "age_deploy.prototxt"),
    "age_model": os.path.join(MODELS_DIR, "age_net.caffemodel"),
    "gender_prototxt": os.path.join(MODELS_DIR, "gender_deploy.prototxt"),
    "gender_model": os.path.join(MODELS_DIR, "gender_net.caffemodel"),
    "emotion_model": os.path.join(MODELS_DIR, "emotion_model.h5"),
}

APP_TITLE = "FaceVision Pro — Real-Time Facial Recognition System"
APP_MIN_WIDTH = 1280
APP_MIN_HEIGHT = 800
SIDEBAR_WIDTH = 220
