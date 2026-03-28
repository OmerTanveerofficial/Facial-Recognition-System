import cv2
import threading
import queue
import time
import numpy as np
from PIL import Image
import customtkinter as ctk
from core.analysis_pipeline import AnalysisPipeline
from utils.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
from utils.image_utils import draw_rounded_rect, draw_label_with_background
from gui.theme import Colors
from utils.logger import log


class CameraThread(threading.Thread):

    def __init__(self, pipeline, frame_queue, camera_index=None):
        super().__init__(daemon=True)
        self.pipeline = pipeline
        self.frame_queue = frame_queue
        self.camera_index = camera_index if camera_index is not None else CAMERA_INDEX
        self._running = False
        self._cap = None

    def run(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not self._cap.isOpened():
            log.error(f"Cannot open camera {self.camera_index}")
            return

        self._running = True
        log.info(f"Camera {self.camera_index} started")

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            results = self.pipeline.analyze_frame(frame)
            annotated_frame = self._draw_overlays(frame, results)

            try:
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                self.frame_queue.put_nowait((annotated_frame, results, self.pipeline.fps))
            except queue.Full:
                pass

        self._cleanup()

    def _draw_overlays(self, frame, results):
        annotated = frame.copy()

        for result in results:
            fr = result.face_region
            x, y, w, h = fr.bbox

            if result.is_known:
                color = Colors.FACE_KNOWN
            else:
                color = Colors.FACE_UNKNOWN

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

            for i, line in enumerate(reversed(info_lines)):
                bg_color = (color[0] // 2, color[1] // 2, color[2] // 2)
                draw_label_with_background(
                    annotated, line,
                    (x, label_y),
                    font_scale=0.55,
                    color=(255, 255, 255),
                    bg_color=bg_color,
                    padding=4
                )
                label_y -= 24

            if result.recognition:
                conf = result.recognition.confidence
                bar_width = int(w * conf)
                bar_y = y + h + 4
                cv2.rectangle(annotated, (x, bar_y), (x + w, bar_y + 4),
                              (50, 50, 50), -1)
                cv2.rectangle(annotated, (x, bar_y), (x + bar_width, bar_y + 4),
                              color, -1)

        return annotated

    def stop(self):
        self._running = False

    def _cleanup(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()
            log.info("Camera released")

    @property
    def is_running(self):
        return self._running


class CameraView(ctk.CTkLabel):

    def __init__(self, parent, frame_queue):
        super().__init__(parent, text="Camera Feed", text_color=Colors.TEXT_MUTED)
        self.frame_queue = frame_queue
        self._current_image = None
        self._polling = False
        self._on_frame_callback = None

    def start_polling(self, interval=33):
        self._polling = True
        self._poll(interval)

    def stop_polling(self):
        self._polling = False

    def _poll(self, interval):
        if not self._polling:
            return

        try:
            frame, results, fps = self.frame_queue.get_nowait()
            self._display_frame(frame)

            if self._on_frame_callback:
                self._on_frame_callback(results, fps)

        except queue.Empty:
            pass

        self.after(interval, lambda: self._poll(interval))

    def _display_frame(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            widget_w = self.winfo_width()
            widget_h = self.winfo_height()

            if widget_w > 1 and widget_h > 1:
                img_w, img_h = pil_img.size
                scale = min(widget_w / img_w, widget_h / img_h)
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

            ctk_img = ctk.CTkImage(pil_img, size=pil_img.size)
            self.configure(image=ctk_img, text="")
            self._current_image = ctk_img

        except Exception as e:
            pass

    def set_frame_callback(self, callback):
        self._on_frame_callback = callback

    def show_placeholder(self, text="Click 'Start Camera' to begin"):
        self.configure(image=None, text=text)
        self._current_image = None
