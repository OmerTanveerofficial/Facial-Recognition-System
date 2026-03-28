import cv2
import time
import threading
import queue
import numpy as np
import customtkinter as ctk
from PIL import Image
from gui.theme import Colors, Fonts, Spacing
from core.face_detector import FaceDetector
from utils.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
from utils.logger import log

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False


class RegistrationPanel(ctk.CTkFrame):

    def __init__(self, parent, face_store, pipeline, toast_manager):
        super().__init__(parent, fg_color=Colors.BG_PRIMARY, corner_radius=0)
        self.face_store = face_store
        self.pipeline = pipeline
        self.toast = toast_manager
        self.detector = FaceDetector()

        self._step = 1
        self._cap = None
        self._camera_active = False
        self._captured_frames = []
        self._captured_encodings = []
        self._captured_thumbnails = []
        self._face_quality = "No face detected"

        self._build_ui()

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=Spacing.PANEL_PADDING, pady=(Spacing.XL, Spacing.MD))

        ctk.CTkLabel(
            header, text="Register New Face",
            font=Fonts.HEADING_LG, text_color=Colors.TEXT_PRIMARY
        ).pack(side="left")

        self.step_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.step_frame.pack(fill="x", padx=Spacing.PANEL_PADDING, pady=(0, Spacing.LG))

        self.step_labels = []
        steps = ["1. Position Face", "2. Capture", "3. Save Details"]
        for i, step_text in enumerate(steps):
            label = ctk.CTkLabel(
                self.step_frame, text=step_text,
                font=Fonts.BODY_BOLD if i == 0 else Fonts.BODY,
                text_color=Colors.ACCENT if i == 0 else Colors.TEXT_MUTED
            )
            label.pack(side="left", padx=(0, Spacing.XL))
            self.step_labels.append(label)

        self.progress_bar = ctk.CTkProgressBar(
            self, width=400, height=4,
            fg_color=Colors.BG_INPUT, progress_color=Colors.ACCENT
        )
        self.progress_bar.pack(padx=Spacing.PANEL_PADDING, pady=(0, Spacing.LG))
        self.progress_bar.set(0.33)

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True,
                                padx=Spacing.PANEL_PADDING, pady=(0, Spacing.PANEL_PADDING))

        self._show_step_1()

    def _update_step_indicator(self):
        for i, label in enumerate(self.step_labels):
            step_num = i + 1
            if step_num == self._step:
                label.configure(font=Fonts.BODY_BOLD, text_color=Colors.ACCENT)
            elif step_num < self._step:
                label.configure(font=Fonts.BODY, text_color=Colors.SUCCESS)
            else:
                label.configure(font=Fonts.BODY, text_color=Colors.TEXT_MUTED)

        self.progress_bar.set(self._step / 3)

    def _clear_content(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def _show_step_1(self):
        self._step = 1
        self._update_step_indicator()
        self._clear_content()

        left = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, Spacing.MD))

        right = ctk.CTkFrame(self.content_frame, fg_color="transparent", width=300)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        camera_border = ctk.CTkFrame(
            left, fg_color=Colors.BG_SECONDARY,
            corner_radius=Spacing.CARD_RADIUS,
            border_color=Colors.BORDER, border_width=1
        )
        camera_border.pack(fill="both", expand=True)

        self.reg_camera_label = ctk.CTkLabel(
            camera_border, text="Starting camera...",
            text_color=Colors.TEXT_MUTED
        )
        self.reg_camera_label.pack(fill="both", expand=True, padx=4, pady=4)

        instructions = ctk.CTkFrame(right, fg_color=Colors.BG_SECONDARY,
                                     corner_radius=Spacing.CARD_RADIUS,
                                     border_color=Colors.BORDER, border_width=1)
        instructions.pack(fill="x")

        inst_inner = ctk.CTkFrame(instructions, fg_color="transparent")
        inst_inner.pack(padx=Spacing.CARD_PADDING, pady=Spacing.CARD_PADDING)

        ctk.CTkLabel(
            inst_inner, text="Instructions",
            font=Fonts.SUBHEADING, text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", pady=(0, Spacing.SM))

        tips = [
            "\u2022 Face the camera directly",
            "\u2022 Ensure good lighting",
            "\u2022 Remove sunglasses/mask",
            "\u2022 Keep face centered",
            "\u2022 Maintain neutral expression",
        ]
        for tip in tips:
            ctk.CTkLabel(
                inst_inner, text=tip,
                font=Fonts.BODY, text_color=Colors.TEXT_SECONDARY
            ).pack(anchor="w", pady=1)

        quality_frame = ctk.CTkFrame(right, fg_color=Colors.BG_SECONDARY,
                                      corner_radius=Spacing.CARD_RADIUS,
                                      border_color=Colors.BORDER, border_width=1)
        quality_frame.pack(fill="x", pady=(Spacing.MD, 0))

        q_inner = ctk.CTkFrame(quality_frame, fg_color="transparent")
        q_inner.pack(padx=Spacing.CARD_PADDING, pady=Spacing.CARD_PADDING)

        ctk.CTkLabel(
            q_inner, text="Face Quality",
            font=Fonts.SUBHEADING, text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w")

        self.quality_label = ctk.CTkLabel(
            q_inner, text="Starting...",
            font=Fonts.BODY, text_color=Colors.WARNING
        )
        self.quality_label.pack(anchor="w", pady=(Spacing.SM, 0))

        self.quality_bar = ctk.CTkProgressBar(
            q_inner, width=200, height=6,
            fg_color=Colors.BG_INPUT, progress_color=Colors.WARNING
        )
        self.quality_bar.pack(anchor="w", pady=(Spacing.SM, 0))
        self.quality_bar.set(0)

        btn_frame = ctk.CTkFrame(right, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(Spacing.LG, 0))

        self.capture_btn = ctk.CTkButton(
            btn_frame, text="Capture Faces  \u27a1",
            font=Fonts.BODY_BOLD, fg_color=Colors.ACCENT,
            hover_color=Colors.ACCENT_HOVER, height=40,
            corner_radius=8, state="disabled",
            command=self._start_capture
        )
        self.capture_btn.pack(fill="x")

        self._start_preview()

    def _start_preview(self):
        self._camera_active = True
        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()

    def _preview_loop(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self._cap = cap

        if not cap.isOpened():
            self.after(0, lambda: self.quality_label.configure(
                text="Cannot open camera", text_color=Colors.DANGER
            ))
            return

        while self._camera_active:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            faces = self.detector.detect(frame)
            display = frame.copy()

            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            axes = (w // 5, h // 3)
            cv2.ellipse(display, center, axes, 0, 0, 360, (100, 100, 100), 2)

            quality_score = 0.0
            quality_text = "No face detected"
            quality_color = Colors.WARNING

            if faces:
                face = faces[0]
                fx, fy, fw, fh = face.bbox

                color = (0, 212, 170)
                cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), color, 2)

                face_center_x = fx + fw // 2
                face_center_y = fy + fh // 2
                offset_x = abs(face_center_x - center[0]) / (w // 2)
                offset_y = abs(face_center_y - center[1]) / (h // 2)
                size_ratio = (fw * fh) / (w * h)

                if size_ratio < 0.03:
                    quality_text = "Too far — move closer"
                    quality_score = 0.3
                elif size_ratio > 0.4:
                    quality_text = "Too close — move back"
                    quality_score = 0.3
                elif offset_x > 0.3 or offset_y > 0.3:
                    quality_text = "Center your face"
                    quality_score = 0.5
                else:
                    quality_text = "Good position!"
                    quality_score = 1.0
                    quality_color = Colors.SUCCESS

                if len(faces) > 1:
                    quality_text = "Multiple faces — ensure only one person"
                    quality_score = 0.2

            self._update_preview_ui(display, quality_text, quality_score, quality_color)
            time.sleep(0.033)

        cap.release()

    def _update_preview_ui(self, frame, quality_text, quality_score, quality_color):
        def update():
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                widget_w = self.reg_camera_label.winfo_width()
                widget_h = self.reg_camera_label.winfo_height()
                if widget_w > 1 and widget_h > 1:
                    img_w, img_h = pil_img.size
                    scale = min(widget_w / img_w, widget_h / img_h)
                    pil_img = pil_img.resize(
                        (int(img_w * scale), int(img_h * scale)), Image.LANCZOS
                    )

                ctk_img = ctk.CTkImage(pil_img, size=pil_img.size)
                self.reg_camera_label.configure(image=ctk_img, text="")
                self.reg_camera_label._preview_img = ctk_img

                self.quality_label.configure(text=quality_text, text_color=quality_color)
                self.quality_bar.set(quality_score)
                bar_color = Colors.SUCCESS if quality_score >= 0.8 else Colors.WARNING
                self.quality_bar.configure(progress_color=bar_color)

                if quality_score >= 0.8:
                    self.capture_btn.configure(state="normal")
                else:
                    self.capture_btn.configure(state="disabled")
            except Exception:
                pass

        self.after(0, update)

    def _start_capture(self):
        self._camera_active = False
        time.sleep(0.1)

        self._step = 2
        self._update_step_indicator()
        self._clear_content()

        ctk.CTkLabel(
            self.content_frame, text="Capturing face data...",
            font=Fonts.HEADING, text_color=Colors.TEXT_PRIMARY
        ).pack(pady=(Spacing.XL, Spacing.MD))

        ctk.CTkLabel(
            self.content_frame, text="Please hold still and look at the camera",
            font=Fonts.BODY, text_color=Colors.TEXT_SECONDARY
        ).pack()

        self.capture_progress = ctk.CTkProgressBar(
            self.content_frame, width=400, height=8,
            fg_color=Colors.BG_INPUT, progress_color=Colors.ACCENT
        )
        self.capture_progress.pack(pady=Spacing.XL)
        self.capture_progress.set(0)

        self.capture_status = ctk.CTkLabel(
            self.content_frame, text="Capturing 0/5",
            font=Fonts.BODY, text_color=Colors.TEXT_SECONDARY
        )
        self.capture_status.pack()

        self.thumb_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.thumb_frame.pack(pady=Spacing.XL)

        self._captured_frames = []
        self._captured_encodings = []
        self._captured_thumbnails = []
        threading.Thread(target=self._capture_faces, daemon=True).start()

    def _capture_faces(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            self.after(0, lambda: self.toast.show("Cannot open camera", "error"))
            return

        target_captures = 5
        captured = 0

        while captured < target_captures:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            faces = self.detector.detect(frame)
            if not faces:
                time.sleep(0.1)
                continue

            face = faces[0]
            face_roi = face.get_padded_roi(frame)

            if FACE_REC_AVAILABLE:
                rgb = frame[:, :, ::-1]
                locations = [face.to_location()]
                encodings = face_recognition.face_encodings(rgb, locations)

                if encodings:
                    self._captured_encodings.append(encodings[0])
                    self._captured_frames.append(face_roi.copy())
                    captured += 1

                    self.after(0, lambda c=captured, roi=face_roi.copy():
                               self._update_capture_ui(c, target_captures, roi))
                    time.sleep(0.5)
            else:
                self._captured_frames.append(face_roi.copy())
                captured += 1
                self.after(0, lambda c=captured, roi=face_roi.copy():
                           self._update_capture_ui(c, target_captures, roi))
                time.sleep(0.5)

        cap.release()
        self.after(0, self._show_step_3)

    def _update_capture_ui(self, current, total, face_roi):
        try:
            self.capture_progress.set(current / total)
            self.capture_status.configure(text=f"Capturing {current}/{total}")

            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((64, 64), Image.LANCZOS)
            ctk_img = ctk.CTkImage(pil_img, size=(64, 64))

            thumb = ctk.CTkLabel(
                self.thumb_frame, text="", image=ctk_img,
                fg_color=Colors.BG_SECONDARY, corner_radius=8,
                width=72, height=72
            )
            thumb.pack(side="left", padx=4)
            thumb._img_ref = ctk_img
        except Exception:
            pass

    def _show_step_3(self):
        self._step = 3
        self._update_step_indicator()
        self._clear_content()

        card = ctk.CTkFrame(
            self.content_frame, fg_color=Colors.BG_SECONDARY,
            corner_radius=Spacing.CARD_RADIUS,
            border_color=Colors.BORDER, border_width=1,
            width=500
        )
        card.pack(expand=True)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(padx=Spacing.XXL, pady=Spacing.XXL)

        ctk.CTkLabel(
            inner, text="\u2713  Faces Captured Successfully!",
            font=Fonts.HEADING, text_color=Colors.SUCCESS
        ).pack(pady=(0, Spacing.LG))

        thumb_row = ctk.CTkFrame(inner, fg_color="transparent")
        thumb_row.pack(pady=(0, Spacing.LG))

        for face_img in self._captured_frames[:5]:
            try:
                rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb).resize((56, 56), Image.LANCZOS)
                ctk_img = ctk.CTkImage(pil_img, size=(56, 56))
                lbl = ctk.CTkLabel(
                    thumb_row, text="", image=ctk_img,
                    fg_color=Colors.BG_INPUT, corner_radius=8,
                    width=64, height=64
                )
                lbl.pack(side="left", padx=3)
                lbl._ref = ctk_img
            except Exception:
                pass

        ctk.CTkLabel(
            inner, text="Name", font=Fonts.BODY_BOLD,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w")

        self.name_entry = ctk.CTkEntry(
            inner, placeholder_text="Enter person's name",
            font=Fonts.BODY, fg_color=Colors.BG_INPUT,
            border_color=Colors.BORDER, text_color=Colors.TEXT_PRIMARY,
            height=40, corner_radius=8
        )
        self.name_entry.pack(fill="x", pady=(Spacing.XS, Spacing.MD))

        ctk.CTkLabel(
            inner, text="Notes (optional)", font=Fonts.BODY_BOLD,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w")

        self.notes_entry = ctk.CTkTextbox(
            inner, font=Fonts.BODY, fg_color=Colors.BG_INPUT,
            border_color=Colors.BORDER, text_color=Colors.TEXT_PRIMARY,
            height=80, corner_radius=8
        )
        self.notes_entry.pack(fill="x", pady=(Spacing.XS, Spacing.LG))

        btn_frame = ctk.CTkFrame(inner, fg_color="transparent")
        btn_frame.pack(fill="x")

        ctk.CTkButton(
            btn_frame, text="Cancel",
            font=Fonts.BODY, fg_color=Colors.BG_INPUT,
            hover_color=Colors.BG_HOVER, text_color=Colors.TEXT_SECONDARY,
            height=40, corner_radius=8, width=120,
            command=self._reset_wizard
        ).pack(side="left")

        ctk.CTkButton(
            btn_frame, text="Save Person  \u2713",
            font=Fonts.BODY_BOLD, fg_color=Colors.SUCCESS,
            hover_color="#00b894", text_color=Colors.TEXT_PRIMARY,
            height=40, corner_radius=8, width=180,
            command=self._save_person
        ).pack(side="right")

        self.name_entry.focus_set()

    def _save_person(self):
        name = self.name_entry.get().strip()
        if not name:
            self.toast.show("Please enter a name", "warning")
            return

        notes = self.notes_entry.get("1.0", "end").strip()

        if self._captured_encodings:
            person_id = self.face_store.register_face(
                name, self._captured_encodings,
                self._captured_frames, notes
            )
        else:
            person_id = self.face_store.db.add_person(name, notes)
            for i, frame in enumerate(self._captured_frames):
                import os
                from utils.config import FACES_DIR
                person_dir = os.path.join(FACES_DIR, str(person_id))
                os.makedirs(person_dir, exist_ok=True)
                cv2.imwrite(os.path.join(person_dir, f"face_{i}.jpg"), frame)

        if person_id:
            self.pipeline.reload_faces()
            self.toast.show(f"'{name}' registered successfully!", "success")
            log.info(f"Registered new person: {name} (ID: {person_id})")
        else:
            self.toast.show("Registration failed", "error")

        self._reset_wizard()

    def _reset_wizard(self):
        self._camera_active = False
        self._captured_frames = []
        self._captured_encodings = []
        self._captured_thumbnails = []
        time.sleep(0.1)
        self._show_step_1()

    def cleanup(self):
        self._camera_active = False
        if self._cap and self._cap.isOpened():
            self._cap.release()
