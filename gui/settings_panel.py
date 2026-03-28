import threading
import customtkinter as ctk
from gui.theme import Colors, Fonts, Spacing
from utils.config import (
    DETECTION_CONFIDENCE, RECOGNITION_TOLERANCE, ANALYSIS_INTERVAL,
    CAMERA_INDEX
)
from utils.model_downloader import download_all_models, models_available, check_models
from utils.logger import log


class SettingsPanel(ctk.CTkFrame):

    def __init__(self, parent, pipeline, toast_manager):
        super().__init__(parent, fg_color=Colors.BG_PRIMARY, corner_radius=0)
        self.pipeline = pipeline
        self.toast = toast_manager

        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(
            self, text="Settings",
            font=Fonts.HEADING_LG, text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", padx=Spacing.PANEL_PADDING, pady=(Spacing.XL, Spacing.MD))

        scroll = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
            scrollbar_button_color=Colors.BG_HOVER,
            scrollbar_button_hover_color=Colors.ACCENT
        )
        scroll.pack(fill="both", expand=True,
                    padx=Spacing.PANEL_PADDING, pady=(0, Spacing.PANEL_PADDING))

        self._section_header(scroll, "Face Detection")

        conf_frame = self._setting_row(scroll, "Detection Confidence",
                                        "Minimum confidence to detect a face")
        self.conf_slider = ctk.CTkSlider(
            conf_frame, from_=0.1, to=1.0,
            number_of_steps=18,
            fg_color=Colors.BG_INPUT, progress_color=Colors.ACCENT,
            button_color=Colors.ACCENT, button_hover_color=Colors.ACCENT_HOVER,
            width=250
        )
        self.conf_slider.set(DETECTION_CONFIDENCE)
        self.conf_slider.pack(side="left")

        self.conf_value = ctk.CTkLabel(
            conf_frame, text=f"{DETECTION_CONFIDENCE:.0%}",
            font=Fonts.BODY_BOLD, text_color=Colors.ACCENT, width=50
        )
        self.conf_value.pack(side="left", padx=Spacing.SM)

        self.conf_slider.configure(command=self._on_conf_change)

        self._section_header(scroll, "Face Recognition")

        tol_frame = self._setting_row(scroll, "Recognition Tolerance",
                                       "Lower = stricter matching (more accurate but may miss)")
        self.tol_slider = ctk.CTkSlider(
            tol_frame, from_=0.1, to=1.0,
            number_of_steps=18,
            fg_color=Colors.BG_INPUT, progress_color=Colors.ACCENT,
            button_color=Colors.ACCENT, button_hover_color=Colors.ACCENT_HOVER,
            width=250
        )
        self.tol_slider.set(RECOGNITION_TOLERANCE)
        self.tol_slider.pack(side="left")

        self.tol_value = ctk.CTkLabel(
            tol_frame, text=f"{RECOGNITION_TOLERANCE:.2f}",
            font=Fonts.BODY_BOLD, text_color=Colors.ACCENT, width=50
        )
        self.tol_value.pack(side="left", padx=Spacing.SM)

        self.tol_slider.configure(command=self._on_tol_change)

        self._section_header(scroll, "Analysis Pipeline")

        interval_frame = self._setting_row(scroll, "Analysis Interval",
                                            "Full analysis every N frames (higher = faster, less accurate)")
        self.interval_slider = ctk.CTkSlider(
            interval_frame, from_=1, to=10,
            number_of_steps=9,
            fg_color=Colors.BG_INPUT, progress_color=Colors.ACCENT,
            button_color=Colors.ACCENT, button_hover_color=Colors.ACCENT_HOVER,
            width=250
        )
        self.interval_slider.set(ANALYSIS_INTERVAL)
        self.interval_slider.pack(side="left")

        self.interval_value = ctk.CTkLabel(
            interval_frame, text=f"{ANALYSIS_INTERVAL}",
            font=Fonts.BODY_BOLD, text_color=Colors.ACCENT, width=50
        )
        self.interval_value.pack(side="left", padx=Spacing.SM)

        self.interval_slider.configure(command=self._on_interval_change)

        self._section_header(scroll, "Feature Toggles")

        emo_frame = self._setting_row(scroll, "Emotion Detection",
                                       "Detect facial emotions (requires emotion model)")
        self.emotion_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            emo_frame, text="", variable=self.emotion_var,
            fg_color=Colors.BG_INPUT, progress_color=Colors.SUCCESS,
            button_color=Colors.TEXT_PRIMARY,
            button_hover_color=Colors.TEXT_SECONDARY,
            command=self._on_emotion_toggle
        ).pack(side="left")

        ag_frame = self._setting_row(scroll, "Age & Gender Estimation",
                                      "Estimate age range and gender")
        self.age_gender_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            ag_frame, text="", variable=self.age_gender_var,
            fg_color=Colors.BG_INPUT, progress_color=Colors.SUCCESS,
            button_color=Colors.TEXT_PRIMARY,
            button_hover_color=Colors.TEXT_SECONDARY,
            command=self._on_age_gender_toggle
        ).pack(side="left")

        self._section_header(scroll, "Camera")

        cam_frame = self._setting_row(scroll, "Camera Source",
                                       "Select which camera to use")
        self.camera_var = ctk.StringVar(value=f"Camera {CAMERA_INDEX}")
        ctk.CTkOptionMenu(
            cam_frame, values=["Camera 0", "Camera 1", "Camera 2"],
            variable=self.camera_var,
            font=Fonts.BODY, fg_color=Colors.BG_INPUT,
            button_color=Colors.ACCENT, button_hover_color=Colors.ACCENT_HOVER,
            dropdown_fg_color=Colors.BG_SECONDARY,
            dropdown_hover_color=Colors.BG_HOVER,
            dropdown_text_color=Colors.TEXT_PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=200, height=36, corner_radius=8
        ).pack(side="left")

        self._section_header(scroll, "Models")

        model_frame = self._setting_row(scroll, "Pre-trained Models",
                                         "Download required AI models for all features")

        model_status = "All models ready" if models_available() else "Some models missing"
        status_color = Colors.SUCCESS if models_available() else Colors.WARNING

        self.model_status_label = ctk.CTkLabel(
            model_frame, text=model_status,
            font=Fonts.BODY, text_color=status_color
        )
        self.model_status_label.pack(side="left", padx=(0, Spacing.MD))

        self.download_btn = ctk.CTkButton(
            model_frame, text="Download Models",
            font=Fonts.BODY, fg_color=Colors.ACCENT,
            hover_color=Colors.ACCENT_HOVER, height=36,
            corner_radius=8, width=160,
            command=self._download_models
        )
        self.download_btn.pack(side="left")

        self.download_progress = ctk.CTkProgressBar(
            model_frame, width=200, height=6,
            fg_color=Colors.BG_INPUT, progress_color=Colors.ACCENT
        )

        self._section_header(scroll, "System Status")

        status = self.pipeline.status
        status_items = [
            ("Face Detector", status["detector"]),
            ("Face Recognizer", status["recognizer"]),
            ("Emotion Analyzer", status["emotion"]),
            ("Age/Gender Estimator", status["age_gender"]),
        ]

        for name, is_ready in status_items:
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)

            dot_color = Colors.SUCCESS if is_ready else Colors.DANGER
            ctk.CTkLabel(
                row, text="\u25cf", font=("", 10),
                text_color=dot_color, width=20
            ).pack(side="left")

            ctk.CTkLabel(
                row, text=name, font=Fonts.BODY,
                text_color=Colors.TEXT_PRIMARY
            ).pack(side="left")

            state_text = "Ready" if is_ready else "Not Available"
            ctk.CTkLabel(
                row, text=state_text, font=Fonts.CAPTION,
                text_color=Colors.TEXT_MUTED
            ).pack(side="right")

        registered_row = ctk.CTkFrame(scroll, fg_color="transparent")
        registered_row.pack(fill="x", pady=2)
        ctk.CTkLabel(
            registered_row, text="\u25cf", font=("", 10),
            text_color=Colors.INFO, width=20
        ).pack(side="left")
        ctk.CTkLabel(
            registered_row, text="Registered Faces",
            font=Fonts.BODY, text_color=Colors.TEXT_PRIMARY
        ).pack(side="left")
        ctk.CTkLabel(
            registered_row, text=str(status["registered_faces"]),
            font=Fonts.BODY_BOLD, text_color=Colors.ACCENT
        ).pack(side="right")

    def _section_header(self, parent, title):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(Spacing.LG, Spacing.SM))

        ctk.CTkLabel(
            frame, text=title,
            font=Fonts.SUBHEADING, text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w")

        sep = ctk.CTkFrame(frame, height=1, fg_color=Colors.BORDER)
        sep.pack(fill="x", pady=(Spacing.XS, 0))

    def _setting_row(self, parent, label, description):
        card = ctk.CTkFrame(parent, fg_color=Colors.BG_SECONDARY,
                            corner_radius=8, border_color=Colors.BORDER,
                            border_width=1)
        card.pack(fill="x", pady=(Spacing.XS, 0))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=Spacing.CARD_PADDING, pady=Spacing.SM)

        text_frame = ctk.CTkFrame(inner, fg_color="transparent")
        text_frame.pack(fill="x")

        ctk.CTkLabel(
            text_frame, text=label,
            font=Fonts.BODY_BOLD, text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w")

        ctk.CTkLabel(
            text_frame, text=description,
            font=Fonts.CAPTION, text_color=Colors.TEXT_MUTED
        ).pack(anchor="w")

        control_frame = ctk.CTkFrame(inner, fg_color="transparent")
        control_frame.pack(fill="x", pady=(Spacing.SM, 0))

        return control_frame

    def _on_conf_change(self, value):
        self.conf_value.configure(text=f"{value:.0%}")
        self.pipeline.set_detection_confidence(value)

    def _on_tol_change(self, value):
        self.tol_value.configure(text=f"{value:.2f}")
        self.pipeline.set_recognition_tolerance(value)

    def _on_interval_change(self, value):
        val = int(value)
        self.interval_value.configure(text=str(val))
        self.pipeline.set_analysis_interval(val)

    def _on_emotion_toggle(self):
        self.pipeline.toggle_emotion(self.emotion_var.get())
        state = "enabled" if self.emotion_var.get() else "disabled"
        self.toast.show(f"Emotion detection {state}", "info")

    def _on_age_gender_toggle(self):
        self.pipeline.toggle_age_gender(self.age_gender_var.get())
        state = "enabled" if self.age_gender_var.get() else "disabled"
        self.toast.show(f"Age/gender estimation {state}", "info")

    def _download_models(self):
        self.download_btn.configure(state="disabled", text="Downloading...")
        self.download_progress.pack(side="left", padx=Spacing.SM)
        self.download_progress.set(0)

        def callback(status_msg, progress):
            self.after(0, lambda: self._update_download_progress(status_msg, progress))

        def do_download():
            success = download_all_models(callback)
            self.after(0, lambda: self._download_complete(success))

        threading.Thread(target=do_download, daemon=True).start()

    def _update_download_progress(self, status, progress):
        try:
            self.download_progress.set(progress)
            self.model_status_label.configure(text=status)
        except Exception:
            pass

    def _download_complete(self, success):
        self.download_btn.configure(state="normal", text="Download Models")
        self.download_progress.pack_forget()

        if success:
            self.model_status_label.configure(
                text="All models ready", text_color=Colors.SUCCESS
            )
            self.toast.show("All models downloaded successfully!", "success")
        else:
            self.model_status_label.configure(
                text="Some downloads failed", text_color=Colors.DANGER
            )
            self.toast.show("Some model downloads failed", "error")
