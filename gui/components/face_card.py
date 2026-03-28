import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
from gui.theme import Colors, Fonts, Spacing, Sizes, EMOTION_COLORS


class FaceCard(ctk.CTkFrame):

    def __init__(self, parent, face_result=None):
        super().__init__(
            parent, fg_color=Colors.BG_SECONDARY,
            border_color=Colors.BORDER, border_width=1,
            corner_radius=Spacing.CARD_RADIUS
        )

        self._face_result = face_result
        self._build_ui()

        if face_result:
            self.update(face_result)

    def _build_ui(self):
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=Spacing.CARD_PADDING,
                     pady=Spacing.CARD_PADDING)

        self.thumbnail_label = ctk.CTkLabel(
            content, text="", width=Sizes.THUMBNAIL_SIZE,
            height=Sizes.THUMBNAIL_SIZE, fg_color=Colors.BG_INPUT,
            corner_radius=8
        )
        self.thumbnail_label.pack(side="left", padx=(0, Spacing.MD))

        info_frame = ctk.CTkFrame(content, fg_color="transparent")
        info_frame.pack(side="left", fill="both", expand=True)

        self.name_label = ctk.CTkLabel(
            info_frame, text="Unknown", font=Fonts.BODY_BOLD,
            text_color=Colors.TEXT_PRIMARY, anchor="w"
        )
        self.name_label.pack(anchor="w")

        conf_frame = ctk.CTkFrame(info_frame, fg_color="transparent", height=20)
        conf_frame.pack(anchor="w", fill="x", pady=(2, 0))

        self.conf_label = ctk.CTkLabel(
            conf_frame, text="Match: --",
            font=Fonts.CAPTION, text_color=Colors.TEXT_SECONDARY, anchor="w"
        )
        self.conf_label.pack(side="left")

        self.conf_bar = ctk.CTkProgressBar(
            conf_frame, width=80, height=Sizes.CONFIDENCE_BAR_HEIGHT,
            fg_color=Colors.BG_INPUT, progress_color=Colors.ACCENT
        )
        self.conf_bar.pack(side="left", padx=(Spacing.SM, 0))
        self.conf_bar.set(0)

        self.emotion_label = ctk.CTkLabel(
            info_frame, text="",
            font=Fonts.CAPTION, text_color=Colors.TEXT_SECONDARY, anchor="w"
        )
        self.emotion_label.pack(anchor="w", pady=(2, 0))

        self.age_gender_label = ctk.CTkLabel(
            info_frame, text="",
            font=Fonts.CAPTION, text_color=Colors.TEXT_SECONDARY, anchor="w"
        )
        self.age_gender_label.pack(anchor="w")

    def update(self, face_result):
        self._face_result = face_result

        if face_result.is_known:
            self.name_label.configure(text=face_result.display_name,
                                      text_color=Colors.SUCCESS)
            self.configure(border_color=Colors.SUCCESS)
        else:
            self.name_label.configure(text="Unknown",
                                      text_color=Colors.WARNING)
            self.configure(border_color=Colors.WARNING)

        if face_result.recognition:
            conf = face_result.recognition.confidence
            self.conf_label.configure(text=f"Match: {int(conf * 100)}%")
            self.conf_bar.set(conf)

            if conf >= 0.7:
                self.conf_bar.configure(progress_color=Colors.SUCCESS)
            elif conf >= 0.5:
                self.conf_bar.configure(progress_color=Colors.WARNING)
            else:
                self.conf_bar.configure(progress_color=Colors.DANGER)

        if face_result.emotion:
            emo = face_result.emotion
            emo_color = EMOTION_COLORS.get(emo.dominant_emotion, Colors.TEXT_SECONDARY)
            self.emotion_label.configure(
                text=f"{emo.emoji} {emo.dominant_emotion} ({int(emo.confidence * 100)}%)",
                text_color=emo_color
            )
        else:
            self.emotion_label.configure(text="")

        if face_result.age_gender:
            ag = face_result.age_gender
            self.age_gender_label.configure(
                text=f"{ag.gender} \u2022 Age: {ag.age_range}"
            )
        else:
            self.age_gender_label.configure(text="")

    def set_thumbnail(self, face_image):
        try:
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img = pil_img.resize(
                (Sizes.THUMBNAIL_SIZE, Sizes.THUMBNAIL_SIZE),
                Image.LANCZOS
            )
            ctk_img = ctk.CTkImage(pil_img, size=(Sizes.THUMBNAIL_SIZE, Sizes.THUMBNAIL_SIZE))
            self.thumbnail_label.configure(image=ctk_img, text="")
            self.thumbnail_label._ctk_image = ctk_img
        except Exception:
            pass
