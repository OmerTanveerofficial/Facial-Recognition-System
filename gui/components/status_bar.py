import customtkinter as ctk
from gui.theme import Colors, Fonts, Spacing


class StatusBar(ctk.CTkFrame):

    def __init__(self, parent):
        super().__init__(parent, height=36, fg_color=Colors.BG_TERTIARY, corner_radius=0)
        self.pack_propagate(False)

        left_frame = ctk.CTkFrame(self, fg_color="transparent")
        left_frame.pack(side="left", padx=Spacing.MD)

        self.camera_dot = ctk.CTkLabel(
            left_frame, text="\u25cf", font=("", 12),
            text_color=Colors.STATUS_OFFLINE, width=16
        )
        self.camera_dot.pack(side="left")

        self.camera_label = ctk.CTkLabel(
            left_frame, text="Camera: Off",
            font=Fonts.CAPTION, text_color=Colors.TEXT_SECONDARY
        )
        self.camera_label.pack(side="left", padx=(4, Spacing.LG))

        self.fps_label = ctk.CTkLabel(
            left_frame, text="FPS: --",
            font=Fonts.CAPTION, text_color=Colors.TEXT_SECONDARY
        )
        self.fps_label.pack(side="left", padx=(0, Spacing.LG))

        self.face_count_label = ctk.CTkLabel(
            self, text="Faces Detected: 0",
            font=Fonts.CAPTION_BOLD, text_color=Colors.TEXT_SECONDARY
        )
        self.face_count_label.pack(side="left", expand=True)

        right_frame = ctk.CTkFrame(self, fg_color="transparent")
        right_frame.pack(side="right", padx=Spacing.MD)

        self.db_label = ctk.CTkLabel(
            right_frame, text="Registered: 0",
            font=Fonts.CAPTION, text_color=Colors.TEXT_SECONDARY
        )
        self.db_label.pack(side="right")

    def update_camera_status(self, is_active):
        color = Colors.STATUS_ONLINE if is_active else Colors.STATUS_OFFLINE
        text = "Camera: On" if is_active else "Camera: Off"
        self.camera_dot.configure(text_color=color)
        self.camera_label.configure(text=text)

    def update_fps(self, fps):
        color = Colors.SUCCESS if fps >= 20 else (Colors.WARNING if fps >= 10 else Colors.DANGER)
        self.fps_label.configure(text=f"FPS: {fps:.0f}", text_color=color)

    def update_face_count(self, count):
        self.face_count_label.configure(text=f"Faces Detected: {count}")

    def update_db_count(self, count):
        self.db_label.configure(text=f"Registered: {count}")
