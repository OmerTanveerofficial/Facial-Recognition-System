import queue
import customtkinter as ctk
from gui.theme import Colors, Fonts, Spacing
from gui.camera_view import CameraView, CameraThread
from gui.components.face_card import FaceCard
from utils.logger import log


class DashboardPanel(ctk.CTkFrame):

    def __init__(self, parent, pipeline, toast_manager):
        super().__init__(parent, fg_color=Colors.BG_PRIMARY, corner_radius=0)
        self.pipeline = pipeline
        self.toast = toast_manager
        self._camera_thread = None
        self._frame_queue = queue.Queue(maxsize=3)
        self._face_cards = []
        self._camera_active = False

        self._build_ui()

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=Spacing.PANEL_PADDING, pady=(Spacing.XL, Spacing.MD))

        ctk.CTkLabel(
            header, text="Live Dashboard",
            font=Fonts.HEADING_LG, text_color=Colors.TEXT_PRIMARY
        ).pack(side="left")

        controls = ctk.CTkFrame(header, fg_color="transparent")
        controls.pack(side="right")

        self.start_btn = ctk.CTkButton(
            controls, text="\u25b6  Start Camera",
            font=Fonts.BODY_BOLD, fg_color=Colors.SUCCESS,
            hover_color="#00b894", text_color=Colors.TEXT_PRIMARY,
            corner_radius=8, height=36,
            command=self._toggle_camera
        )
        self.start_btn.pack(side="left", padx=(0, Spacing.SM))

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=Spacing.PANEL_PADDING,
                     pady=(0, Spacing.PANEL_PADDING))

        camera_frame = ctk.CTkFrame(
            content, fg_color=Colors.BG_SECONDARY,
            corner_radius=Spacing.CARD_RADIUS,
            border_color=Colors.BORDER, border_width=1
        )
        camera_frame.pack(side="left", fill="both", expand=True, padx=(0, Spacing.MD))

        self.camera_view = CameraView(camera_frame, self._frame_queue)
        self.camera_view.pack(fill="both", expand=True, padx=4, pady=4)
        self.camera_view.set_frame_callback(self._on_frame_update)

        right_panel = ctk.CTkFrame(content, fg_color="transparent", width=300)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        stats_frame = ctk.CTkFrame(right_panel, fg_color=Colors.BG_SECONDARY,
                                    corner_radius=Spacing.CARD_RADIUS,
                                    border_color=Colors.BORDER, border_width=1)
        stats_frame.pack(fill="x", pady=(0, Spacing.MD))

        stats_inner = ctk.CTkFrame(stats_frame, fg_color="transparent")
        stats_inner.pack(padx=Spacing.CARD_PADDING, pady=Spacing.CARD_PADDING)

        ctk.CTkLabel(
            stats_inner, text="Detection Stats",
            font=Fonts.SUBHEADING, text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w")

        self.stats_faces = ctk.CTkLabel(
            stats_inner, text="Faces: 0",
            font=Fonts.BODY, text_color=Colors.TEXT_SECONDARY
        )
        self.stats_faces.pack(anchor="w", pady=(4, 0))

        self.stats_known = ctk.CTkLabel(
            stats_inner, text="Known: 0  |  Unknown: 0",
            font=Fonts.BODY, text_color=Colors.TEXT_SECONDARY
        )
        self.stats_known.pack(anchor="w")

        self.stats_fps = ctk.CTkLabel(
            stats_inner, text="FPS: --",
            font=Fonts.BODY, text_color=Colors.TEXT_SECONDARY
        )
        self.stats_fps.pack(anchor="w")

        faces_header = ctk.CTkLabel(
            right_panel, text="Detected Faces",
            font=Fonts.SUBHEADING, text_color=Colors.TEXT_PRIMARY
        )
        faces_header.pack(anchor="w", pady=(Spacing.SM, Spacing.SM))

        self.faces_scroll = ctk.CTkScrollableFrame(
            right_panel, fg_color="transparent",
            scrollbar_button_color=Colors.BG_HOVER,
            scrollbar_button_hover_color=Colors.ACCENT
        )
        self.faces_scroll.pack(fill="both", expand=True)

        self.no_faces_label = ctk.CTkLabel(
            self.faces_scroll, text="No faces detected",
            font=Fonts.BODY, text_color=Colors.TEXT_MUTED
        )
        self.no_faces_label.pack(pady=Spacing.XL)

    def _toggle_camera(self):
        if self._camera_active:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        self._camera_thread = CameraThread(
            self.pipeline, self._frame_queue
        )
        self._camera_thread.start()
        self.camera_view.start_polling()
        self._camera_active = True

        self.start_btn.configure(
            text="\u25a0  Stop Camera",
            fg_color=Colors.DANGER,
            hover_color="#c62828"
        )

        if hasattr(self, '_status_callback') and self._status_callback:
            self._status_callback(True)

        self.toast.show("Camera started", "success")
        log.info("Camera started from dashboard")

    def _stop_camera(self):
        if self._camera_thread:
            self._camera_thread.stop()
            self._camera_thread = None

        self.camera_view.stop_polling()
        self.camera_view.show_placeholder()
        self._camera_active = False

        self.start_btn.configure(
            text="\u25b6  Start Camera",
            fg_color=Colors.SUCCESS,
            hover_color="#00b894"
        )

        self._clear_face_cards()

        if hasattr(self, '_status_callback') and self._status_callback:
            self._status_callback(False)

        self.toast.show("Camera stopped", "info")

    def _on_frame_update(self, results, fps):
        known = sum(1 for r in results if r.is_known)
        unknown = len(results) - known
        self.stats_faces.configure(text=f"Faces: {len(results)}")
        self.stats_known.configure(text=f"Known: {known}  |  Unknown: {unknown}")
        self.stats_fps.configure(text=f"FPS: {fps:.1f}")

        if hasattr(self, '_status_update_callback') and self._status_update_callback:
            self._status_update_callback(fps, len(results))

        self._update_face_cards(results)

    def _update_face_cards(self, results):
        if results:
            self.no_faces_label.pack_forget()
        else:
            self._clear_face_cards()
            self.no_faces_label.pack(pady=Spacing.XL)
            return

        while len(self._face_cards) < len(results):
            card = FaceCard(self.faces_scroll)
            card.pack(fill="x", pady=(0, Spacing.SM))
            self._face_cards.append(card)

        while len(self._face_cards) > len(results):
            card = self._face_cards.pop()
            card.destroy()

        for card, result in zip(self._face_cards, results):
            card.update(result)

    def _clear_face_cards(self):
        for card in self._face_cards:
            card.destroy()
        self._face_cards.clear()

    def set_status_callback(self, callback):
        self._status_callback = callback

    def set_status_update_callback(self, callback):
        self._status_update_callback = callback

    def cleanup(self):
        if self._camera_active:
            self._stop_camera()

    @property
    def is_camera_active(self):
        return self._camera_active
