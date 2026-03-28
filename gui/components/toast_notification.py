import customtkinter as ctk
from gui.theme import Colors, Fonts, Spacing


class ToastNotification(ctk.CTkFrame):

    TYPES = {
        "success": {"bg": "#1b5e20", "border": Colors.SUCCESS, "icon": "\u2713"},
        "error": {"bg": "#b71c1c", "border": Colors.DANGER, "icon": "\u2717"},
        "warning": {"bg": "#e65100", "border": Colors.WARNING, "icon": "\u26a0"},
        "info": {"bg": "#0d47a1", "border": Colors.INFO, "icon": "\u2139"},
    }

    def __init__(self, parent, message, toast_type="info", duration=3000):
        config = self.TYPES.get(toast_type, self.TYPES["info"])
        super().__init__(
            parent, fg_color=config["bg"],
            border_color=config["border"], border_width=2,
            corner_radius=10
        )

        self.duration = duration
        self._parent = parent

        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(padx=Spacing.MD, pady=Spacing.SM, fill="x")

        icon_label = ctk.CTkLabel(
            content_frame, text=config["icon"],
            font=("", 16), text_color=Colors.TEXT_PRIMARY, width=24
        )
        icon_label.pack(side="left", padx=(0, Spacing.SM))

        msg_label = ctk.CTkLabel(
            content_frame, text=message,
            font=Fonts.BODY, text_color=Colors.TEXT_PRIMARY,
            anchor="w"
        )
        msg_label.pack(side="left", fill="x", expand=True)

        close_btn = ctk.CTkButton(
            content_frame, text="\u2715", width=24, height=24,
            font=("", 12), fg_color="transparent",
            hover_color=Colors.BG_HOVER,
            text_color=Colors.TEXT_SECONDARY,
            command=self._dismiss
        )
        close_btn.pack(side="right")

        self.place(relx=1.0, y=10, anchor="ne", x=-10)
        self.after(duration, self._dismiss)

    def _dismiss(self):
        try:
            self.destroy()
        except Exception:
            pass


class ToastManager:

    def __init__(self, parent):
        self.parent = parent
        self._toasts = []

    def show(self, message, toast_type="info", duration=3000):
        toast = ToastNotification(self.parent, message, toast_type, duration)
        self._toasts.append(toast)

        self._reposition()

        self.parent.after(duration + 100, lambda: self._remove(toast))

    def _reposition(self):
        y_offset = 10
        for toast in self._toasts:
            try:
                toast.place_configure(y=y_offset)
                y_offset += 60
            except Exception:
                pass

    def _remove(self, toast):
        if toast in self._toasts:
            self._toasts.remove(toast)
            self._reposition()
