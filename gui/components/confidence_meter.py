import customtkinter as ctk
from gui.theme import Colors, Fonts, Spacing


class ConfidenceMeter(ctk.CTkFrame):

    def __init__(self, parent, label="Confidence", size=80):
        super().__init__(parent, fg_color="transparent")
        self._value = 0.0
        self._size = size

        self.label = ctk.CTkLabel(
            self, text=label,
            font=Fonts.CAPTION, text_color=Colors.TEXT_SECONDARY
        )
        self.label.pack()

        self.progress = ctk.CTkProgressBar(
            self, width=size, height=8,
            fg_color=Colors.BG_INPUT,
            progress_color=Colors.ACCENT,
            corner_radius=4
        )
        self.progress.pack(pady=(4, 2))
        self.progress.set(0)

        self.value_label = ctk.CTkLabel(
            self, text="0%",
            font=Fonts.CAPTION_BOLD, text_color=Colors.TEXT_PRIMARY
        )
        self.value_label.pack()

    def set_value(self, value):
        self._value = max(0.0, min(1.0, value))
        self.progress.set(self._value)
        self.value_label.configure(text=f"{int(self._value * 100)}%")

        if self._value >= 0.7:
            color = Colors.SUCCESS
        elif self._value >= 0.5:
            color = Colors.WARNING
        else:
            color = Colors.DANGER
        self.progress.configure(progress_color=color)
