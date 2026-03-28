import customtkinter as ctk
from gui.theme import Colors, Fonts, Spacing


class SidebarButton(ctk.CTkButton):

    def __init__(self, parent, text, icon, command=None, **kwargs):
        super().__init__(
            parent,
            text=f"  {icon}  {text}",
            font=Fonts.BODY,
            fg_color="transparent",
            hover_color=Colors.BG_HOVER,
            text_color=Colors.TEXT_SECONDARY,
            anchor="w",
            height=44,
            corner_radius=10,
            command=command,
            **kwargs
        )
        self._is_active = False

    def set_active(self, active):
        self._is_active = active
        if active:
            self.configure(
                fg_color=Colors.ACCENT,
                text_color=Colors.TEXT_ON_ACCENT,
                hover_color=Colors.ACCENT_HOVER
            )
        else:
            self.configure(
                fg_color="transparent",
                text_color=Colors.TEXT_SECONDARY,
                hover_color=Colors.BG_HOVER
            )


class Sidebar(ctk.CTkFrame):

    def __init__(self, parent, on_navigate=None):
        super().__init__(
            parent, width=Spacing.SIDEBAR_WIDTH,
            fg_color=Colors.BG_TERTIARY, corner_radius=0
        )
        self.pack_propagate(False)
        self._on_navigate = on_navigate
        self._buttons = {}

        self._build_ui()

    def _build_ui(self):
        logo_frame = ctk.CTkFrame(self, fg_color="transparent")
        logo_frame.pack(fill="x", padx=Spacing.MD, pady=(Spacing.XL, Spacing.SM))

        ctk.CTkLabel(
            logo_frame, text="\U0001f441",
            font=("", 32)
        ).pack(anchor="center")

        ctk.CTkLabel(
            logo_frame, text="FaceVision Pro",
            font=Fonts.HEADING, text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="center", pady=(4, 0))

        ctk.CTkLabel(
            logo_frame, text="Real-Time Recognition",
            font=Fonts.CAPTION, text_color=Colors.TEXT_MUTED
        ).pack(anchor="center")

        sep = ctk.CTkFrame(self, height=1, fg_color=Colors.BORDER)
        sep.pack(fill="x", padx=Spacing.LG, pady=Spacing.MD)

        nav_frame = ctk.CTkFrame(self, fg_color="transparent")
        nav_frame.pack(fill="x", padx=Spacing.SM)

        nav_items = [
            ("dashboard", "\U0001f4ca", "Dashboard"),
            ("register", "\U0001f464", "Register Face"),
            ("gallery", "\U0001f5bc", "Gallery"),
            ("settings", "\u2699", "Settings"),
        ]

        for key, icon, label in nav_items:
            btn = SidebarButton(
                nav_frame, text=label, icon=icon,
                command=lambda k=key: self._navigate(k)
            )
            btn.pack(fill="x", pady=2)
            self._buttons[key] = btn

        self._buttons["dashboard"].set_active(True)

        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(side="bottom", fill="x", padx=Spacing.MD, pady=Spacing.MD)

        ctk.CTkLabel(
            bottom_frame, text="v1.0.0",
            font=Fonts.SMALL, text_color=Colors.TEXT_MUTED
        ).pack(anchor="center")

    def _navigate(self, key):
        for name, btn in self._buttons.items():
            btn.set_active(name == key)

        if self._on_navigate:
            self._on_navigate(key)

    def set_active(self, key):
        for name, btn in self._buttons.items():
            btn.set_active(name == key)
