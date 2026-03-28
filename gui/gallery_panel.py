import cv2
import customtkinter as ctk
from PIL import Image
from gui.theme import Colors, Fonts, Spacing, Sizes
from utils.logger import log


class PersonCard(ctk.CTkFrame):

    def __init__(self, parent, person, face_store, on_delete=None, on_edit=None):
        super().__init__(
            parent, fg_color=Colors.BG_SECONDARY,
            corner_radius=Spacing.CARD_RADIUS,
            border_color=Colors.BORDER, border_width=1,
            width=Sizes.GALLERY_CARD_SIZE, height=Sizes.GALLERY_CARD_SIZE + 60
        )
        self.pack_propagate(False)
        self.person = person
        self.face_store = face_store
        self._on_delete = on_delete
        self._on_edit = on_edit
        self._build_ui()

    def _build_ui(self):
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=Spacing.CARD_PADDING,
                   pady=Spacing.CARD_PADDING)

        thumbnail = self.face_store.get_person_thumbnail(self.person["id"])
        if thumbnail is not None:
            try:
                rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb).resize((80, 80), Image.LANCZOS)
                ctk_img = ctk.CTkImage(pil_img, size=(80, 80))
                img_label = ctk.CTkLabel(
                    inner, text="", image=ctk_img,
                    fg_color=Colors.BG_INPUT, corner_radius=8,
                    width=88, height=88
                )
                img_label.pack(pady=(0, Spacing.SM))
                img_label._ref = ctk_img
            except Exception:
                ctk.CTkLabel(
                    inner, text="\U0001f464", font=("", 40),
                    fg_color=Colors.BG_INPUT, corner_radius=8,
                    width=88, height=88
                ).pack(pady=(0, Spacing.SM))
        else:
            ctk.CTkLabel(
                inner, text="\U0001f464", font=("", 40),
                fg_color=Colors.BG_INPUT, corner_radius=8,
                width=88, height=88
            ).pack(pady=(0, Spacing.SM))

        ctk.CTkLabel(
            inner, text=self.person["name"],
            font=Fonts.BODY_BOLD, text_color=Colors.TEXT_PRIMARY
        ).pack()

        created = self.person.get("created_at", "Unknown")
        if isinstance(created, str) and len(created) > 10:
            created = created[:10]
        ctk.CTkLabel(
            inner, text=f"Added: {created}",
            font=Fonts.CAPTION, text_color=Colors.TEXT_MUTED
        ).pack()

        count = self.face_store.db.get_recognition_count(self.person["id"])
        ctk.CTkLabel(
            inner, text=f"Recognized {count} times",
            font=Fonts.CAPTION, text_color=Colors.TEXT_SECONDARY
        ).pack(pady=(2, 0))

        btn_frame = ctk.CTkFrame(inner, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(Spacing.SM, 0))

        ctk.CTkButton(
            btn_frame, text="Edit",
            font=Fonts.CAPTION, fg_color=Colors.BG_INPUT,
            hover_color=Colors.BG_HOVER, text_color=Colors.TEXT_SECONDARY,
            height=28, corner_radius=6, width=70,
            command=lambda: self._on_edit(self.person) if self._on_edit else None
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            btn_frame, text="Delete",
            font=Fonts.CAPTION, fg_color=Colors.DANGER,
            hover_color="#c62828", text_color=Colors.TEXT_PRIMARY,
            height=28, corner_radius=6, width=70,
            command=lambda: self._confirm_delete()
        ).pack(side="right")

    def _confirm_delete(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Confirm Delete")
        dialog.geometry("400x180")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 400) // 2
        y = (dialog.winfo_screenheight() - 180) // 2
        dialog.geometry(f"+{x}+{y}")

        frame = ctk.CTkFrame(dialog, fg_color=Colors.BG_PRIMARY)
        frame.pack(fill="both", expand=True)

        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.pack(expand=True, padx=Spacing.XL, pady=Spacing.XL)

        ctk.CTkLabel(
            inner, text=f"Delete '{self.person['name']}'?",
            font=Fonts.SUBHEADING, text_color=Colors.TEXT_PRIMARY
        ).pack()

        ctk.CTkLabel(
            inner, text="This will remove all their face data permanently.",
            font=Fonts.BODY, text_color=Colors.TEXT_SECONDARY
        ).pack(pady=(Spacing.SM, Spacing.LG))

        btn_frame = ctk.CTkFrame(inner, fg_color="transparent")
        btn_frame.pack(fill="x")

        ctk.CTkButton(
            btn_frame, text="Cancel",
            fg_color=Colors.BG_INPUT, hover_color=Colors.BG_HOVER,
            text_color=Colors.TEXT_SECONDARY, height=36,
            corner_radius=8, width=120,
            command=dialog.destroy
        ).pack(side="left")

        def do_delete():
            dialog.destroy()
            if self._on_delete:
                self._on_delete(self.person["id"])

        ctk.CTkButton(
            btn_frame, text="Delete",
            fg_color=Colors.DANGER, hover_color="#c62828",
            text_color=Colors.TEXT_PRIMARY, height=36,
            corner_radius=8, width=120,
            command=do_delete
        ).pack(side="right")


class GalleryPanel(ctk.CTkFrame):

    def __init__(self, parent, face_store, pipeline, toast_manager):
        super().__init__(parent, fg_color=Colors.BG_PRIMARY, corner_radius=0)
        self.face_store = face_store
        self.pipeline = pipeline
        self.toast = toast_manager

        self._build_ui()

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=Spacing.PANEL_PADDING, pady=(Spacing.XL, Spacing.MD))

        ctk.CTkLabel(
            header, text="Face Gallery",
            font=Fonts.HEADING_LG, text_color=Colors.TEXT_PRIMARY
        ).pack(side="left")

        self.count_label = ctk.CTkLabel(
            header, text="0 persons registered",
            font=Fonts.BODY, text_color=Colors.TEXT_SECONDARY
        )
        self.count_label.pack(side="left", padx=Spacing.LG)

        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.pack(fill="x", padx=Spacing.PANEL_PADDING, pady=(0, Spacing.MD))

        self.search_entry = ctk.CTkEntry(
            search_frame, placeholder_text="\U0001f50d  Search by name...",
            font=Fonts.BODY, fg_color=Colors.BG_INPUT,
            border_color=Colors.BORDER, text_color=Colors.TEXT_PRIMARY,
            height=40, corner_radius=8, width=300
        )
        self.search_entry.pack(side="left")
        self.search_entry.bind("<KeyRelease>", lambda e: self._filter_gallery())

        ctk.CTkButton(
            search_frame, text="Refresh",
            font=Fonts.BODY, fg_color=Colors.ACCENT,
            hover_color=Colors.ACCENT_HOVER, height=40,
            corner_radius=8, width=100,
            command=self.refresh
        ).pack(side="left", padx=Spacing.SM)

        self.gallery_scroll = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
            scrollbar_button_color=Colors.BG_HOVER,
            scrollbar_button_hover_color=Colors.ACCENT
        )
        self.gallery_scroll.pack(fill="both", expand=True,
                                  padx=Spacing.PANEL_PADDING,
                                  pady=(0, Spacing.PANEL_PADDING))

        self.gallery_scroll.columnconfigure((0, 1, 2), weight=1)

        self.no_persons_label = ctk.CTkLabel(
            self.gallery_scroll, text="No registered faces yet.\nGo to 'Register Face' to add someone.",
            font=Fonts.BODY, text_color=Colors.TEXT_MUTED
        )

    def refresh(self):
        for widget in self.gallery_scroll.winfo_children():
            widget.destroy()

        persons = self.face_store.db.get_all_persons()
        self.count_label.configure(text=f"{len(persons)} persons registered")

        if not persons:
            self.no_persons_label = ctk.CTkLabel(
                self.gallery_scroll,
                text="No registered faces yet.\nGo to 'Register Face' to add someone.",
                font=Fonts.BODY, text_color=Colors.TEXT_MUTED
            )
            self.no_persons_label.grid(row=0, column=0, columnspan=3, pady=Spacing.XXL)
            return

        for i, person in enumerate(persons):
            row = i // 3
            col = i % 3
            card = PersonCard(
                self.gallery_scroll, person, self.face_store,
                on_delete=self._delete_person,
                on_edit=self._edit_person
            )
            card.grid(row=row, column=col, padx=Spacing.SM, pady=Spacing.SM, sticky="nsew")

    def _filter_gallery(self):
        query = self.search_entry.get().strip()
        if not query:
            self.refresh()
            return

        for widget in self.gallery_scroll.winfo_children():
            widget.destroy()

        persons = self.face_store.db.search_persons(query)
        self.count_label.configure(text=f"{len(persons)} results")

        for i, person in enumerate(persons):
            row = i // 3
            col = i % 3
            card = PersonCard(
                self.gallery_scroll, person, self.face_store,
                on_delete=self._delete_person,
                on_edit=self._edit_person
            )
            card.grid(row=row, column=col, padx=Spacing.SM, pady=Spacing.SM, sticky="nsew")

    def _delete_person(self, person_id):
        self.face_store.delete_person(person_id)
        self.pipeline.reload_faces()
        self.toast.show("Person deleted", "info")
        self.refresh()

    def _edit_person(self, person):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Edit Person")
        dialog.geometry("400x250")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 400) // 2
        y = (dialog.winfo_screenheight() - 250) // 2
        dialog.geometry(f"+{x}+{y}")

        frame = ctk.CTkFrame(dialog, fg_color=Colors.BG_PRIMARY)
        frame.pack(fill="both", expand=True)

        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.pack(expand=True, padx=Spacing.XL, pady=Spacing.XL)

        ctk.CTkLabel(
            inner, text="Edit Person",
            font=Fonts.SUBHEADING, text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w")

        ctk.CTkLabel(
            inner, text="Name", font=Fonts.BODY,
            text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w", pady=(Spacing.MD, Spacing.XS))

        name_entry = ctk.CTkEntry(
            inner, font=Fonts.BODY, fg_color=Colors.BG_INPUT,
            border_color=Colors.BORDER, text_color=Colors.TEXT_PRIMARY,
            height=40, corner_radius=8
        )
        name_entry.pack(fill="x")
        name_entry.insert(0, person["name"])

        ctk.CTkLabel(
            inner, text="Notes", font=Fonts.BODY,
            text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w", pady=(Spacing.SM, Spacing.XS))

        notes_entry = ctk.CTkEntry(
            inner, font=Fonts.BODY, fg_color=Colors.BG_INPUT,
            border_color=Colors.BORDER, text_color=Colors.TEXT_PRIMARY,
            height=40, corner_radius=8
        )
        notes_entry.pack(fill="x")
        notes_entry.insert(0, person.get("notes", ""))

        def save():
            new_name = name_entry.get().strip()
            new_notes = notes_entry.get().strip()
            if new_name:
                self.face_store.db.update_person(person["id"], new_name, new_notes)
                self.pipeline.reload_faces()
                self.toast.show(f"Updated '{new_name}'", "success")
                dialog.destroy()
                self.refresh()

        btn_frame = ctk.CTkFrame(inner, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(Spacing.LG, 0))

        ctk.CTkButton(
            btn_frame, text="Cancel",
            fg_color=Colors.BG_INPUT, hover_color=Colors.BG_HOVER,
            text_color=Colors.TEXT_SECONDARY, height=36, corner_radius=8,
            command=dialog.destroy
        ).pack(side="left")

        ctk.CTkButton(
            btn_frame, text="Save",
            fg_color=Colors.SUCCESS, hover_color="#00b894",
            text_color=Colors.TEXT_PRIMARY, height=36, corner_radius=8,
            command=save
        ).pack(side="right")
