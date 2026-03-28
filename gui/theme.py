class Colors:
    BG_PRIMARY = "#0f0f1a"
    BG_SECONDARY = "#1a1a2e"
    BG_TERTIARY = "#16213e"
    BG_INPUT = "#252540"
    BG_HOVER = "#2a2a4a"

    ACCENT = "#6c63ff"
    ACCENT_HOVER = "#7c74ff"
    ACCENT_LIGHT = "#8b83ff"

    SUCCESS = "#00d4aa"
    WARNING = "#ffa726"
    DANGER = "#ef5350"
    INFO = "#42a5f5"

    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#b0b0c8"
    TEXT_MUTED = "#6c6c88"
    TEXT_ON_ACCENT = "#ffffff"

    BORDER = "#2a2a4a"
    BORDER_LIGHT = "#3a3a5a"

    FACE_KNOWN = (0, 212, 170)
    FACE_UNKNOWN = (38, 167, 255)
    FACE_ERROR = (80, 83, 239)

    STATUS_ONLINE = "#00d4aa"
    STATUS_OFFLINE = "#ef5350"
    STATUS_PROCESSING = "#ffa726"


class Fonts:
    FAMILY = "Segoe UI"
    FAMILY_MONO = "Cascadia Code"

    HEADING_XL = (FAMILY, 28, "bold")
    HEADING_LG = (FAMILY, 22, "bold")
    HEADING = (FAMILY, 18, "bold")
    SUBHEADING = (FAMILY, 15, "bold")
    BODY = (FAMILY, 13)
    BODY_BOLD = (FAMILY, 13, "bold")
    CAPTION = (FAMILY, 11)
    CAPTION_BOLD = (FAMILY, 11, "bold")
    SMALL = (FAMILY, 10)
    MONO = (FAMILY_MONO, 12)
    EMOJI = (FAMILY, 20)


class Spacing:
    XS = 4
    SM = 8
    MD = 12
    LG = 16
    XL = 24
    XXL = 32

    SIDEBAR_WIDTH = 220
    PANEL_PADDING = 24
    CARD_PADDING = 16
    CARD_RADIUS = 12
    BUTTON_PADDING_X = 20
    BUTTON_PADDING_Y = 10
    INPUT_PADDING = 12


class Sizes:
    SIDEBAR_ICON = 20
    FACE_CARD_WIDTH = 280
    FACE_CARD_HEIGHT = 100
    THUMBNAIL_SIZE = 64
    GALLERY_CARD_SIZE = 200
    STATUS_DOT = 8
    CONFIDENCE_BAR_HEIGHT = 6
    CAMERA_OVERLAY_OPACITY = 0.7


EMOTION_COLORS = {
    "Happy": Colors.SUCCESS,
    "Sad": Colors.INFO,
    "Angry": Colors.DANGER,
    "Fear": Colors.WARNING,
    "Surprise": "#ce93d8",
    "Disgust": "#a1887f",
    "Neutral": Colors.TEXT_SECONDARY,
}
