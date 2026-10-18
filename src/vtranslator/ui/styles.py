"""UI stylesheet and design tokens for vTranslate Studio.

Restrained dark theme for a focused desktop editing workspace.
All widget styling is centralized here — no inline setStyleSheet in widgets.
"""

# ─── Design Tokens ──────────────────────────────────────────────────
_BG_BASE = "#0E0F12"        # Main application background
_BG_SURFACE = "#15161C"     # Panels, tabs, left sidebar
_BG_ELEVATED = "#1C1E26"    # Cards, dialogs, headers
_BG_OVERLAY = "#242732"     # Hover, popups, dropdown items
_BG_INPUT = "#181A22"       # Inputs, search boxes

_BORDER_DEFAULT = "#2A2D3A" # Crisp subtle dark borders
_BORDER_MUTED = "#1F212B"

_FG_DEFAULT = "#EAEAEA"
_FG_MUTED = "#8E92A4"
_FG_SUBTLE = "#636779"
_FG_ON_EMPHASIS = "#FFFFFF"

_ACCENT_CYAN = "#78A6FF"
_ACCENT_CYAN_HOVER = "#9AB9FF"
_ACCENT_CYAN_DARK = "#416FC9"
_ACCENT_BLUE = "#416FC9"
_ACCENT_GREEN = "#4CB98A"
_DANGER_RED = "#FF3B30"
_WARNING_YELLOW = "#FFB300"

_RADIUS_SM = "4px"
_RADIUS_MD = "6px"
_RADIUS_LG = "8px"


DARK_THEME_QSS = f"""
/* ═══════════════════════════════════════════════════════════
   GLOBAL
   ═══════════════════════════════════════════════════════════ */

QMainWindow {{
    background-color: {_BG_BASE};
}}

QWidget {{
    color: {_FG_DEFAULT};
    font-family: -apple-system, "SF Pro Display", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}

QToolTip {{
    background-color: {_BG_ELEVATED};
    color: {_FG_ON_EMPHASIS};
    border: 1px solid {_BORDER_DEFAULT};
    border-radius: {_RADIUS_SM};
    padding: 5px 9px;
    font-size: 12px;
}}


/* ═══════════════════════════════════════════════════════════
   WORKSPACE HEADER
   ═══════════════════════════════════════════════════════════ */

QWidget#topBar {{
    background-color: {_BG_SURFACE};
    border-bottom: 1px solid {_BORDER_DEFAULT};
}}

QLabel#logoLabel {{
    font-size: 16px;
    font-weight: 800;
    color: {_ACCENT_CYAN};
    letter-spacing: 0.5px;
}}

QLabel#videoBadge {{
    background-color: {_BG_ELEVATED};
    color: #D0D4E4;
    border: 1px solid {_BORDER_DEFAULT};
    border-radius: {_RADIUS_MD};
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 600;
}}


/* ═══════════════════════════════════════════════════════════
   MENU BAR & MENUS
   ═══════════════════════════════════════════════════════════ */

QMenuBar {{
    background-color: {_BG_SURFACE};
    color: #C2C6D6;
    border-bottom: 1px solid {_BORDER_MUTED};
    padding: 2px 6px;
}}
QMenuBar::item:selected {{
    background-color: {_BG_ELEVATED};
    color: {_ACCENT_CYAN};
    border-radius: {_RADIUS_SM};
}}
QMenu {{
    background-color: {_BG_ELEVATED};
    color: {_FG_DEFAULT};
    border: 1px solid {_BORDER_DEFAULT};
    padding: 4px;
    border-radius: {_RADIUS_MD};
}}
QMenu::item {{
    padding: 6px 20px;
    border-radius: {_RADIUS_SM};
}}
QMenu::item:selected {{
    background-color: {_ACCENT_CYAN_DARK};
    color: {_FG_ON_EMPHASIS};
}}
QMenu::separator {{
    height: 1px;
    background: {_BORDER_DEFAULT};
    margin: 4px 8px;
}}


/* ═══════════════════════════════════════════════════════════
   TABS
   ═══════════════════════════════════════════════════════════ */

QTabWidget::pane {{
    border: 1px solid {_BORDER_DEFAULT};
    background-color: {_BG_SURFACE};
    border-radius: {_RADIUS_MD};
    top: -1px;
}}
QTabBar::tab {{
    background: {_BG_BASE};
    color: {_FG_MUTED};
    padding: 8px 18px;
    font-weight: 600;
    font-size: 12px;
    border-top-left-radius: {_RADIUS_MD};
    border-top-right-radius: {_RADIUS_MD};
    margin-right: 4px;
    border: 1px solid transparent;
}}
QTabBar::tab:selected {{
    background: {_BG_SURFACE};
    color: {_ACCENT_CYAN};
    border: 1px solid {_BORDER_DEFAULT};
    border-bottom: 2px solid {_ACCENT_CYAN};
}}
QTabBar::tab:hover:!selected {{
    background: {_BG_ELEVATED};
    color: {_FG_DEFAULT};
}}


/* ═══════════════════════════════════════════════════════════
   PUSH BUTTONS
   ═══════════════════════════════════════════════════════════ */

QPushButton {{
    background-color: #20222B;
    border: 1px solid #323543;
    border-radius: {_RADIUS_MD};
    padding: 6px 14px;
    font-weight: 600;
    color: #D6DAEA;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: #2A2D3A;
    border-color: #404557;
    color: {_FG_ON_EMPHASIS};
}}
QPushButton:pressed {{
    background-color: #171820;
}}
QPushButton:disabled {{
    background-color: #171820;
    color: #4B4E5D;
    border-color: #242632;
}}
QPushButton:checked {{
    background-color: {_ACCENT_CYAN_DARK};
    color: {_FG_ON_EMPHASIS};
    border-color: {_ACCENT_CYAN};
}}

/* Named button variants */
QPushButton#primaryButton {{
    background-color: {_ACCENT_CYAN_DARK};
    border-color: {_ACCENT_CYAN_DARK};
    color: #FFFFFF;
    font-weight: 650;
}}
QPushButton#primaryButton:hover {{
    background-color: #527FD8;
    border-color: #527FD8;
}}
QPushButton#secondaryButton {{
    background-color: #242832;
    border-color: {_BORDER_DEFAULT};
    color: {_FG_DEFAULT};
}}
QPushButton#accentButton {{
    background-color: rgba(120, 166, 255, 0.12);
    border: 1px solid {_ACCENT_CYAN};
    color: {_ACCENT_CYAN};
    font-weight: 600;
}}
QPushButton#accentButton:hover {{
    background-color: rgba(120, 166, 255, 0.22);
    color: #FFFFFF;
}}
QPushButton#dangerButton {{
    background-color: rgba(255, 59, 48, 0.18);
    border: 1px solid {_DANGER_RED};
    color: #FF5A52;
}}
QPushButton#dangerButton:hover {{
    background-color: {_DANGER_RED};
    color: #FFFFFF;
}}


/* ═══════════════════════════════════════════════════════════
   TABLES
   ═══════════════════════════════════════════════════════════ */

QTableWidget {{
    background-color: {_BG_SURFACE};
    border: 1px solid {_BORDER_DEFAULT};
    gridline-color: {_BORDER_MUTED};
    border-radius: {_RADIUS_MD};
    selection-background-color: rgba(0, 229, 255, 0.22);
    selection-color: {_FG_ON_EMPHASIS};
    alternate-background-color: #121319;
}}
QHeaderView::section {{
    background-color: {_BG_ELEVATED};
    color: {_FG_MUTED};
    font-weight: 700;
    font-size: 11px;
    padding: 6px 8px;
    border: none;
    border-right: 1px solid {_BORDER_MUTED};
    border-bottom: 1px solid {_BORDER_DEFAULT};
}}


/* ═══════════════════════════════════════════════════════════
   TEXT INPUTS
   ═══════════════════════════════════════════════════════════ */

QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {_BG_INPUT};
    border: 1px solid {_BORDER_DEFAULT};
    border-radius: {_RADIUS_MD};
    padding: 6px 10px;
    color: {_FG_DEFAULT};
    font-size: 13px;
    selection-background-color: {_ACCENT_CYAN_DARK};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid {_ACCENT_CYAN};
}}
QLineEdit:disabled, QTextEdit:disabled {{
    background-color: #15161C;
    color: #484F58;
}}


/* ═══════════════════════════════════════════════════════════
   SPINBOX & DOUBLESPINBOX
   ═══════════════════════════════════════════════════════════ */

QSpinBox, QDoubleSpinBox {{
    background-color: {_BG_INPUT};
    border: 1px solid {_BORDER_DEFAULT};
    border-radius: {_RADIUS_MD};
    padding: 5px 8px;
    color: {_FG_DEFAULT};
    font-size: 13px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {_ACCENT_CYAN};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: #20222B;
    border: none;
    width: 18px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {_BORDER_DEFAULT};
}}


/* ═══════════════════════════════════════════════════════════
   COMBOBOX
   ═══════════════════════════════════════════════════════════ */

QComboBox {{
    background-color: #20222B;
    border: 1px solid #323543;
    border-radius: {_RADIUS_MD};
    padding: 5px 12px;
    color: {_FG_DEFAULT};
    font-size: 12px;
    font-weight: 500;
}}
QComboBox:hover {{
    border-color: {_ACCENT_CYAN};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {_BG_ELEVATED};
    border: 1px solid {_BORDER_DEFAULT};
    selection-background-color: {_ACCENT_CYAN_DARK};
    color: {_FG_DEFAULT};
    padding: 4px;
    border-radius: {_RADIUS_SM};
}}


/* ═══════════════════════════════════════════════════════════
   CHECKBOX & RADIOBUTTON
   ═══════════════════════════════════════════════════════════ */

QCheckBox, QRadioButton {{
    spacing: 8px;
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {_BORDER_DEFAULT};
    border-radius: 3px;
    background: {_BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background-color: {_ACCENT_CYAN};
    border-color: {_ACCENT_CYAN_HOVER};
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {_BORDER_DEFAULT};
    border-radius: 8px;
    background: {_BG_INPUT};
}}
QRadioButton::indicator:checked {{
    background-color: {_ACCENT_CYAN};
    border-color: {_ACCENT_CYAN_HOVER};
}}


/* ═══════════════════════════════════════════════════════════
   SLIDERS
   ═══════════════════════════════════════════════════════════ */

QSlider::groove:horizontal {{
    height: 5px;
    background: #252834;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {_ACCENT_CYAN};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: #FFFFFF;
    border: 2px solid {_ACCENT_CYAN};
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {_ACCENT_CYAN_HOVER};
}}


/* ═══════════════════════════════════════════════════════════
   PROGRESS BAR
   ═══════════════════════════════════════════════════════════ */

QProgressBar {{
    background-color: #1A1C24;
    border: 1px solid {_BORDER_DEFAULT};
    border-radius: {_RADIUS_MD};
    text-align: center;
    color: {_FG_ON_EMPHASIS};
    font-weight: 700;
    font-size: 11px;
    height: 16px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {_ACCENT_CYAN}, stop:1 {_ACCENT_BLUE});
    border-radius: 4px;
}}


/* ═══════════════════════════════════════════════════════════
   STATUS BAR
   ═══════════════════════════════════════════════════════════ */

QStatusBar {{
    background-color: {_BG_SURFACE};
    color: {_FG_MUTED};
    border-top: 1px solid {_BORDER_DEFAULT};
    padding: 3px 10px;
    font-size: 12px;
}}


/* ═══════════════════════════════════════════════════════════
   GROUP BOX
   ═══════════════════════════════════════════════════════════ */

QGroupBox {{
    border: 1px solid {_BORDER_DEFAULT};
    border-radius: {_RADIUS_MD};
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
    font-weight: 700;
    font-size: 12px;
    color: {_ACCENT_CYAN};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 12px;
}}


/* ═══════════════════════════════════════════════════════════
   SCROLL BARS
   ═══════════════════════════════════════════════════════════ */

QScrollBar:vertical {{
    border: none;
    background: {_BG_BASE};
    width: 8px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: #2D303E;
    min-height: 20px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{
    background: #464A5F;
}}
QScrollBar:horizontal {{
    border: none;
    background: {_BG_BASE};
    height: 8px;
    margin: 0px;
}}
QScrollBar::handle:horizontal {{
    background: #2D303E;
    min-width: 20px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal:hover {{
    background: #464A5F;
}}


/* ═══════════════════════════════════════════════════════════
   SPLITTER & DIALOGS
   ═══════════════════════════════════════════════════════════ */

QSplitter::handle {{
    background-color: {_BORDER_MUTED};
}}
QSplitter::handle:hover {{
    background-color: {_ACCENT_CYAN};
}}
QDialog {{
    background-color: {_BG_BASE};
}}
QMessageBox {{
    background-color: {_BG_ELEVATED};
}}


/* ═══════════════════════════════════════════════════════════
   CUSTOM LABELS & BADGES
   ═══════════════════════════════════════════════════════════ */

QLabel#panelTitle {{
    font-size: 14px;
    font-weight: 800;
    color: {_ACCENT_CYAN};
}}
QLabel#mutedLabel {{
    color: {_FG_MUTED};
    font-size: 11px;
}}
QLabel#accentLabel {{
    color: {_ACCENT_CYAN};
    font-weight: 700;
}}
QLabel#statusOk {{
    color: {_ACCENT_GREEN};
    font-weight: 600;
}}
QLabel#statusError {{
    color: {_DANGER_RED};
    font-weight: 600;
}}
QLabel#badgeOk {{
    background-color: rgba(0, 230, 118, 0.15);
    color: {_ACCENT_GREEN};
    border: 1px solid {_ACCENT_GREEN};
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
}}
QLabel#badgeWarn {{
    background-color: rgba(255, 179, 0, 0.12);
    color: {_WARNING_YELLOW};
    border: 1px solid {_WARNING_YELLOW};
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
}}
QLabel#badgeOff {{
    background-color: rgba(142, 146, 164, 0.12);
    color: {_FG_MUTED};
    border: 1px solid #363A4B;
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
}}
"""
