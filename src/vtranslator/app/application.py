"""Installed application entry point."""

import sys
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("QT_FFMPEG_DECODER_FORCE_SW", "1")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from PySide6.QtWidgets import QApplication
from .dependency_container import DependencyContainer
from ..ui.main_window import MainWindow


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("vTranslate Studio")
    app.setOrganizationName("vTranslate")
    window = MainWindow(DependencyContainer())
    window.show()
    sys.exit(app.exec())
