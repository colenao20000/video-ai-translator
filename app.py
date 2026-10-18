"""Application Entrypoint for Video AI Translator Desktop Studio."""

import sys
import os

# Environment setup for macOS / PySide6 / OpenMP compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["QT_FFMPEG_DECODER_FORCE_SW"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

# Add the source package root for direct `python app.py` launches.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from PySide6.QtWidgets import QApplication
from vtranslator.app.dependency_container import DependencyContainer
from vtranslator.ui.main_window import MainWindow


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("vTranslate Studio")
    app.setOrganizationName("vTranslate")

    container = DependencyContainer()
    window = MainWindow(container)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
