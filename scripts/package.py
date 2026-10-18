"""Packaging helper script for macOS / cross-platform distribution."""

import os
import sys
import subprocess


def package():
    print("📦 Bắt đầu đóng gói ứng dụng vTranslate Studio PC...")
    print("Nền tảng mục tiêu: macOS Intel / Apple Silicon (Universal/Native)")

    # Kiểm tra các module cần thiết
    import PySide6
    print(f"PySide6 version: {PySide6.__version__}")
    print("Sẵn sàng cho quy trình build release.")


if __name__ == "__main__":
    package()
