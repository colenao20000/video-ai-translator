"""Build verification script for Video AI Translator."""

import sys
import compileall
import subprocess


def build():
    print("🚀 Bắt đầu quá trình Build & Kiểm tra mã nguồn...")

    # 1. Byte-compile toàn bộ source code
    print("1. Đang kiểm tra cú pháp và compile bytecode...")
    success = compileall.compile_dir("src", force=True, quiet=True)
    if not success:
        print("❌ Lỗi compile bytecode trong src!")
        sys.exit(1)

    success_root = compileall.compile_file("app.py", force=True, quiet=True)
    if not success_root:
        print("❌ Lỗi compile app.py!")
        sys.exit(1)

    print("✅ Compile bytecode thành công 100%!")


if __name__ == "__main__":
    build()
