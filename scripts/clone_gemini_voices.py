"""Register the bundled voice-reference WAVs with local VieNeu, without APIs."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from vtranslator.tts.engine import TtsEngine  # noqa: E402


def main() -> int:
    engine = TtsEngine()
    if not engine.model_manager.is_loaded():
        print("VieNeu v3 Turbo chưa được tải. Mở Cài đặt → TTS và bấm Tải mô hình.")
        return 1

    engine._init_vieneu_backend()
    if engine._vieneu_tts is None:
        print("Không nạp được VieNeu. Kiểm tra gói vieneu và các tệp mô hình ONNX.")
        return 1

    print(f"Đã đăng ký {len(engine.model_manager.get_cloned_gemini_voices())} giọng mẫu cục bộ.")
    print(f"Tệp giọng: {engine.model_manager.model_dir}/voices_v3_turbo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
