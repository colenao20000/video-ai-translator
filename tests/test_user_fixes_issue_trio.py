"""Tests verifying the fixes for:
1. Translation & Dialogue Assignment using Gemini via 9router with robust chunking and timeout resistance.
2. Automatic voice assignment based on script taking VieNeu cloned Gemini voices (not stock VieNeu voices).
3. Safe master dub WAV export when destination directory is non-writable (/Users/...).
"""

import os
import json
import pytest
import soundfile as sf
import numpy as np
from unittest.mock import MagicMock, patch

from src.vtranslator.domain.characters import Character
from src.vtranslator.domain.segments import SubtitleSegment
from src.vtranslator.domain.enums import SpeakerGender, CharacterMatchStatus
from src.vtranslator.domain.voices import VoiceLibrary, CLONED_GEMINI_VOICES, CLONED_GEMINI_VOICE_IDS
from src.vtranslator.ai.character_resolver import CharacterResolver
from src.vtranslator.ai.vision_character import VisionCharacterService
from src.vtranslator.services.nine_router_service import NineRouterService
from src.vtranslator.audio.fit_to_slot import AudioTimelineScheduler
from src.vtranslator.app.settings import AppSettings


class TestUserFixesIssueTrio:
    def test_settings_defaults_to_gemini_model_via_9router(self):
        """Kiểm tra cài đặt mặc định sử dụng model Gemini trên 9router."""
        settings = AppSettings()
        assert "gemini" in settings.vision_model.lower()
        assert settings.vision_model == "ag/gemini-3.8-flash-low"

    def test_nine_router_gemini_recommended(self):
        """Kiểm tra 9router nhận diện và khuyên dùng model Gemini nhanh."""
        gemini_model = NineRouterService.get_recommended_gemini_model()
        assert "gemini" in gemini_model.lower()
        assert gemini_model == "ag/gemini-3.8-flash-low"

    def test_cloned_gemini_voices_are_prioritized_at_top_of_library(self):
        """Kiểm tra danh mục giọng ưu tiên các giọng Gemini Clone lên đầu bảng."""
        all_voices = VoiceLibrary.get_all()
        top_10_ids = {v.id for v in all_voices[:10]}
        assert "Gemini Puck" in top_10_ids
        assert "Gemini Charon" in top_10_ids
        assert "Gemini Kore" in top_10_ids
        assert "Gemini Zephyr" in top_10_ids
        # Đảm bảo không phải giọng gốc Minh Đức/Trúc Ly ở vị trí đầu tiên
        assert all_voices[0].id == "Gemini Puck"

    def test_auto_assign_voices_prefers_cloned_gemini_voices_not_stock(self):
        """
        Kiểm tra CharacterResolver.auto_assign_voices tự động chọn các giọng
        Gemini Clone (Gemini Puck, Gemini Kore...) thay vì giọng gốc (Minh Đức, Trúc Ly).
        """
        char_male = Character(id="c1", speaker_id="SPK_01", name="Nam chính", gender=SpeakerGender.MALE)
        char_female = Character(id="c2", speaker_id="SPK_02", name="Nữ chính", gender=SpeakerGender.FEMALE)

        assigned = CharacterResolver.auto_assign_voices([char_male, char_female])

        # Phải được gán giọng Gemini Clone, TUYỆT ĐỐI không phải Minh Đức hay Trúc Ly
        assert assigned[0].voice_id.startswith("Gemini ")
        assert assigned[0].voice_id != "Minh Đức"
        assert assigned[1].voice_id.startswith("Gemini ")
        assert assigned[1].voice_id != "Trúc Ly"

    def test_assign_dialogue_smart_syncs_cloned_gemini_voices_to_segments(self, monkeypatch):
        """
        Kiểm tra assign_dialogue_smart chia vai và đồng bộ giọng đọc VieNeu Gemini Clone
        tương ứng cho từng câu thoại trong kịch bản.
        """
        char_1 = Character(id="char_01", speaker_id="SPK_01", name="Trần Tứ", gender=SpeakerGender.MALE, voice_id="Charon")
        char_2 = Character(id="char_02", speaker_id="SPK_02", name="Tiểu Lan", gender=SpeakerGender.FEMALE, voice_id="Zephyr")

        segs = [
            SubtitleSegment(segment_id="s1", start_ms=0, end_ms=1500, original_text="Ta là Trần Tứ"),
            SubtitleSegment(segment_id="s2", start_ms=1600, end_ms=3000, original_text="Cứu tôi với!"),
        ]

        # Mock phản hồi phân vai từ Gemini 9router
        mock_response_data = [
            {"id": "s1", "character_id": "char_01", "speech_style": "uy nghiêm"},
            {"id": "s2", "character_id": "char_02", "speech_style": "hoảng loạn"}
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps({"choices": [{"message": {"content": json.dumps(mock_response_data)}}]})

        import httpx
        monkeypatch.setattr(httpx.Client, "post", lambda *args, **kwargs: mock_resp)
        monkeypatch.setattr(NineRouterService, "check_connection", lambda *args, **kwargs: (True, "OK", ["ag/gemini-3.8-flash-low"]))

        out_segs, out_chars = VisionCharacterService.assign_dialogue_smart(segs, [char_1, char_2])

        assert out_segs[0].character_id == "char_01"
        assert out_segs[0].voice_id == "Gemini Charon"
        assert out_segs[0].speech_style == "uy nghiêm"

        assert out_segs[1].character_id == "char_02"
        assert out_segs[1].voice_id == "Gemini Zephyr"
        assert out_segs[1].speech_style == "hoảng loạn"

    def test_build_master_dubbing_track_safe_fallback_on_non_writable_users_dir(self, tmp_path):
        """
        Kiểm tra build_master_dubbing_track khi đường dẫn là /Users/testtt_vi_master_dub.wav
        (không có quyền ghi trên macOS) thì tự động chuyển sang thư mục an toàn và ghi thành công.
        """
        # Tạo 1 segment có file âm thanh giả lập
        dummy_wav = str(tmp_path / "seg_audio.wav")
        data = np.zeros(22050, dtype=np.float32)
        sf.write(dummy_wav, data, 22050)

        seg = SubtitleSegment(
            segment_id="s1",
            start_ms=0,
            end_ms=1000,
            effective_tts_start_ms=0,
            effective_tts_end_ms=1000,
            tts_audio_path=dummy_wav
        )

        # Đường dẫn không có quyền ghi: /Users/testtt_vi_master_dub.wav
        bad_path = "/Users/testtt_vi_master_dub.wav"

        # Gọi hàm tạo master dubbing
        result_path = AudioTimelineScheduler.build_master_dubbing_track(
            segments=[seg],
            total_duration_sec=2.0,
            output_wav_path=bad_path,
            sample_rate=44100
        )

        # Không được ném ngoại lệ System error và phải trả về đường dẫn tệp âm thanh hợp lệ
        assert result_path != bad_path
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 1000
        # Đọc lại file WAV thành công
        info = sf.info(result_path)
        assert info.samplerate == 44100
        assert info.duration >= 2.0

    def test_default_volume_levels_10_percent_original(self):
        """Kiểm tra âm lượng mặc định: 10% âm thanh gốc, 100% âm thanh thuyết minh AI."""
        from src.vtranslator.domain.models import AudioMixSettings
        app_settings = AppSettings()
        assert app_settings.original_volume == 0.10
        assert app_settings.dubbing_volume == 1.00

        mix_settings = AudioMixSettings()
        assert mix_settings.original_volume == 0.10
        assert mix_settings.dubbing_volume == 1.00

    def test_female_dialogue_auto_correction_to_female_character(self):
        """
        Kiểm tra câu thoại nữ giới (như 'Đừng mà!', 'Đồ biến thái!') không bao giờ
        bị gán nhầm cho giọng nam, tự động sửa sang nhân vật nữ.
        """
        char_male = Character(id="char_01", speaker_id="SPK_01", name="Trần Tứ", gender=SpeakerGender.MALE, voice_id="Charon")
        char_female = Character(id="char_02", speaker_id="SPK_02", name="Cô gái", gender=SpeakerGender.FEMALE, voice_id="Zephyr")

        # Giả lập câu thoại nữ nhưng bị map nhầm sang char_01 (nam)
        seg = SubtitleSegment(segment_id="s_fem", start_ms=0, end_ms=1000, original_text="Đừng mà! Cứu tôi với!")
        with patch.object(VisionCharacterService, "analyze_characters_from_script", return_value=[char_male, char_female]):
            with patch("httpx.Client.post") as mock_post:
                # Mock AI trả về nhầm char_01
                mock_post.return_value.status_code = 200
                mock_post.return_value.text = json.dumps({"choices": [{"message": {"content": '[{"id": "s_fem", "character_id": "char_01", "speech_style": "hoảng sợ"}]'}}]})

                segs, chars = VisionCharacterService.assign_dialogue_smart([seg], [char_male, char_female])
                # Phải tự động nắn sang char_02 (nữ) và gán giọng nữ Gemini Zephyr
                assert segs[0].character_id == "char_02"
                assert segs[0].voice_id == "Gemini Zephyr"

    def test_tts_emotional_punctuation_cleaner(self):
        """Kiểm tra bộ tạo dấu câu cảm xúc tự động cho VieNeu dựa vào style_desc."""
        from src.vtranslator.tts.engine import TtsEngine
        eng = TtsEngine()
        # Câu khẩn cấp/hét lớn/hoảng loạn tự động nhận dấu !
        assert eng._clean_tts_text("Mau cứu tôi", style_desc="hét lớn hoảng loạn") == "Mau cứu tôi!"
        # Câu hỏi/nghi ngờ tự động nhận dấu ?
        assert eng._clean_tts_text("Ngươi là ai", style_desc="hỏi ngờ vực") == "Ngươi là ai?"
        # Câu thì thầm/yếu ớt tự động nhận dấu ...
        assert eng._clean_tts_text("Tôi sợ quá", style_desc="thì thầm run rẩy") == "Tôi sợ quá..."
        # Câu bình thường nhận dấu .
        assert eng._clean_tts_text("Tôi đi đây", style_desc="bình tĩnh") == "Tôi đi đây."
