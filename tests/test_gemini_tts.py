"""Unit and Integration Tests for Gemini Next-Gen TTS & Anti-Timeout Chunking."""

import os
import io
import json
import threading
import numpy as np
import soundfile as sf
import pytest

from src.vtranslator.tts.providers.gemini_provider import (
    GeminiTtsProvider,
    DEFAULT_GEMINI_TTS_MODEL,
    GEMINI_PREBUILT_VOICES,
    VIETNAMESE_VOICE_TO_GEMINI
)
from src.vtranslator.tts.engine import TtsEngine
from src.vtranslator.domain.segments import SubtitleSegment
from src.vtranslator.domain.enums import SpeakerGender, VoiceStyle, VoiceCharacter, TtsStatus
from src.vtranslator.domain.voices import VoiceLibrary


def create_dummy_wav_bytes(duration_sec: float = 0.5, sample_rate: int = 24000) -> bytes:
    """Helper tạo buffer WAV hợp lệ trong bộ nhớ."""
    num_samples = int(duration_sec * sample_rate)
    # Sine wave để có dữ liệu âm thanh thật
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestGeminiTtsChunking:
    """Kiểm thử thuật toán chia nhỏ text thông minh chống Timeout."""

    def test_short_text_stays_single_chunk(self):
        text = "Xin chào các bạn."
        chunks = GeminiTtsProvider.split_text_into_chunks(text, max_chars=150)
        assert len(chunks) == 1
        assert chunks[0] == "Xin chào các bạn."

    def test_adds_terminal_punctuation_if_missing(self):
        text = "Một câu thoại không có dấu chấm"
        chunks = GeminiTtsProvider.split_text_into_chunks(text, max_chars=150)
        assert len(chunks) == 1
        assert chunks[0].endswith(".")

    def test_splits_long_paragraph_by_sentences(self):
        text = (
            "Hôm nay trời rất đẹp và chúng ta cùng bắt đầu hành trình mới. "
            "Đây là một câu chuyện vô cùng kịch tính về giang hồ và tình nghĩa sư đồ trong hẻm nhỏ miếu nhai. "
            "Mọi chuyện xảy ra vào một đêm mưa bão khi nhân vật chính xuất hiện."
        )
        chunks = GeminiTtsProvider.split_text_into_chunks(text, max_chars=100)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 100
            assert c.endswith(('.', '!', '?', ',', ';', '…'))

    def test_splits_long_clause_by_commas_and_words(self):
        text = (
            "Khi bước vào căn phòng tối tăm ấy, anh ta nhìn thấy một bóng người kỳ lạ đang đứng sau cánh cửa, "
            "tay cầm một thanh kiếm sắc bén phát ra ánh sáng màu xanh lấp lánh giữa đêm tối vô tận"
        )
        chunks = GeminiTtsProvider.split_text_into_chunks(text, max_chars=70)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 70

    def test_cleans_brackets_and_unwanted_characters(self):
        raw_text = ' [Cười lớn] (tiếng nhạc) "Xin chào mọi người!" 【Ghi chú】 '
        clean = GeminiTtsProvider.clean_text_for_tts(raw_text)
        assert "[Cười lớn]" not in clean
        assert "(tiếng nhạc)" not in clean
        assert "【Ghi chú】" not in clean
        assert "Xin chào mọi người!" in clean


class TestGeminiVoiceMapping:
    """Kiểm thử ánh xạ giọng đọc và phong cách biểu cảm cho Gemini TTS."""

    def test_direct_gemini_voice_resolves(self):
        voice, style = GeminiTtsProvider.resolve_voice_and_style("Kore")
        assert voice == "Kore"

        voice, style = GeminiTtsProvider.resolve_voice_and_style("puck")
        assert voice == "Puck"

    def test_vietnamese_personas_mapped_correctly(self):
        # Minh Đức -> Charon (Tin tức)
        voice, style = GeminiTtsProvider.resolve_voice_and_style("Minh Đức")
        assert voice == "Charon"
        assert "tin tức" in style

        # Trúc Ly -> Zephyr (Tươi sáng)
        voice, style = GeminiTtsProvider.resolve_voice_and_style("Trúc Ly")
        assert voice == "Zephyr"

        # Ngọc Linh -> Sulafat (Kể chuyện)
        voice, style = GeminiTtsProvider.resolve_voice_and_style("Ngọc Linh")
        assert voice == "Sulafat"

    def test_fallback_by_gender_and_character(self):
        # Nam + Deep -> Orus
        voice, style = GeminiTtsProvider.resolve_voice_and_style(
            "Nhân vật nam lạ",
            gender=SpeakerGender.MALE,
            voice_character=VoiceCharacter.DEEP
        )
        assert voice == "Orus"

        # Nữ + Soft -> Sulafat
        voice, style = GeminiTtsProvider.resolve_voice_and_style(
            "Nhân vật nữ lạ",
            gender=SpeakerGender.FEMALE,
            voice_character=VoiceCharacter.SOFT
        )
        assert voice == "Sulafat"


class TestGeminiPayloadAndStitching:
    """Kiểm thử cấu trúc payload API và ghép nối âm thanh."""

    def test_build_payload_structure(self):
        provider = GeminiTtsProvider(api_key="AIza_test_key", model_name="gemini-3.8-flash-tts")
        payload = provider.build_payload(
            text_chunk="Xin chào bạn.",
            voice_name="Puck",
            style_desc="ấm áp và thân thiện"
        )

        assert payload["contents"][0]["role"] == "user"
        part = payload["contents"][0]["parts"][0]
        assert part["text"] == "Xin chào bạn."
        assert part["speech_metadata"]["style"] == "ấm áp và thân thiện"

        gen_cfg = payload["generationConfig"]
        assert gen_cfg["responseModalities"] == ["AUDIO"]
        v_cfg = gen_cfg["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
        assert v_cfg["voiceName"] == "Puck"

    def test_stitch_audio_chunks(self, tmp_path):
        chunk1 = create_dummy_wav_bytes(0.3, sample_rate=24000)
        chunk2 = create_dummy_wav_bytes(0.4, sample_rate=24000)

        out_path = str(tmp_path / "stitched_test.wav")
        res = GeminiTtsProvider.stitch_audio_chunks([chunk1, chunk2], out_path, micro_silence_ms=50)

        assert os.path.exists(res)
        info = sf.info(res)
        assert info.samplerate == 24000
        # Tổng độ dài: 0.3s + 0.4s + 0.05s (silence) + 0.1s (tail pad) = ~0.85s
        assert info.duration >= 0.75


class TestTtsEngineIntegration:
    """Kiểm thử tích hợp TtsEngine với Gemini TTS và fallback."""

    def test_engine_initializes_with_gemini(self, tmp_path):
        engine = TtsEngine(
            temp_dir=str(tmp_path),
            preferred_engine="gemini",
            gemini_model="gemini-3.8-flash-tts",
            max_chunk_chars=120
        )
        assert engine._gemini_tts is not None
        assert engine._gemini_tts.model_name == "gemini-3.8-flash-tts"
        assert engine._gemini_tts.max_chunk_chars == 120

    def test_generate_segments_batch_progress(self, tmp_path, monkeypatch):
        engine = TtsEngine(temp_dir=str(tmp_path), inter_segment_delay=0.0)

        # Mock _synthesize_sync để tạo file dummy thay vì gọi mạng thật
        def mock_synth(text, voice_name, output_wav):
            with open(output_wav, "wb") as f:
                f.write(create_dummy_wav_bytes(0.2))

        monkeypatch.setattr(engine, "_synthesize_sync", mock_synth)

        seg1 = SubtitleSegment(segment_id="s1", start_ms=0, end_ms=1000, translated_text="Câu thứ nhất")
        seg2 = SubtitleSegment(segment_id="s2", start_ms=1500, end_ms=2500, translated_text="Câu thứ hai")

        progress_log = []
        engine.generate_segments_batch(
            [seg1, seg2],
            force_refresh=True,
            inter_segment_delay=0.0,
            progress_callback=lambda p, msg: progress_log.append((p, msg))
        )

        assert seg1.tts_status == TtsStatus.READY
        assert seg2.tts_status == TtsStatus.READY
        assert os.path.exists(seg1.tts_audio_path)
        assert os.path.exists(seg2.tts_audio_path)
        assert len(progress_log) == 2
        assert progress_log[-1][0] == 100

    def test_validate_and_sync_segment_characters(self):
        from src.vtranslator.domain.characters import Character

        char1 = Character(id="char_01", speaker_id="SPEAKER_00", name="Trần Tứ", voice_id="Charon")
        char2 = Character(id="char_02", speaker_id="SPEAKER_01", name="Cô Gái", voice_id="Zephyr")

        seg1 = SubtitleSegment(
            segment_id="seg_01", start_ms=0, end_ms=1000,
            speaker_id="SPEAKER_00", character_id="char_01",
            voice_id="Minh Đức", tts_audio_path="/fake/path.wav"
        )
        seg2 = SubtitleSegment(
            segment_id="seg_02", start_ms=1000, end_ms=2000,
            speaker_id="SPEAKER_01", character_id="char_02",
            voice_id="Trúc Ly", tts_audio_path="/fake/path2.wav"
        )

        TtsEngine.validate_and_sync_segment_characters([seg1, seg2], [char1, char2])

        # Đã đồng bộ đúng giọng theo Character và hủy cache cũ nếu giọng khác
        assert seg1.voice_id == "Charon"
        assert seg1.tts_audio_path is None  # Đã làm mới cache vì đổi giọng

        assert seg2.voice_id == "Zephyr"
        assert seg2.tts_audio_path is None

    def test_batch_sequential_rate_limit_delay(self, tmp_path, monkeypatch):
        import time

        engine = TtsEngine(temp_dir=str(tmp_path), inter_segment_delay=0.05)

        calls = []
        def mock_synth(text, voice_name, output_wav):
            calls.append(time.time())
            with open(output_wav, "wb") as f:
                f.write(create_dummy_wav_bytes(0.1))

        monkeypatch.setattr(engine, "_synthesize_sync", mock_synth)
        if engine._gemini_tts:
            engine._gemini_tts.api_keys = ["single_key"]

        seg1 = SubtitleSegment(segment_id="s1", start_ms=0, end_ms=1000, translated_text="A")
        seg2 = SubtitleSegment(segment_id="s2", start_ms=1500, end_ms=2500, translated_text="B")

        engine.generate_segments_batch(
            [seg1, seg2],
            force_refresh=True,
            inter_segment_delay=0.08
        )

        assert len(calls) == 2
        # Giữa 2 lần gọi phải cách nhau ít nhất 0.07s
        assert (calls[1] - calls[0]) >= 0.07


class TestMultiKeyRotationAndPooling:
    """Kiểm thử cơ chế xoay vòng đa API Key và giãn cách nhịp nghỉ độc lập."""

    def test_init_multiple_keys_parsing(self):
        # Kiểm tra nạp từ danh sách
        provider = GeminiTtsProvider(api_keys=["key1", "key2", "key3"])
        assert len(provider.api_keys) == 3
        assert provider.api_keys == ["key1", "key2", "key3"]
        assert provider.api_key == "key1"

        # Kiểm tra nạp từ chuỗi phân cách bởi dấu phẩy và xuống dòng
        provider2 = GeminiTtsProvider(api_key="keyA, keyB\nkeyC; keyA")
        assert len(provider2.api_keys) == 3
        assert provider2.api_keys == ["keyA", "keyB", "keyC"]

    def test_independent_key_rate_limiting(self):
        import time
        provider = GeminiTtsProvider(api_keys=["key_alpha", "key_beta"], min_request_interval_sec=1.0)

        # Sử dụng key_alpha tại thời điểm hiện tại
        provider.mark_key_used("key_alpha")

        # Kiểm tra key_alpha: Cần phải sleep ~1.0s
        k1, sleep1 = provider.get_available_key(preferred_key="key_alpha")
        assert k1 == "key_alpha"
        assert sleep1 > 0.5

        # Trong khi đó key_beta chưa dùng: Có thể gọi NGAY LẬP TỨC (sleep == 0)
        k2, sleep2 = provider.get_available_key(preferred_key="key_beta")
        assert k2 == "key_beta"
        assert sleep2 == 0.0

    def test_key_failover_on_429(self):
        import time
        provider = GeminiTtsProvider(api_keys=["key_1", "key_2"], min_request_interval_sec=0.1)

        # Đánh dấu key_1 bị 429
        provider.mark_key_429("key_1", cooldown_sec=10.0)

        # get_available_key sẽ tự động né key_1 và chọn key_2
        chosen, sleep_needed = provider.get_available_key()
        assert chosen == "key_2"
        assert sleep_needed == 0.0


class TestParallelTtsBatch:
    """Kiểm thử sinh TTS theo đợt song song qua nhiều API Key."""

    def test_parallel_batch_generation_with_multi_keys(self, tmp_path, monkeypatch):
        import time
        from src.vtranslator.domain.characters import Character

        engine = TtsEngine(temp_dir=str(tmp_path), inter_segment_delay=0.0)
        # Cung cấp 2 API keys
        engine._gemini_tts.api_keys = ["mock_key_1", "mock_key_2"]
        engine._gemini_tts.api_key = "mock_key_1"

        char1 = Character(id="char_01", speaker_id="SPEAKER_00", name="Nhân vật A", voice_id="Puck")
        char2 = Character(id="char_02", speaker_id="SPEAKER_01", name="Nhân vật B", voice_id="Kore")

        keys_used = []
        key_lock = threading.Lock()

        def mock_synth(text, voice_name, output_wav, style_desc=None, specific_key=None):
            with key_lock:
                keys_used.append(specific_key)
            time.sleep(0.05)
            with open(output_wav, "wb") as f:
                f.write(create_dummy_wav_bytes(0.1))

        monkeypatch.setattr(engine, "_synthesize_sync", mock_synth)

        # 4 câu thoại: 2 câu của char_01, 2 câu của char_02
        s1 = SubtitleSegment(segment_id="seg_1", start_ms=0, end_ms=1000, character_id="char_01", translated_text="Câu 1")
        s2 = SubtitleSegment(segment_id="seg_2", start_ms=1000, end_ms=2000, character_id="char_01", translated_text="Câu 2")
        s3 = SubtitleSegment(segment_id="seg_3", start_ms=2000, end_ms=3000, character_id="char_02", translated_text="Câu 3")
        s4 = SubtitleSegment(segment_id="seg_4", start_ms=3000, end_ms=4000, character_id="char_02", translated_text="Câu 4")

        start_time = time.time()
        engine.generate_segments_batch(
            [s1, s2, s3, s4],
            characters=[char1, char2],
            force_refresh=True,
            inter_segment_delay=0.0
        )
        elapsed = time.time() - start_time

        # Tất cả các segment đều thành công
        for s in [s1, s2, s3, s4]:
            assert s.tts_status == TtsStatus.READY

        # Cả 2 key đều đã được sử dụng đồng thời
        assert "mock_key_1" in keys_used
        assert "mock_key_2" in keys_used


class TestScriptAnalysisGeminiCasting:
    """Kiểm thử nhận diện nhân vật & gán giọng Gemini TTS theo ngữ cảnh phim."""

    def test_character_script_analysis_assigns_gemini_voice_and_acting_style(self, monkeypatch):
        from src.vtranslator.ai.vision_character import VisionCharacterService
        import httpx

        mock_ai_response = {
            "characters": [
                {
                    "id": "char_01",
                    "name": "Trần Tứ (Đạo sĩ bắt ma)",
                    "gender": "male",
                    "age_group": "adult",
                    "voice_character": "deep",
                    "gemini_voice": "Charon",
                    "acting_style": "đạo sĩ bắt ma uy nghiêm, khẩn trương, giọng đanh thép",
                    "role": "main"
                },
                {
                    "id": "char_02",
                    "name": "Tiểu Lan (Nạn nhân)",
                    "gender": "female",
                    "age_group": "young_adult",
                    "voice_character": "soft",
                    "gemini_voice": "Zephyr",
                    "acting_style": "cô gái hoảng loạn, sợ hãi run rẩy, van xin khẩn thiết",
                    "role": "main"
                }
            ]
        }

        # Mock cuộc gọi AI
        class MockResponse:
            status_code = 200
            text = json.dumps({"choices": [{"message": {"content": json.dumps(mock_ai_response)}}]})

        monkeypatch.setattr(httpx.Client, "post", lambda *args, **kwargs: MockResponse())

        segments = [
            SubtitleSegment(segment_id="seg_01", start_ms=0, end_ms=1000, original_text="Mau tránh ra, yêu quái sắp tới rồi!"),
            SubtitleSegment(segment_id="seg_02", start_ms=1000, end_ms=2000, original_text="Cứu tôi với, tôi sợ quá!")
        ]

        chars = VisionCharacterService.analyze_characters_from_script(segments)
        assert len(chars) == 2

        # Kiểm tra nhân vật 1
        assert chars[0].id == "char_01"
        assert chars[0].gender == SpeakerGender.MALE
        assert chars[0].voice_id == "Charon"
        assert "đạo sĩ" in chars[0].custom_prompt

        # Kiểm tra nhân vật 2
        assert chars[1].id == "char_02"
        assert chars[1].gender == SpeakerGender.FEMALE
        assert chars[1].voice_id == "Zephyr"
        assert "sợ hãi" in chars[1].custom_prompt

