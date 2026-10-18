"""Tests for 9router Adaptive Batching, Resilient JSON Parsing & Zero-Collision Timeline Synchronization."""

import os
import json
import numpy as np
import soundfile as sf
import pytest
from unittest.mock import patch, MagicMock

from src.vtranslator.domain.segments import SubtitleSegment
from src.vtranslator.domain.enums import SegmentFitStatus
from src.vtranslator.services.nine_router_service import NineRouterService
from src.vtranslator.audio.fit_to_slot import FitToSlotProcessor, AudioTimelineScheduler


def make_test_wav(filepath: str, duration_sec: float, sample_rate: int = 44100) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    # Sine wave with tone so audio is not pure silence
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    data = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(filepath, data, sample_rate, subtype="PCM_16")
    return filepath


class TestNineRouterResilience:
    def test_clean_markdown_json(self):
        raw = "```json\n[\"câu 1\", \"câu 2\"]\n```"
        cleaned = NineRouterService._clean_markdown_json(raw)
        assert cleaned == '["câu 1", "câu 2"]'

    def test_extract_json_array_with_trailing_comma(self):
        raw = '[\n  "câu 1",\n  "câu 2",\n]'
        parsed = NineRouterService._extract_json_array(raw)
        assert parsed == ["câu 1", "câu 2"]

    def test_extract_id_translations_standard(self):
        raw = json.dumps([
            {"id": "seg_1", "translation": "Xin chào"},
            {"id": "seg_2", "translation": "Tạm biệt"}
        ])
        mapping = NineRouterService._extract_id_translations(raw)
        assert mapping["seg_1"] == "Xin chào"
        assert mapping["seg_2"] == "Tạm biệt"

    def test_extract_id_translations_from_truncated_stream(self):
        # Truncated before array closing bracket
        truncated_raw = '[{"id": "s1", "translation": "Tôi hiểu rồi"}, {"id": "s2", "translation": "Đi thôi"'
        mapping = NineRouterService._extract_id_translations(truncated_raw)
        assert mapping["s1"] == "Tôi hiểu rồi"

    def test_adaptive_halving_on_batch_timeout(self):
        """Khi batch lớn gặp timeout, hệ thống tự động chia đôi lô và xử lý thành công."""
        texts = [f"Sentence {i}" for i in range(6)]

        call_counts = {"count": 0}

        def mock_post(url, headers=None, json=None, **kwargs):
            call_counts["count"] += 1
            payload = json or {}
            msgs = payload.get("messages", [])
            user_msg = msgs[-1]["content"] if msgs else ""

            # Giả lập lô 6 câu bị timeout
            if "Sentence 0" in user_msg and "Sentence 5" in user_msg:
                import httpx
                raise httpx.ReadTimeout("timed out on batch size 6")

            # Các sub-batch nhỏ (3 câu) thành công
            extracted = []
            for i in range(6):
                if f"Sentence {i}" in user_msg:
                    extracted.append(f"Dịch câu {i}")

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = json_lib.dumps({"choices": [{"message": {"content": json_lib.dumps(extracted)}}]})
            return mock_resp

        import json as json_lib
        with patch("httpx.Client.post", side_effect=mock_post):
            results = NineRouterService.translate_batch(texts, source_lang="en", target_lang="vi")

        assert len(results) == 6
        assert results[0] == "Dịch câu 0"
        assert results[5] == "Dịch câu 5"
        assert call_counts["count"] >= 3  # 1 fail on size 6, 2 successes on size 3

    def test_translate_segments_by_id_preserves_text_on_failure(self):
        """Nếu tất cả API thất bại, bảo toàn nguyên bản lời thoại gốc thay vì để rỗng."""
        items = [{"id": "seg_err", "text": "Hello World", "character": "c1"}]

        with patch("httpx.Client.post", side_effect=Exception("Connection refused")):
            res = NineRouterService.translate_segments_by_id(items)

        assert res["seg_err"] == "Hello World"

    def test_get_recommended_model_prioritizes_fast_models(self):
        """Kiểm tra get_recommended_model ưu tiên model nhanh như cx/gpt-5.6-terra."""
        models = ["ag/gemini-3.8-flash-low", "cx/gpt-5.6-terra", "tokenharbor/deepseek-v4.1-flash:free"]
        rec = NineRouterService.get_recommended_model(models)
        assert rec == "cx/gpt-5.6-terra"

    def test_sort_models_by_recommendation(self):
        """Kiểm tra sort_models_by_recommendation sắp xếp các model tối ưu lên đầu."""
        models = ["unknown-model", "ag/gemini-3.8-flash-low", "cx/gpt-5.6-terra", "cx/gpt-5.6-luna"]
        sorted_models = NineRouterService.sort_models_by_recommendation(models)
        assert sorted_models[0] == "cx/gpt-5.6-terra"
        assert sorted_models[1] == "cx/gpt-5.6-luna"

    def test_test_model_speed_mock(self):
        """Kiểm tra hàm test_model_speed đo độ trễ chuẩn xác."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps({"choices": [{"message": {"content": "2"}}]})

        with patch("httpx.Client.post", return_value=mock_resp):
            ok, lat, msg = NineRouterService.test_model_speed("cx/gpt-5.6-terra")
            assert ok is True
            assert "Phản hồi trong" in msg
            assert lat >= 0.0


class TestTimelineZeroCollision:
    def test_consecutive_segments_never_spill_over(self, tmp_path):
        """
        Kiểm thử cốt lõi: Câu 1 có TTS dài 3.5s nhưng chỉ có slot 1.2s trước câu 2.
        - Câu 1 BẮT BUỘC kết thúc trước khi Câu 2 bắt đầu (nghỉ tối thiểu 80ms).
        - Phụ đề Câu 1 BẮT BUỘC tắt trước Câu 2 ít nhất 20ms.
        - Câu 2 xuất phát CHÍNH XÁC tại 2200ms.
        """
        wav_1 = make_test_wav(str(tmp_path / "long_1.wav"), 3.5)
        wav_2 = make_test_wav(str(tmp_path / "normal_2.wav"), 1.0)

        seg_1 = SubtitleSegment(
            segment_id="seg_01",
            start_ms=1000,
            end_ms=2000,
            original_text="Sentence 1",
            tts_audio_path=wav_1
        )
        seg_2 = SubtitleSegment(
            segment_id="seg_02",
            start_ms=2200,
            end_ms=3500,
            original_text="Sentence 2",
            tts_audio_path=wav_2
        )

        scheduled = AudioTimelineScheduler.schedule_segments([seg_1, seg_2], max_tempo=1.45, min_gap_ms=80)
        by_id = {s.segment_id: s for s in scheduled}

        s1 = by_id["seg_01"]
        s2 = by_id["seg_02"]

        # Câu 2 không đổi thời điểm bắt đầu
        assert s2.effective_tts_start_ms == 2200
        assert s2.start_ms == 2200

        # Câu 1 kết thúc âm thanh trước Câu 2 ít nhất 80ms (<= 2120ms)
        assert s1.effective_tts_end_ms <= 2200 - 80

        # Phụ đề Câu 1 tắt trước Câu 2 ít nhất 20ms (<= 2180ms)
        assert s1.end_ms <= 2200 - 20

        # Âm thanh được đánh dấu điều chỉnh
        assert s1.fit_status == SegmentFitStatus.ADJUSTED

        # Kiểm tra ghép master dubbing track không bị lỗi
        master_out = str(tmp_path / "master_dub.wav")
        AudioTimelineScheduler.build_master_dubbing_track(scheduled, total_duration_sec=6.0, output_wav_path=master_out)
        assert os.path.exists(master_out)

    def test_fit_to_slot_smooth_fade_out_clamping(self, tmp_path):
        """Xác minh FitToSlotProcessor áp dụng fade-out mượt mà khi âm thanh bị cắt gọn."""
        long_wav = make_test_wav(str(tmp_path / "overflow.wav"), 3.0)
        fitted_wav = str(tmp_path / "fitted.wav")

        target_dur = 1.0  # Yêu cầu co vào 1.0 giây
        res = FitToSlotProcessor.process(long_wav, target_dur, fitted_wav, max_tempo_stretch=1.45)

        assert os.path.exists(res)
        info = sf.info(res)
        # Thời lượng tệp phải nằm trong giới hạn target_dur (+/- 30ms tolerance)
        assert info.duration <= target_dur + 0.03

        # Đọc dữ liệu kiểm tra mẫu âm thanh cuối cùng được làm mờ dần về 0
        data, _ = sf.read(res)
        assert abs(data[-1]) < 0.05
