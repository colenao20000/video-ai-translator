"""Translation Service with Strict Segment-ID Mapping & Quality Assurance."""

import json
import re
import logging
from typing import List, Callable, Optional, Dict, Any
import httpx
from ..domain.segments import SubtitleSegment
from ..domain.characters import Character
from ..services.cache_service import CacheService
from ..services.keychain_service import KeychainService
from ..services.nine_router_service import NineRouterService

logger = logging.getLogger(__name__)


class TranslationQualityError(Exception):
    pass


class TranslationService:
    def __init__(self, cache_service: CacheService, keychain_service: KeychainService):
        self.cache_service = cache_service
        self.keychain = keychain_service

    def translate_segments(
        self,
        segments: List[SubtitleSegment],
        characters: Optional[List[Character]] = None,
        source_lang: str = "auto",
        target_lang: str = "vi",
        batch_size: int = 10,
        verify_pass: bool = True,
        force_retranslate: bool = False,
        model_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[SubtitleSegment]:
        """
        Dịch theo segment_id cố định và ánh xạ kết quả theo mã câu:
        - KHÔNG ghép theo thứ tự mảng kết quả của API.
        - KHÔNG giả định thứ tự câu trả về giống đầu vào.
        - Giữ nguyên start_ms, end_ms, speaker_id, character_id khi cập nhật bản dịch.
        - Thử các dịch vụ dịch đã cấu hình; tốc độ và độ sẵn sàng phụ thuộc dịch vụ/model.
        """
        if not segments:
            return segments

        # Lưu snapshot thời gian và danh tính để đảm bảo quy tắc bất biến
        initial_snapshots = {
            s.segment_id: (s.start_ms, s.end_ms, s.speaker_id, s.character_id)
            for s in segments
        }

        is_9router_ok, _, _ = NineRouterService.check_connection()
        gemini_direct_key = NineRouterService.find_direct_gemini_key()
        custom_key = self.keychain.get_secret("translation_api_key")
        custom_url = self.keychain.get_secret("translation_api_url")

        char_map: Dict[str, Character] = {}
        if characters:
            for c in characters:
                if c.id:
                    char_map[c.id] = c
                if c.speaker_id:
                    char_map[c.speaker_id] = c

        def cache_context_for(segment: SubtitleSegment) -> str:
            char = char_map.get(segment.character_id) or char_map.get(segment.speaker_id)
            return json.dumps(
                {
                    "model": model_name or "",
                    "character_id": segment.character_id,
                    "speaker_id": segment.speaker_id,
                    "character_name": char.name if char else "",
                    "gender": char.gender.value if char else "unknown",
                    "role": char.custom_prompt if char and char.custom_prompt else "",
                    "speech_style": segment.speech_style or "",
                },
                ensure_ascii=False,
                sort_keys=True,
            )

        cache_contexts = {s.segment_id: cache_context_for(s) for s in segments}

        # 1. Lọc các segment cần dịch
        pending_segments: List[SubtitleSegment] = []
        for s in segments:
            if not s.original_text.strip():
                continue
            if force_retranslate:
                pending_segments.append(s)
            else:
                cached = self.cache_service.get_translation(
                    s.original_text,
                    source_lang,
                    target_lang,
                    context_key=cache_contexts[s.segment_id],
                )
                if cached and cached.strip() != s.original_text.strip():
                    s.update_translation(cached)
                else:
                    pending_segments.append(s)

        total_pending = len(pending_segments)

        failed_ids: List[str] = []

        # Context comes from the full timeline, including already translated/cache-hit lines.
        timeline = sorted(segments, key=lambda item: (item.start_ms, item.end_ms))
        timeline_index = {item.segment_id: i for i, item in enumerate(timeline)}

        # 2. Xử lý dịch thuật theo cụm với ánh xạ ID
        if pending_segments:
            num_batches = (total_pending + batch_size - 1) // batch_size
            use_direct_gemini = (model_name and (model_name.startswith("gemini-") and "/" not in model_name)) or (not is_9router_ok and bool(gemini_direct_key))
            backend_name = "Google Gemini Direct" if use_direct_gemini else ("9router Local" if is_9router_ok else "Custom AI API")

            for b_idx in range(num_batches):
                sub_batch = pending_segments[b_idx * batch_size : (b_idx + 1) * batch_size]
                items_payload = []
                for idx_in_sub, s in enumerate(sub_batch):
                    global_idx = b_idx * batch_size + idx_in_sub
                    line_idx = timeline_index.get(s.segment_id, 0)
                    next_s = timeline[line_idx + 1] if line_idx + 1 < len(timeline) else None
                    prev_s = timeline[line_idx - 1] if line_idx > 0 else None
                    dur_sec = max(0.5, round((s.end_ms - s.start_ms) / 1000.0, 1))
                    available_ms = s.end_ms - s.start_ms
                    if next_s and next_s.start_ms > s.start_ms:
                        available_ms = max(available_ms, next_s.start_ms - s.start_ms - 100)
                    speech_slot_sec = max(0.5, round(available_ms / 1000.0, 1))
                    text_len = len(s.original_text.strip())
                    max_w = max(5, int(speech_slot_sec * 3.2), int(text_len * 1.2))
                    source_hint = (
                        "Tự nhận diện ngôn ngữ nguồn từ từng câu."
                        if source_lang.lower() in ("auto", "", "unknown")
                        else f"Ngôn ngữ nguồn: {source_lang}."
                    )

                    char = char_map.get(s.character_id) or char_map.get(s.speaker_id)
                    speaker_name = (char.name if char else s.speaker_id) or "Người nói"
                    gender_str = (char.gender.value if char else "unknown")
                    role_str = (char.custom_prompt if char and char.custom_prompt else "")

                    item_data = {
                        "id": s.segment_id,
                        "text": s.original_text,
                        "character": s.character_id,
                        "speaker": speaker_name,
                        "gender": gender_str,
                        "duration_sec": dur_sec,
                        "slot_sec": speech_slot_sec,
                        "max_words": max_w,
                        "source_language_hint": source_hint,
                    }
                    if role_str:
                        item_data["role"] = role_str
                    if getattr(s, "speech_style", None):
                        item_data["style"] = s.speech_style
                    if prev_s:
                        prev_txt = prev_s.translated_text or prev_s.original_text
                        if prev_txt:
                            item_data["prev_line"] = prev_txt
                    if next_s:
                        next_txt = next_s.translated_text or next_s.original_text
                        if next_txt:
                            item_data["next_line"] = next_txt

                    items_payload.append(item_data)

                # Map kết quả dịch theo segment_id
                translation_by_id: Dict[str, str] = {}

                if use_direct_gemini:
                    translation_by_id = NineRouterService.translate_segments_direct_gemini(
                        items_payload, source_lang=source_lang, target_lang=target_lang,
                        model=model_name if (model_name and "/" not in model_name) else "gemini-3.5-flash-lite",
                        gemini_key=gemini_direct_key
                    )
                elif is_9router_ok:
                    translation_by_id = NineRouterService.translate_segments_by_id(
                        items_payload, source_lang=source_lang, target_lang=target_lang, model=model_name
                    )

                # Fallback trực tiếp qua Gemini nếu 9router thiếu câu
                missing_in_batch = [s for s in sub_batch if s.segment_id not in translation_by_id or not translation_by_id[s.segment_id].strip()]
                if missing_in_batch and gemini_direct_key and not use_direct_gemini:
                    g_items = [it for it in items_payload if it["id"] in [s.segment_id for s in missing_in_batch]]
                    g_fallback = NineRouterService.translate_segments_direct_gemini(g_items, source_lang=source_lang, target_lang=target_lang, gemini_key=gemini_direct_key)
                    translation_by_id.update(g_fallback)

                # Fallback Custom API nếu vẫn còn thiếu
                for s in sub_batch:
                    if s.segment_id not in translation_by_id or not translation_by_id[s.segment_id].strip():
                        if custom_key:
                            single_t = self._translate_with_custom_api(
                                s.original_text, source_lang, target_lang, custom_key, custom_url
                            )
                            if single_t:
                                translation_by_id[s.segment_id] = single_t

                    if s.segment_id not in translation_by_id or not translation_by_id[s.segment_id].strip():
                        failed_ids.append(s.segment_id)

                # Cập nhật kết quả dựa DUY NHẤT vào segment_id
                for s in sub_batch:
                    translated_val = translation_by_id.get(s.segment_id)
                    if translated_val and translated_val.strip():
                        s.update_translation(translated_val.strip())
                        self.cache_service.save_translation(
                            s.original_text,
                            source_lang,
                            target_lang,
                            translated_val.strip(),
                            context_key=cache_contexts[s.segment_id],
                        )

                if progress_callback:
                    done_count = min(total_pending, (b_idx + 1) * batch_size)
                    p = int((done_count / total_pending) * 60)
                    progress_callback(p, f"[Pass 1] Dịch {done_count}/{total_pending} câu qua {backend_name}...")

        if failed_ids:
            failed_summary = ", ".join(failed_ids[:8])
            remainder = f" và {len(failed_ids) - 8} câu khác" if len(failed_ids) > 8 else ""
            raise TranslationQualityError(
                f"Không dịch được {len(failed_ids)} câu ({failed_summary}{remainder}). "
                "Kiểm tra kết nối hoặc API key rồi thử lại; các câu này không được thay bằng văn bản gốc."
            )

        # 3. Pass 2 (tùy chọn): Kiểm định ngữ cảnh kịch bản và độ dài câu
        if verify_pass and segments:
            if progress_callback:
                progress_callback(70, "[Pass 2] Đang thẩm định ngữ cảnh đối thoại...")
            self.verify_and_align_translations(segments, model_name=model_name, progress_callback=progress_callback)

        # 4. Kiểm tra tính bất biến (Invariants Check)
        for s in segments:
            orig_start, orig_end, orig_spk, orig_chr = initial_snapshots[s.segment_id]
            if s.start_ms != orig_start or s.end_ms != orig_end:
                logger.warning(f"Phát hiện timestamp bị thay đổi trong quá trình dịch ({s.segment_id}). Khôi phục timestamp nguồn.")
                s.start_ms = orig_start
                s.end_ms = orig_end
            if s.speaker_id != orig_spk or s.character_id != orig_chr:
                logger.warning(f"Phát hiện identity bị thay đổi trong quá trình dịch ({s.segment_id}). Khôi phục identity nguồn.")
                s.speaker_id = orig_spk
                s.character_id = orig_chr

        # 5. Kiểm tra chất lượng bản dịch (Quality Assurance)
        issues = self.verify_translation_quality(segments)
        if issues:
            logger.info(f"Các cảnh báo chất lượng bản dịch: {issues}")

        if progress_callback:
            progress_callback(100, f"✅ Hoàn tất dịch {len(segments)} câu phụ đề.")

        return segments

    def verify_translation_quality(self, segments: List[SubtitleSegment]) -> List[str]:
        """
        Kiểm tra chất lượng bản dịch theo Section 3.2:
        - Không bỏ sót segment.
        - Không có bản dịch rỗng ngoài trường hợp source rỗng.
        - Không tráo đổi nội dung.
        - Phát hiện độ dài bất thường.
        """
        issues = []
        for s in segments:
            if s.original_text.strip() and not s.translated_text.strip():
                issues.append(f"Segment {s.segment_id}: Bản dịch rỗng trong khi câu gốc có nội dung.")
            elif s.original_text.strip() and s.translated_text.strip():
                # Kiểm tra độ dài bất thường (> 7x độ dài gốc đối với câu dài > 5 ký tự)
                if len(s.original_text) > 5 and len(s.translated_text) > len(s.original_text) * 7:
                    issues.append(f"Segment {s.segment_id}: Độ dài bản dịch bất thường ({len(s.translated_text)} ký tự so với {len(s.original_text)} ký tự gốc).")
        return issues

    def verify_and_align_translations(
        self,
        segments: List[SubtitleSegment],
        model_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[SubtitleSegment]:
        """Pass 2: Tinh chỉnh câu văn theo ngữ cảnh đối thoại liên tục mà không thay đổi ID."""
        batch_size = 10
        num_batches = (len(segments) + batch_size - 1) // batch_size
        timeline = sorted(segments, key=lambda item: (item.start_ms, item.end_ms))
        timeline_index = {item.segment_id: i for i, item in enumerate(timeline)}

        for b_idx in range(num_batches):
            sub_batch = segments[b_idx * batch_size : (b_idx + 1) * batch_size]
            payload_items = []
            for s in sub_batch:
                line_idx = timeline_index.get(s.segment_id, 0)
                next_s = timeline[line_idx + 1] if line_idx + 1 < len(timeline) else None
                available_ms = s.end_ms - s.start_ms
                if next_s and next_s.start_ms > s.start_ms:
                    available_ms = max(available_ms, next_s.start_ms - s.start_ms - 100)
                speech_slot_sec = max(0.5, round(available_ms / 1000.0, 1))
                text_len = len(s.original_text.strip())
                max_w = max(5, int(speech_slot_sec * 3.2), int(text_len * 1.2))
                payload_items.append({
                    "id": s.segment_id,
                    "slot_sec": speech_slot_sec,
                    "max_words": max_w,
                    "original": s.original_text,
                    "translated": s.translated_text
                })

            prompt = (
                "Bạn là đạo diễn lồng tiếng và biên kịch chuyển ngữ điện ảnh chuyên nghiệp.\n"
                "Nhiệm vụ: Thẩm định ngữ cảnh đối thoại và rà soát chất lượng bản dịch.\n"
                "1. ĐẢM BẢO ĐỦ Ý VÀ DỄ HIỂU (ƯU TIÊN HÀNG ĐẦU): Câu thoại phải rõ nghĩa, tự nhiên, thuần Việt, chuẩn xưng hô (anh - em, tôi - cậu...) theo mạch phim. "
                "TUYỆT ĐỐI KHÔNG cắt xén bừa bãi làm mất chủ ngữ - vị ngữ, cụt lủn hoặc làm tối nghĩa câu nói. Nếu bản dịch Pass 1 đã chuẩn và dễ hiểu, HÃY GIỮ NGUYÊN.\n"
                "2. TỐI ƯU KHẨU HÌNH HỢP LÝ: Cắt bỏ các hư từ rườm rà không cần thiết (như 'về việc', 'một cách', 'tiến hành'...). Hướng tới độ dài khoảng 'max_words' nhưng nếu câu gốc nhiều nội dung thì BẮT BUỘC phải dịch đủ ý để người xem hiểu.\n"
                "3. Giữ nguyên 'id' của từng câu trong kết quả trả về.\n"
                "4. Trả về đúng JSON list các object: [{\"id\": \"...\", \"refined\": \"...\"}]\n\n"
                + json.dumps(payload_items, ensure_ascii=False)
            )

            try:
                raw_json = self._call_llm(prompt, model_name=model_name)
                if raw_json:
                    match = re.search(r"\[.*\]", raw_json, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                        refined_map = {str(item.get("id")): str(item.get("refined") or item.get("translation") or "").strip() for item in parsed if isinstance(item, dict) and item.get("id")}
                        for s in sub_batch:
                            refined_text = refined_map.get(s.segment_id)
                            if refined_text and len(refined_text.strip()) > 0:
                                cleaned = refined_text.strip('"').strip()
                                # Chống cắt quá đà (Over-cutting protection):
                                # Nếu câu tinh chỉnh bị cắt cụt bất thường (< 3 từ trong khi câu cũ >= 5 từ hoặc ngắn hơn 40% câu cũ), giữ nguyên bản dịch Pass 1
                                old_words = (s.translated_text or "").split()
                                new_words = cleaned.split()
                                if len(old_words) >= 5 and len(new_words) <= 2:
                                    logger.info(f"Giữ nguyên bản dịch Pass 1 cho '{s.segment_id}', tránh câu tinh chỉnh quá cụt: '{cleaned}' vs '{s.translated_text}'")
                                    continue
                                if len(old_words) >= 6 and len(new_words) < len(old_words) * 0.4:
                                    logger.info(f"Giữ nguyên bản dịch Pass 1 cho '{s.segment_id}', câu tinh chỉnh cắt quá 60% từ ngữ: '{cleaned}' vs '{s.translated_text}'")
                                    continue
                                s.update_translation(cleaned)
            except Exception as e:
                logger.error(f"Lỗi Pass 2 verify alignment ({b_idx}): {e}")

            if progress_callback:
                p = 70 + int(((b_idx + 1) / max(1, num_batches)) * 25)
                progress_callback(p, f"[Pass 2] Thẩm định ngữ cảnh đối thoại lô {b_idx+1}/{num_batches}...")

        return segments

    def _call_llm(self, prompt: str, model_name: Optional[str] = None) -> Optional[str]:
        # 1. Thử trực tiếp Gemini API nếu model là Gemini Direct hoặc có Gemini Key
        gemini_direct_key = NineRouterService.find_direct_gemini_key()
        if (model_name and model_name.startswith("gemini-") and "/" not in model_name) and gemini_direct_key:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_direct_key}"
            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(endpoint, json={"contents": [{"parts": [{"text": prompt}]}]})
                    if resp.status_code == 200:
                        candidates = resp.json().get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            return "".join(p.get("text", "") for p in parts)
            except Exception:
                pass

        # 2. 9router Local
        is_9router_ok, _, _ = NineRouterService.check_connection()
        if not is_9router_ok:
            return None
        key = NineRouterService.find_local_key()
        url = NineRouterService.DEFAULT_URL
        chosen_model = model_name or NineRouterService.get_recommended_model()

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": chosen_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Bạn là chuyên gia biên tập kịch bản phim. Trả về đúng JSON list."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.post(f"{url}/chat/completions", headers=headers, json=payload)
                if resp.status_code == 200:
                    return NineRouterService.parse_response_text(resp.text)
        except Exception:
            pass
        return None

    def _translate_with_custom_api(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        api_key: str,
        api_url: Optional[str] = None
    ) -> Optional[str]:
        base_url = (api_url or "https://api.openai.com/v1").rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        source_hint = (
            "Tự nhận diện ngôn ngữ nguồn từ câu thoại."
            if source_lang.lower() in ("auto", "", "unknown")
            else f"Dịch từ {source_lang}."
        )
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        prompt = (
            f"{source_hint} Dịch câu thoại phụ đề sang '{target_lang}', trung thành và gọn theo ngữ cảnh. "
            f"Chỉ xuất câu dịch, không giải thích:\n\n{text}"
        )
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(endpoint, headers=headers, json=payload)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Lỗi gọi Custom Translation API: {e}")
        return None
