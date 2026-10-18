"""Google Gemini TTS provider with chunking, retries, and audio stitching.

Supports:
- Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts`).
- Gemini 3.8 Flash-Lite TTS (`gemini-3.8-flash-lite-tts`).
- Text chunking, retries, and audio stitching.
- Seamless audio stitching with micro-silence between sentences.
- Multi-voice persona mapping & style inflection (`speech_metadata.style`).
"""

import os
import re
import io
import time
import base64
import logging
import threading
from typing import List, Dict, Optional, Tuple, Any, Set
import httpx
import soundfile as sf
import numpy as np

from ...domain.enums import SpeakerGender, SpeakerAgeGroup, VoiceStyle, VoiceCharacter

logger = logging.getLogger("vtranslator.tts.gemini")

# Các model TTS thế hệ mới của Gemini
DEFAULT_GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"
GEMINI_TTS_LITE_MODEL = "gemini-3.8-flash-lite-tts"
GEMINI_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

# Danh mục giọng đọc Prebuilt chuẩn của Gemini TTS
GEMINI_PREBUILT_VOICES = {
    # Giọng Nam
    "Puck": {"gender": SpeakerGender.MALE, "desc": "Nam · Ấm áp, tự nhiên, sinh động"},
    "Charon": {"gender": SpeakerGender.MALE, "desc": "Nam · Trầm ấm, dõng dạc, tin tức/nghiêm túc"},
    "Orus": {"gender": SpeakerGender.MALE, "desc": "Nam · Dày, uy lực, điềm đạm"},
    "Fenrir": {"gender": SpeakerGender.MALE, "desc": "Nam · Trẻ trung, sôi nổi, năng động"},
    "Gacrux": {"gender": SpeakerGender.MALE, "desc": "Nam · Trưởng thành, sâu lắng"},
    "Achernar": {"gender": SpeakerGender.MALE, "desc": "Nam · Mộc mạc, nhẹ nhàng, tự sự"},
    "Algieba": {"gender": SpeakerGender.MALE, "desc": "Nam · Êm ái, mượt mà"},
    "Iapetus": {"gender": SpeakerGender.MALE, "desc": "Nam · Rõ ràng, dứt khoát"},
    "Sadaltager": {"gender": SpeakerGender.MALE, "desc": "Nam · Tri thức, đĩnh đạc"},
    # Giọng Nữ
    "Kore": {"gender": SpeakerGender.FEMALE, "desc": "Nữ · Thanh lịch, tự tin, chuẩn mực"},
    "Zephyr": {"gender": SpeakerGender.FEMALE, "desc": "Nữ · Trong trẻo, tươi sáng, linh hoạt"},
    "Sulafat": {"gender": SpeakerGender.FEMALE, "desc": "Nữ · Ấm áp, truyền cảm kể chuyện"},
    "Aoede": {"gender": SpeakerGender.FEMALE, "desc": "Nữ · Nhẹ nhàng, tự nhiên, trong sáng"},
    "Leda": {"gender": SpeakerGender.FEMALE, "desc": "Nữ · Trẻ trung, đáng yêu"},
    "Vindemiatrix": {"gender": SpeakerGender.FEMALE, "desc": "Nữ · Dịu ngọt, đằm thắm"},
    "Autonoe": {"gender": SpeakerGender.FEMALE, "desc": "Nữ · Sáng, hoạt bát"},
    "Callirrhoe": {"gender": SpeakerGender.FEMALE, "desc": "Nữ · Dễ mến, tự nhiên"},
    "Despina": {"gender": SpeakerGender.FEMALE, "desc": "Nữ · Êm dịu, mượt mà"},
    "Erinome": {"gender": SpeakerGender.FEMALE, "desc": "Nữ · Rõ ràng, mạch lạc"},
}

# Ánh xạ tên nhân vật tiếng Việt sang giọng Gemini và sắc thái
VIETNAMESE_VOICE_TO_GEMINI: Dict[str, Dict[str, str]] = {
    "Minh Đức": {"voice": "Charon", "style": "chuyên nghiệp, dõng dạc, tin tức"},
    "Phạm Tuyên": {"voice": "Puck", "style": "tự nhiên, thân thiện và ấm áp"},
    "Thái Sơn": {"voice": "Orus", "style": "trầm ấm, sâu lắng kể chuyện"},
    "Xuân Vĩnh": {"voice": "Puck", "style": "ấm áp, chân thành"},
    "Thanh Bình": {"voice": "Charon", "style": "điềm đạm, truyền cảm"},
    "Trúc Ly": {"voice": "Zephyr", "style": "trong trẻo, tươi sáng và tự nhiên"},
    "Ngọc Linh": {"voice": "Sulafat", "style": "dịu dàng, truyền cảm kể chuyện"},
    "Đoan Trang": {"voice": "Kore", "style": "thanh lịch, trang nhã và rõ ràng"},
    "Mai Anh": {"voice": "Zephyr", "style": "nhanh nhẹn, phong cách tin tức"},
    "Thục Đoan": {"voice": "Sulafat", "style": "ngọt ngào, sâu lắng"},
    "Minh Triết": {"voice": "Puck", "style": "dõng dạc, rõ ràng"},
    "Thùy Dung": {"voice": "Kore", "style": "chuẩn mực, tự tin"},
    "Quang Sơn": {"voice": "Achernar", "style": "mộc mạc, bình dị"},
    "Ngọc Trân": {"voice": "Vindemiatrix", "style": "dịu dàng, đằm thắm"},
    "Mỹ Duyên": {"voice": "Sulafat", "style": "truyền cảm, diễn cảm sâu sắc"},
    "Quỳnh Anh": {"voice": "Aoede", "style": "nhẹ nhàng, trong sáng"},
    "Đức Trí": {"voice": "Gacrux", "style": "trầm ấm, già dặn"},
    "Kim Thanh": {"voice": "Sulafat", "style": "ngọt ngào, ấm cúng"},
    "Ngọc Huyền": {"voice": "Zephyr", "style": "tự nhiên, chuẩn giọng"},
    "Adam": {"voice": "Fenrir", "style": "năng động, trẻ trung, hiện đại"},
    "Mạnh Dũng": {"voice": "Orus", "style": "khỏe khoắn, mạnh mẽ"},
    "Minh Quân": {"voice": "Puck", "style": "năng động, tươi vui"},
    "Anh Khôi": {"voice": "Achernar", "style": "sâu lắng, tự sự"},
}


class GeminiTtsProvider:
    """Provider thực thi sinh giọng đọc chất lượng cao bằng Gemini TTS có xoay vòng đa API Key, chống Timeout & Rate Limit."""

    _last_request_time: float = 0.0
    _key_lock = threading.Lock()
    _key_last_request_time: Dict[str, float] = {}
    _key_cooldown_until: Dict[str, float] = {}

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        model_name: str = DEFAULT_GEMINI_TTS_MODEL,
        custom_endpoint: Optional[str] = None,
        max_chunk_chars: int = 150,
        request_timeout_sec: float = 20.0,
        max_retries: int = 5,
        min_request_interval_sec: float = 6.0,
    ):
        raw_keys: List[str] = []
        if api_keys:
            raw_keys.extend(api_keys)
        if api_key:
            # Cho phép api_key là chuỗi chứa nhiều key cách nhau bằng dấu phẩy hoặc xuống dòng
            for k in re.split(r'[,;\n]+', api_key):
                if k.strip():
                    raw_keys.append(k.strip())

        if not raw_keys:
            env_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
            for k in re.split(r'[,;\n]+', env_keys):
                if k.strip():
                    raw_keys.append(k.strip())

        seen = set()
        self.api_keys: List[str] = []
        for k in raw_keys:
            cleaned = k.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                self.api_keys.append(cleaned)

        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.model_name = model_name or DEFAULT_GEMINI_TTS_MODEL
        self.custom_endpoint = custom_endpoint
        self.max_chunk_chars = max(50, min(max_chunk_chars, 400))
        self.request_timeout_sec = request_timeout_sec
        self.max_retries = max(5, max_retries)
        self.min_request_interval_sec = min_request_interval_sec

    def get_available_key(
        self,
        exclude_keys: Optional[Set[str]] = None,
        preferred_key: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Chọn một API Key sẵn sàng từ pool:
        - Ưu tiên preferred_key nếu được chỉ định và chưa bị cooldown.
        - Tránh các key đang bị 429 cooldown (nếu có key khác không bị cooldown).
        - Đảm bảo mỗi key độc lập giãn cách >= min_request_interval_sec.
        - Trả về (key, sleep_needed).
        """
        if not self.api_keys:
            return "", 0.0

        exclude = exclude_keys or set()
        candidates = [k for k in self.api_keys if k not in exclude]
        if not candidates:
            candidates = self.api_keys

        now = time.time()
        with self._key_lock:
            # 1. Nếu có preferred_key trong candidates và chưa bị cooldown
            if preferred_key and preferred_key in candidates:
                cooldown_exp = self._key_cooldown_until.get(preferred_key, 0.0)
                if now >= cooldown_exp:
                    last_req = self._key_last_request_time.get(preferred_key, 0.0)
                    elapsed = now - last_req
                    sleep_needed = max(0.0, self.min_request_interval_sec - elapsed) if last_req > 0 else 0.0
                    return preferred_key, sleep_needed

            # 2. Lọc các candidate không bị cooldown
            active_candidates = [k for k in candidates if now >= self._key_cooldown_until.get(k, 0.0)]
            if not active_candidates:
                active_candidates = candidates

            # 3. Tìm key có sleep_needed nhỏ nhất
            best_key = active_candidates[0]
            min_sleep = float("inf")
            for k in active_candidates:
                last_req = self._key_last_request_time.get(k, 0.0)
                elapsed = now - last_req
                sleep = max(0.0, self.min_request_interval_sec - elapsed) if last_req > 0 else 0.0
                if sleep < min_sleep:
                    min_sleep = sleep
                    best_key = k
                    if sleep == 0.0:
                        break

            return best_key, min_sleep

    def mark_key_used(self, key: str) -> None:
        """Ghi nhận thời điểm sử dụng key để tính giãn cách nhịp nghỉ độc lập."""
        if not key:
            return
        with self._key_lock:
            self._key_last_request_time[key] = time.time()
            GeminiTtsProvider._last_request_time = time.time()

    def mark_key_429(self, key: str, cooldown_sec: float = 20.0) -> None:
        """Đưa key vào danh sách cooldown tạm thời khi gặp lỗi 429."""
        if not key:
            return
        with self._key_lock:
            self._key_cooldown_until[key] = time.time() + cooldown_sec
            masked = key[:6] + "..." + key[-4:] if len(key) > 10 else key
            logger.warning(f"⚠️ Key {masked} bị Rate Limit (429)! Tạm thời khóa {cooldown_sec}s để hồi phục quota.")

    @staticmethod
    def clean_text_for_tts(text: str) -> str:
        """Làm sạch câu và bổ sung dấu câu kết thúc để tránh nuốt âm cuối."""
        t = text.strip()
        # Bỏ dấu ngoặc kép ở đầu và đuôi
        t = re.sub(r'^["\'“”‘’]+|["\'“”‘’]+$', '', t).strip()
        # Preserve bracketed dialogue; acting directions are stored separately.
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    @classmethod
    def split_text_into_chunks(cls, text: str, max_chars: int = 150) -> List[str]:
        """
        Chia nhỏ văn bản theo dấu câu và ranh giới từ để giảm rủi ro timeout:
        1. Ưu tiên ngắt tại dấu chấm câu chính (. ! ? ... … hoặc xuống dòng).
        2. Nếu một câu vẫn quá dài, ngắt tại dấu vế câu (, ; : —).
        3. Nếu một đoạn không có dấu câu mà vẫn dài, ngắt theo ranh giới từ.
        4. Gộp các mẩu ngắn lại sao cho tổng chiều dài <= max_chars.
        5. Thêm dấu câu kết thúc khi cần để câu đọc không bị cụt.
        """
        clean = cls.clean_text_for_tts(text)
        if not clean:
            return []
        if len(clean) <= max_chars:
            # Thêm dấu chấm nếu chưa có dấu ngắt câu cuối
            if not clean.endswith(('.', '!', '?', '...', '…', ':', ';', ',')):
                clean += '.'
            return [clean]

        # 1. Tách theo ranh giới câu chính
        raw_sentences = re.split(r'([.!?…\n]+)', clean)
        sentence_units = []
        i = 0
        while i < len(raw_sentences):
            s = raw_sentences[i].strip()
            punct = raw_sentences[i + 1].strip() if (i + 1 < len(raw_sentences)) else ""
            full_s = f"{s}{punct}".strip() if punct else s
            if full_s:
                sentence_units.append(full_s)
            i += 2

        # 2. Xử lý các câu dài hơn max_chars bằng cách ngắt tiếp theo vế câu (, ; : —)
        fine_units = []
        for s in sentence_units:
            if len(s) <= max_chars:
                fine_units.append(s)
            else:
                # Ngắt theo vế câu
                clause_parts = re.split(r'([,;:—–]+)', s)
                j = 0
                while j < len(clause_parts):
                    cp = clause_parts[j].strip()
                    cp_p = clause_parts[j + 1].strip() if (j + 1 < len(clause_parts)) else ""
                    full_cp = f"{cp}{cp_p}".strip() if cp_p else cp
                    if not full_cp:
                        j += 2
                        continue

                    # Nếu một vế vẫn dài hơn max_chars, ngắt theo từ
                    if len(full_cp) <= max_chars:
                        fine_units.append(full_cp)
                    else:
                        words = full_cp.split()
                        curr_words = []
                        curr_len = 0
                        for w in words:
                            if curr_len + len(w) + 1 > max_chars and curr_words:
                                fine_units.append(" ".join(curr_words))
                                curr_words = [w]
                                curr_len = len(w)
                            else:
                                curr_words.append(w)
                                curr_len += len(w) + 1
                        if curr_words:
                            fine_units.append(" ".join(curr_words))
                    j += 2

        # 3. Gom các mẩu nhỏ lại thành từng chunk hợp lý <= max_chars
        chunks: List[str] = []
        curr_chunk = ""
        for unit in fine_units:
            unit = unit.strip()
            if not unit:
                continue

            if not curr_chunk:
                curr_chunk = unit
            elif len(curr_chunk) + len(unit) + 1 <= max_chars:
                # Nối tiếp với dấu cách
                curr_chunk = f"{curr_chunk} {unit}"
            else:
                chunks.append(curr_chunk)
                curr_chunk = unit

        if curr_chunk:
            chunks.append(curr_chunk)

        # 4. Đảm bảo từng chunk có dấu câu kết thúc phù hợp để phát âm tự nhiên
        final_chunks: List[str] = []
        for idx, c in enumerate(chunks):
            c_str = c.strip()
            if not c_str:
                continue
            if not c_str.endswith(('.', '!', '?', '...', '…', ':', ';', ',')):
                # Nếu là chunk cuối thì dùng dấu chấm, chunk giữa thì dùng dấu phẩy
                c_str += '.' if idx == len(chunks) - 1 else ','
            final_chunks.append(c_str)

        return final_chunks

    @classmethod
    def resolve_voice_and_style(
        cls,
        voice_id_or_name: str,
        gender: Optional[SpeakerGender] = None,
        style: Optional[VoiceStyle] = None,
        voice_character: Optional[VoiceCharacter] = None,
    ) -> Tuple[str, str]:
        """
        Xác định giọng đọc Gemini Prebuilt phù hợp và phong cách diễn đọc (style).
        - Nếu truyền trực tiếp tên giọng Gemini (ví dụ: Kore, Puck, Zephyr...), dùng trực tiếp.
        - Nếu truyền tên giọng VieNeu tiếng Việt, ánh xạ theo từ điển chuẩn.
        - Nếu không rõ, căn cứ theo Giới tính và Đặc tính giọng nhân vật.
        """
        v_clean = voice_id_or_name.strip()

        # 1. Trùng trực tiếp với giọng Prebuilt của Gemini (bao gồm cả dạng 'Gemini Puck', 'Gemini Kore')
        for g_name in GEMINI_PREBUILT_VOICES:
            if g_name.lower() == v_clean.lower() or f"gemini {g_name.lower()}" == v_clean.lower():
                s_desc = "tự nhiên, truyền cảm và rõ ràng"
                return g_name, s_desc

        # 2. Ánh xạ từ danh mục 23 giọng VieNeu
        if v_clean in VIETNAMESE_VOICE_TO_GEMINI:
            cfg = VIETNAMESE_VOICE_TO_GEMINI[v_clean]
            return cfg["voice"], cfg["style"]

        for vn_key, cfg in VIETNAMESE_VOICE_TO_GEMINI.items():
            if vn_key.lower() == v_clean.lower():
                return cfg["voice"], cfg["style"]

        # 3. Phân loại theo Giới tính & Đặc tính
        is_female = False
        if gender == SpeakerGender.FEMALE:
            is_female = True
        elif "nữ" in v_clean.lower() or "female" in v_clean.lower():
            is_female = True

        style_hint = "tự nhiên và rõ ràng"
        if voice_character == VoiceCharacter.DEEP:
            selected_voice = "Orus" if not is_female else "Kore"
            style_hint = "trầm ấm, sâu lắng"
        elif voice_character == VoiceCharacter.BRIGHT:
            selected_voice = "Zephyr" if is_female else "Fenrir"
            style_hint = "tươi sáng, linh hoạt"
        elif voice_character == VoiceCharacter.SOFT:
            selected_voice = "Sulafat" if is_female else "Achernar"
            style_hint = "dịu dàng, nhẹ nhàng"
        elif voice_character == VoiceCharacter.ENERGETIC:
            selected_voice = "Fenrir" if not is_female else "Zephyr"
            style_hint = "sôi nổi, tràn đầy năng lượng"
        elif voice_character == VoiceCharacter.CALM:
            selected_voice = "Kore" if is_female else "Charon"
            style_hint = "điềm đạm, trang nhã"
        else:
            selected_voice = "Kore" if is_female else "Puck"

        if style == VoiceStyle.DOC_TRUYEN:
            style_hint += ", phong cách kể chuyện truyền cảm"
        elif style == VoiceStyle.TIN_TUC:
            style_hint += ", phong cách tin tức dõng dạc chuẩn xác"

        return selected_voice, style_hint

    def build_payload(self, text_chunk: str, voice_name: str, style_desc: Optional[str] = None) -> Dict[str, Any]:
        """Tạo payload chuẩn cho API Gemini TTS (generateContent)."""
        part_content: Dict[str, Any] = {"text": text_chunk}
        if style_desc:
            part_content["speech_metadata"] = {"style": style_desc}

        return {
            "contents": [{
                "role": "user",
                "parts": [part_content]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice_name
                        }
                    }
                }
            }
        }

    def synthesize_chunk_wav_bytes(
        self,
        text_chunk: str,
        voice_name: str,
        style_desc: Optional[str] = None,
        specific_key: Optional[str] = None
    ) -> bytes:
        """
        Gọi API sinh âm thanh cho 1 chunk ngắn với timeout riêng, xoay vòng API Key và retry chống lỗi mạng.
        Trả về bytes của file WAV (chuẩn RIFF 24kHz).
        """
        if not self.api_keys and not self.api_key and not specific_key:
            raise ValueError("Chưa cấu hình Gemini API Key cho TTS!")

        payload = self.build_payload(text_chunk, voice_name, style_desc)
        url = self.custom_endpoint or f"{GEMINI_API_ENDPOINT}/{self.model_name}:generateContent"

        last_error = None
        excluded_keys: Set[str] = set()

        for attempt in range(1, self.max_retries + 1):
            # Lấy key tối ưu từ pool
            current_key, sleep_needed = self.get_available_key(
                exclude_keys=excluded_keys, preferred_key=specific_key
            )
            if not current_key:
                current_key = self.api_key or specific_key

            if sleep_needed > 0:
                time.sleep(sleep_needed)

            self.mark_key_used(current_key)

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": current_key
            }

            try:
                with httpx.Client(timeout=self.request_timeout_sec) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            raise ValueError(f"Gemini API không trả về candidate nào: {data}")

                        parts = candidates[0].get("content", {}).get("parts", [])
                        audio_base64 = None
                        for p in parts:
                            inline_data = p.get("inlineData") or p.get("inline_data")
                            if inline_data and "data" in inline_data:
                                audio_base64 = inline_data["data"]
                                break

                        if not audio_base64:
                            raise ValueError(f"Không tìm thấy dữ liệu audio trong phản hồi: {data}")

                        return base64.b64decode(audio_base64)

                    elif resp.status_code == 429:
                        err_text = resp.text.lower()
                        # Detect daily-limit wording without assuming a fixed account quota.
                        is_daily_limit = any(term in err_text for term in ["per day", "perday", "daily", "day quota"])
                        retry_after = resp.headers.get("Retry-After", "").strip()
                        try:
                            cooldown = max(1.0, float(retry_after))
                        except (TypeError, ValueError):
                            cooldown = 43200.0 if is_daily_limit else 25.0

                        masked = current_key[:6] + "..." + current_key[-4:] if len(current_key) > 10 else current_key
                        if is_daily_limit:
                            logger.warning(
                                f"Key {masked} vượt hạn mức trong ngày do nhà cung cấp trả về; "
                                f"tạm ngưng key {cooldown:.0f}s và thử key khác nếu có."
                            )
                        else:
                            logger.warning(
                                f"Key {masked} bị giới hạn yêu cầu; tạm ngưng {cooldown:.0f}s rồi thử lại."
                            )

                        self.mark_key_429(current_key, cooldown_sec=cooldown)
                        excluded_keys.add(current_key)

                        # Nếu còn key khác chưa bị 429 trong pool, ngay lập tức thử lại bằng key khác
                        remaining_keys = [k for k in self.api_keys if k not in excluded_keys]
                        if remaining_keys:
                            logger.info(
                                f"⚡ Chuyển ngay sang API Key khác trong pool ({len(remaining_keys)} key khả dụng) "
                                f"cho chunk '{text_chunk[:25]}...'..."
                            )
                            continue

                        # Nếu tất cả key đều bị 429, backoff chờ
                        wait_time = max(10.0, attempt * 10.0)
                        logger.warning(
                            f"⚠️ Toàn bộ API Key đều chạm Rate Limit (429)! Đang nghỉ {wait_time}s "
                            f"(Thử lại lần {attempt}/{self.max_retries})..."
                        )
                        time.sleep(wait_time)
                        continue

                    elif resp.status_code in (500, 502, 503, 504):
                        wait_time = attempt * 2.0
                        logger.warning(f"Lỗi Gemini TTS ({resp.status_code}), thử lại sau {wait_time}s (Lần {attempt}/{self.max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"Gemini TTS lỗi HTTP {resp.status_code}: {resp.text}")

            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                last_error = net_err
                wait_time = attempt * 1.2
                logger.warning(f"Timeout/Network error khi gọi Gemini TTS cho chunk '{text_chunk[:25]}...': {net_err}. Thử lại lần {attempt}/{self.max_retries}...")
                time.sleep(wait_time)
            except Exception as e:
                last_error = e
                logger.error(f"Lỗi không xác định khi sinh chunk TTS ({attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    raise

        raise RuntimeError(f"Không thể sinh âm thanh qua Gemini TTS sau {self.max_retries} lần thử: {last_error}")

    @staticmethod
    def stitch_audio_chunks(
        wav_bytes_list: List[bytes],
        output_wav_path: str,
        micro_silence_ms: int = 70,
        target_sample_rate: int = 48000
    ) -> str:
        """
        Ghép nối liền mạch các chunk WAV trong bộ nhớ thành một file WAV hoàn chỉnh:
        - Đọc audio mảng số học float32.
        - Chèn micro-silence tự nhiên (70ms) giữa các câu để tạo khoảng ngắt thở người thật.
        - Chuyển đổi và lưu WAV 48kHz / 16-bit PCM chất lượng cao.
        """
        if not wav_bytes_list:
            return ""

        audio_arrays: List[np.ndarray] = []
        base_sr = 24000

        for b in wav_bytes_list:
            if not b:
                continue
            try:
                data, sr = sf.read(io.BytesIO(b), dtype="float32")
                base_sr = sr
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)  # Chuyển mono
                audio_arrays.append(data)
            except Exception as e:
                logger.warning(f"Không thể đọc chunk wav bytes: {e}")

        if not audio_arrays:
            return ""

        # Tạo micro-silence đệm giữa các chunk
        pause_samples = int(base_sr * (micro_silence_ms / 1000.0))
        silence_pad = np.zeros(pause_samples, dtype=np.float32)

        stitched_parts: List[np.ndarray] = []
        for idx, arr in enumerate(audio_arrays):
            stitched_parts.append(arr)
            # Thêm khoảng đệm giữa các câu, câu cuối không thêm
            if idx < len(audio_arrays) - 1 and pause_samples > 0:
                stitched_parts.append(silence_pad)

        full_audio = np.concatenate(stitched_parts)

        # Thêm 100ms đệm yên lặng ở đuôi file để chống nuốt âm cuối và khựng khi phát
        tail_pad_samples = int(base_sr * 0.10)
        full_audio = np.pad(full_audio, (0, tail_pad_samples), mode="constant")

        os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)

        # Lưu file WAV chất lượng cao
        sf.write(output_wav_path, full_audio, base_sr, subtype="PCM_16")
        return output_wav_path

    def synthesize(
        self,
        text: str,
        voice_name: str = "Puck",
        output_wav_path: str = "",
        style_desc: Optional[str] = None,
        specific_key: Optional[str] = None
    ) -> str:
        """
        Hàm thực thi chính: Chia nhỏ câu chống Timeout, gọi API sinh từng phần, ghép nối mượt mà và lưu file.
        Hỗ trợ chỉ định specific_key hoặc tự động xoay vòng đa key từ pool.
        """
        clean_text = self.clean_text_for_tts(text)
        if not clean_text:
            return ""

        chunks = self.split_text_into_chunks(clean_text, max_chars=self.max_chunk_chars)
        if not chunks:
            return ""

        resolved_voice, resolved_style = self.resolve_voice_and_style(voice_name)
        final_style = style_desc or resolved_style

        # Nếu chỉ có 1 chunk duy nhất
        if len(chunks) == 1:
            wav_bytes = self.synthesize_chunk_wav_bytes(
                chunks[0], resolved_voice, final_style, specific_key=specific_key
            )
            return self.stitch_audio_chunks([wav_bytes], output_wav_path)

        # Nhiều chunks: sinh từng chunk và ghép nối
        chunk_wav_bytes: List[bytes] = []
        for c_idx, chunk in enumerate(chunks):
            logger.info(f"Đang sinh Gemini TTS chunk {c_idx+1}/{len(chunks)} ({len(chunk)} ký tự): '{chunk}'")
            b = self.synthesize_chunk_wav_bytes(
                chunk, resolved_voice, final_style, specific_key=specific_key
            )
            chunk_wav_bytes.append(b)

        return self.stitch_audio_chunks(chunk_wav_bytes, output_wav_path)
