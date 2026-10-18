"""9router Local Service Integration (http://127.0.0.1:20128/v1)."""

import os
import re
import json
import sqlite3
import httpx
from typing import Tuple, List, Optional, Dict, Any


class NineRouterService:
    DEFAULT_PORT = 20128
    DEFAULT_URL = "http://127.0.0.1:20128/v1"
    RECOMMENDED_FAST_MODELS = [
        "cx/gpt-5.6-terra",
        "cx/gpt-5.6-luna",
        "cx/gpt-5.5",
        "cx/gpt-5.4",
        "tokenharbor/deepseek-v4.1-flash:free",
        "ag/gemini-3.8-flash-low",
        "ag/gemini-3.8-flash",
    ]
    DEFAULT_MODEL = "cx/gpt-5.6-terra"

    RECOMMENDED_GEMINI_MODELS = [
        "ag/gemini-3.8-flash-low",
        "ag/gemini-3.8-flash",
        "ag/gemini-3.8-flash-medium",
        "ag/gemini-3.7-flash-low",
        "ag/gemini-3.7-flash",
    ]
    DEFAULT_GEMINI_MODEL = "ag/gemini-3.8-flash-low"

    @classmethod
    def find_direct_gemini_key(cls) -> str:
        """Đọc Gemini API Key trực tiếp từ Keychain hoặc biến môi trường."""
        try:
            from .keychain_service import KeychainService
            kc = KeychainService()
            # TTS presets are local; only use the separately configured Gemini key
            # for optional translation/analysis fallback.
            raw_keys = kc.get_secret("gemini_api_key")
            if not raw_keys:
                raw_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
            if raw_keys:
                keys = [k.strip() for k in re.split(r"[,;\n]+", raw_keys) if k.strip()]
                if keys:
                    return keys[0]
        except Exception:
            pass
        return ""

    @classmethod
    def get_recommended_gemini_model(cls, available_models: Optional[List[str]] = None) -> str:
        """Tìm model Gemini tốt nhất, nhanh nhất hiện có trong 9router."""
        if available_models is None:
            ok, _, available_models = cls.check_connection()
            if not ok or not available_models:
                return cls.DEFAULT_GEMINI_MODEL

        available_set = set(available_models)
        for cand in cls.RECOMMENDED_GEMINI_MODELS:
            if cand in available_set:
                return cand

        for m in available_models:
            if "gemini" in m.lower() and not m.startswith("gemini-web/"):
                return m

        return cls.DEFAULT_GEMINI_MODEL

    @classmethod
    def get_recommended_model(cls, available_models: Optional[List[str]] = None, prefer_gemini: bool = False) -> str:
        """Tìm model tốt nhất, nhanh nhất hiện có trong 9router."""
        if prefer_gemini:
            return cls.get_recommended_gemini_model(available_models)

        if available_models is None:
            ok, _, available_models = cls.check_connection()
            if not ok or not available_models:
                return cls.DEFAULT_MODEL

        available_set = set(available_models)
        for cand in cls.RECOMMENDED_FAST_MODELS:
            if cand in available_set:
                return cand

        for m in available_models:
            if m.startswith("cx/") or m.startswith("tokenharbor/"):
                return m

        return available_models[0] if available_models else cls.DEFAULT_MODEL

    @classmethod
    def sort_models_by_recommendation(cls, models: List[str]) -> List[str]:
        """Sắp xếp danh sách model với các model nhanh & ổn định nhất lên đầu."""
        if not models:
            return []

        def sort_key(m: str):
            for idx, cand in enumerate(cls.RECOMMENDED_FAST_MODELS):
                if m == cand:
                    return (0, idx)
            if m.startswith("cx/"):
                return (1, m)
            if m.startswith("tokenharbor/"):
                return (2, m)
            if m.startswith("gemini-web/"):
                return (3, m)
            if m.startswith("ag/"):
                return (4, m)
            return (5, m)

        return sorted(models, key=sort_key)

    @classmethod
    def find_local_key(cls) -> str:
        """Tự động đọc API Key hợp lệ từ cơ sở dữ liệu SQLite của 9router."""
        db_path = os.path.expanduser("~/.9router/db/data.sqlite")
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path, timeout=3.0)
                cur = conn.cursor()
                cur.execute("SELECT key FROM apiKeys WHERE isActive = 1 ORDER BY createdAt DESC LIMIT 1;")
                row = cur.fetchone()
                conn.close()
                if row and row[0]:
                    return row[0].strip()
            except Exception:
                pass

        cli_secret_path = os.path.expanduser("~/.9router/auth/cli-secret")
        if os.path.exists(cli_secret_path):
            try:
                with open(cli_secret_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
        return ""

    @classmethod
    def parse_response_text(cls, resp_text: str) -> str:
        """Xử lý linh hoạt cả định dạng JSON chuẩn lẫn SSE stream (data: {...}) từ 9router."""
        if not resp_text:
            return ""

        # 1. Thử parse JSON chuẩn trước
        try:
            data = json.loads(resp_text)
            if isinstance(data, dict):
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    if "content" in msg and msg["content"]:
                        return msg["content"].strip()
                    delta = choices[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        return delta["content"].strip()
        except Exception:
            pass

        # 2. Parse định dạng SSE streaming (data: {...})
        full_chunks = []
        for line in resp_text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                chunk_str = line[5:].strip()
                if chunk_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(chunk_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if not content and "message" in choices[0]:
                            content = choices[0]["message"].get("content", "")
                        if content:
                            full_chunks.append(content)
                except Exception:
                    pass

        return "".join(full_chunks).strip()

    @classmethod
    def check_connection(cls, url: Optional[str] = None, api_key: Optional[str] = None) -> Tuple[bool, str, List[str]]:
        """Kiểm tra kết nối tới 9router và lấy danh sách mô hình được sắp xếp tối ưu."""
        base_url = (url or cls.DEFAULT_URL).rstrip("/")
        key = api_key or cls.find_local_key()

        endpoint = f"{base_url}/models"
        headers = {"Authorization": f"Bearer {key}"} if key else {}

        try:
            with httpx.Client(timeout=4.0) as client:
                resp = client.get(endpoint, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                    sorted_models = cls.sort_models_by_recommendation(raw_models)
                    return True, "✅ Kết nối 9router Local thành công!", sorted_models
                elif resp.status_code in (401, 403):
                    return False, "❌ 9router phản hồi 401: API Key không chính xác.", []
                else:
                    return False, f"Mã trạng thái {resp.status_code}", []
        except Exception as e:
            return False, f"Không thể kết nối 9router ({e}). Hãy đảm bảo 9router đang chạy.", []

    @classmethod
    def test_model_speed(
        cls,
        model_name: str,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0
    ) -> Tuple[bool, float, str]:
        """Đo tốc độ phản hồi thực tế của một model (9router hoặc Gemini Direct)."""
        import time

        # 1. Direct Google Gemini API
        if (model_name.startswith("gemini-") and "/" not in model_name) or model_name.startswith("gemini_direct"):
            clean_name = model_name.replace("gemini_direct:", "").strip()
            if clean_name.startswith("gemini_direct"):
                clean_name = "gemini-3.5-flash-lite"
            g_key = cls.find_direct_gemini_key()
            if not g_key:
                return False, 0.0, "Chưa tìm thấy Gemini API Key trong Keychain"
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_name}:generateContent?key={g_key}"
            payload = {"contents": [{"parts": [{"text": "Hello"}]}]}
            t0 = time.time()
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(endpoint, json=payload)
                    latency = round(time.time() - t0, 2)
                    if resp.status_code == 200:
                        return True, latency, f"✅ Phản hồi trong {latency}s"
                    return False, latency, f"HTTP {resp.status_code}: {resp.text[:60]}"
            except Exception as e:
                latency = round(time.time() - t0, 2)
                return False, latency, f"Timeout/Lỗi ({latency}s): {e}"

        # 2. 9router Local
        base_url = (url or cls.DEFAULT_URL).rstrip("/")
        key = api_key or cls.find_local_key()
        endpoint = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "1+1=?"}],
            "max_tokens": 10,
            "stream": False
        }
        t0 = time.time()
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(endpoint, headers=headers, json=payload)
                latency = round(time.time() - t0, 2)
                if resp.status_code == 200:
                    text_out = cls.parse_response_text(resp.text)
                    return True, latency, f"✅ Phản hồi trong {latency}s ({text_out[:20]})"
                return False, latency, f"HTTP {resp.status_code}: {resp.text[:60]}"
        except Exception as e:
            latency = round(time.time() - t0, 2)
            return False, latency, f"Timeout/Lỗi ({latency}s): {e}"

    @classmethod
    def translate_segments_direct_gemini(
        cls,
        items: List[Dict[str, Any]],
        source_lang: str = "auto",
        target_lang: str = "vi",
        model: str = "gemini-3.5-flash-lite",
        gemini_key: Optional[str] = None
    ) -> Dict[str, str]:
        """Translate a batch directly through the configured Gemini API model."""
        key = gemini_key or cls.find_direct_gemini_key()
        if not key or not items:
            return {}

        clean_model = model.replace("gemini_direct:", "").strip()
        if not clean_model or clean_model == "gemini_direct":
            clean_model = "gemini-3.5-flash-lite"

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={key}"
        items_json = json.dumps(items, ensure_ascii=False)
        source_hint = (
            "Tự nhận diện ngôn ngữ nguồn riêng cho từng câu."
            if source_lang.lower() in ("auto", "", "unknown")
            else f"Ngôn ngữ nguồn: {source_lang}."
        )
        prompt = (
            f"Bạn là chuyên gia biên kịch và dịch thuật lồng tiếng phim chuyên nghiệp.\n"
            f"Nhiệm vụ: {source_hint} Dịch từng câu sang tiếng Việt ('{target_lang}') tự nhiên, chuẩn nghĩa và đại từ xưng hô phù hợp với ngữ cảnh nhân vật ('speaker', 'gender', 'role').\n"
            f"QUY TẮC DỊCH THUẬT PHIM ĐIỆN ẢNH:\n"
            f"1. ĐỦ Ý VÀ DỄ HIỂU LÀ ƯU TIÊN SỐ 1: Câu dịch phải thuần Việt, trôi chảy, có đầy đủ ngữ cảnh để người xem hiểu trọn vẹn nội dung. TUYỆT ĐỐI KHÔNG vì ép ngắn mà cắt xén què cụt, bỏ mất chủ ngữ - vị ngữ làm người xem không hiểu!\n"
            f"2. VĂN PHONG GÃY GỌN: Tránh các hư từ rườm rà không cần thiết (về việc, một cách, tiến hành...). Hướng tới độ dài gọn gàng quanh 'max_words' để vừa vặn khẩu hình, nhưng NẾU CÂU GỐC NHIỀU Ý THÌ BẮT BUỘC PHẢI DỊCH ĐỦ Ý RÕ RÀNG.\n"
            f"3. Không thêm thông tin, lời chửi thề hoặc cường điệu không có trong bản gốc. Nếu câu nguồn không rõ, giữ nguyên phần chưa thể xác định và không tự bịa.\n"
            f"4. BẮT BUỘC giữ nguyên 'id' của từng câu và chỉ trả về đúng JSON list các object: [{{\"id\": \"...\", \"translation\": \"...\"}}]\n\n"
            + items_json
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(endpoint, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts)
                        res_map = cls._extract_id_translations(text)
                        if res_map:
                            return res_map
        except Exception as e:
            print(f"Lỗi dịch trực tiếp qua Gemini API ({clean_model}): {e}")
        return {}

    @classmethod
    def translate(
        cls,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[str]:
        """Gửi yêu cầu dịch phụ đề tới 9router qua chuẩn OpenAI /chat/completions."""
        batch_res = cls.translate_batch([text], source_lang=source_lang, target_lang=target_lang, url=url, api_key=api_key, model=model)
        return batch_res[0] if batch_res else None

    @classmethod
    def _clean_markdown_json(cls, raw: str) -> str:
        """Loại bỏ các khối markdown ```json ... ``` và làm sạch chuỗi JSON."""
        if not raw:
            return ""
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    @classmethod
    def _extract_json_array(cls, raw_text: str) -> Optional[List[Any]]:
        """Trích xuất mảng JSON an toàn từ phản hồi của LLM."""
        if not raw_text:
            return None
        cleaned = cls._clean_markdown_json(raw_text)
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            json_str = match.group(0).strip()
            # Xóa dấu phẩy thừa trước ngoặc vuông đóng
            json_str = re.sub(r",\s*\]", "]", json_str)
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return None

    @classmethod
    def _extract_id_translations(cls, raw_text: str) -> Dict[str, str]:
        """Trích xuất ánh xạ id -> translation với khả năng cứu vãn cả khi JSON bị cắt cụt."""
        result: Dict[str, str] = {}
        if not raw_text:
            return result

        parsed_list = cls._extract_json_array(raw_text)
        if parsed_list:
            for obj in parsed_list:
                if isinstance(obj, dict) and "id" in obj:
                    s_id = str(obj.get("id"))
                    trans = str(obj.get("translation") or obj.get("refined") or obj.get("text") or "").strip()
                    if trans:
                        result[s_id] = trans
            if result:
                return result

        # Cứu vãn bằng Regex từng object nếu JSON array bị lỗi cú pháp / bị cắt cụt
        cleaned = cls._clean_markdown_json(raw_text)
        pattern = r'\{[^{}]*?"id"\s*:\s*"([^"]+)"[^{}]*?"(?:translation|refined|text)"\s*:\s*"([^"]+)"[^{}]*?\}'
        for m in re.finditer(pattern, cleaned):
            s_id = m.group(1).strip()
            trans = m.group(2).strip()
            if s_id and trans:
                result[s_id] = trans

        return result

    @classmethod
    def _translate_single_text(
        cls,
        text: str,
        client: httpx.Client,
        endpoint: str,
        key: str,
        model_name: str,
        target_lang: str = "vi"
    ) -> Optional[str]:
        """Dịch 1 câu đơn lẻ nhanh với client có sẵn."""
        if not text or not text.strip():
            return ""
        single_payload = {
            "model": model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Bạn là chuyên gia biên dịch phụ đề phim chuyên nghiệp. Dịch ngắn gọn, tự nhiên, chỉ trả về duy nhất câu dịch tiếng Việt."},
                {"role": "user", "content": f"Dịch câu thoại sau sang tiếng Việt:\n\n{text}"}
            ],
            "temperature": 0.2
        }
        try:
            resp = client.post(endpoint, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=single_payload)
            if resp.status_code == 200:
                content = cls.parse_response_text(resp.text).strip().strip("*_\"'").strip()
                if content:
                    return content
        except Exception:
            pass
        return None

    @classmethod
    def translate_batch(
        cls,
        texts: List[str],
        source_lang: str = "auto",
        target_lang: str = "vi",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> List[Optional[str]]:
        """Dịch theo cụm danh sách các câu phụ đề qua JSON array với cơ chế chia đôi thích ứng chống timeout."""
        if not texts:
            return []

        # Hỗ trợ Gemini Direct
        if model and ((model.startswith("gemini-") and "/" not in model) or model.startswith("gemini_direct")):
            items_dummy = [{"id": str(i), "text": t, "duration_sec": 2.0, "max_words": 10} for i, t in enumerate(texts)]
            direct_res = cls.translate_segments_direct_gemini(items_dummy, source_lang=source_lang, target_lang=target_lang, model=model)
            if direct_res:
                return [direct_res.get(str(i)) for i in range(len(texts))]

        base_url = (url or cls.DEFAULT_URL).rstrip("/")
        key = api_key or cls.find_local_key()
        model_name = model or cls.get_recommended_model()
        endpoint = f"{base_url}/chat/completions"

        system_prompt = (
            "Bạn là một dịch giả phụ đề và kịch bản phim ảnh chuyên nghiệp.\n"
            "Nhiệm vụ: Dịch kịch bản thoại/phụ đề từ ngôn ngữ nguồn sang tiếng Việt tự nhiên, phù hợp ngữ cảnh.\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. Đọc hiểu TOÀN BỘ MẠCH CÂU CHUYỆN. Dịch tự nhiên, đúng xưng hô nhân vật theo bối cảnh, bảo toàn nghĩa gốc.\n"
            "2. Câu dịch tiếng Việt gãy gọn, súc tích, khớp thời lượng phát âm của diễn viên.\n"
            "3. Tuyệt đối không bịa đặt thêm từ ngữ thô tục hoặc phóng đại nếu nguyên tác không có.\n"
            "4. BẮT BUỘC chỉ trả về DUY NHẤT một JSON array chứa các câu dịch tiếng Việt tương ứng, đúng số lượng phần tử: [\"câu 1\", \"câu 2\", ...]"
        )
        source_hint = (
            "Tự nhận diện ngôn ngữ nguồn cho từng câu."
            if source_lang.lower() in ("auto", "", "unknown")
            else f"Ngôn ngữ nguồn: {source_lang}."
        )
        prompt = (
            f"{source_hint} Dịch danh sách các câu phụ đề sau sang tiếng Việt ('{target_lang}') tự nhiên, trung thành và gãy gọn:\n"
            + json.dumps(texts, ensure_ascii=False)
        )
        payload = {
            "model": model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        # 1. Try the full batch with a longer request timeout.
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(endpoint, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload)
                if resp.status_code == 200:
                    raw = cls.parse_response_text(resp.text)
                    parsed = cls._extract_json_array(raw)
                    if parsed and len(parsed) == len(texts):
                        return [str(x).strip().strip("\"'") for x in parsed]
        except Exception as e:
            print(f"9router translate_batch (size={len(texts)}) chưa hoàn tất ({e}). Kích hoạt thích ứng...")

        # Thử fallback sang model nhanh khác nếu model hiện tại lỗi
        fallback_cand = cls.get_recommended_model()
        if fallback_cand != model_name:
            try:
                payload["model"] = fallback_cand
                with httpx.Client(timeout=35.0) as client:
                    resp = client.post(endpoint, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload)
                    if resp.status_code == 200:
                        raw = cls.parse_response_text(resp.text)
                        parsed = cls._extract_json_array(raw)
                        if parsed and len(parsed) == len(texts):
                            return [str(x).strip().strip("\"'") for x in parsed]
            except Exception:
                pass

        # 2. Adaptive Halving: Nếu lô lớn hơn 3 câu, tự động chia đôi đệ quy
        if len(texts) > 3:
            mid = len(texts) // 2
            left_res = cls.translate_batch(texts[:mid], source_lang=source_lang, target_lang=target_lang, url=url, api_key=api_key, model=model)
            right_res = cls.translate_batch(texts[mid:], source_lang=source_lang, target_lang=target_lang, url=url, api_key=api_key, model=model)
            return left_res + right_res

        # 3. Fallback câu đơn lẻ với timeout ngắn (8s), tái sử dụng chung 1 client
        results = []
        try:
            with httpx.Client(timeout=8.0) as client:
                for t in texts:
                    try:
                        trans = cls._translate_single_text(t, client, endpoint, key, model_name, target_lang)
                    except Exception:
                        trans = None
                    results.append(trans)
        except Exception:
            return [None] * len(texts)
        return results

    @classmethod
    def translate_segments_by_id(
        cls,
        items: List[Dict[str, Any]],
        source_lang: str = "auto",
        target_lang: str = "vi",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Dịch kịch bản theo ID phân đoạn cố định (Segment-ID Mapped Translation).
        Kháng timeout bằng cơ chế chia đôi thích ứng, đổi model dự phòng và bóc tách regex cứu vãn.
        items: list of dicts: [{"id": segment_id, "text": source_text, "character": character_id, "duration_sec": ..., "max_words": ...}]
        Returns: dict {segment_id: translated_text}
        """
        if not items:
            return {}

        # 1. Hỗ trợ dịch trực tiếp qua Gemini API nếu được chỉ định
        if model and ((model.startswith("gemini-") and "/" not in model) or model.startswith("gemini_direct")):
            direct_res = cls.translate_segments_direct_gemini(items, source_lang=source_lang, target_lang=target_lang, model=model)
            if direct_res:
                return direct_res

        base_url = (url or cls.DEFAULT_URL).rstrip("/")
        key = api_key or cls.find_local_key()
        model_name = model or cls.get_recommended_model()
        endpoint = f"{base_url}/chat/completions"

        system_prompt = (
            "Bạn là một đạo diễn dịch thuật và chuyển ngữ phụ đề phim ảnh chuyên nghiệp.\n"
            "Nhiệm vụ: Chuyển ngữ trung thành, tự nhiên và ngắn gọn; giữ lại sắc thái vốn có nhưng không thêm cảm xúc không có trong nguyên tác.\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. XƯNG HÔ VÀ QUAN HỆ NHÂN VẬT CHUẨN XÁC:\n"
            "   - Dựa vào 'speaker', 'gender', 'role' và bối cảnh đối thoại để xưng hô tự nhiên (anh - em, tôi - cậu, ba/mẹ - con, thầy - trò, chú - cháu...).\n"
            "   - Giữ tính nhất quán của đại từ nhân xưng xuyên suốt toàn bộ phân cảnh đối thoại.\n"
            "2. SẮC THÁI CẢM XÚC VÀ LỜI THOẠI TỰ NHIÊN:\n"
            "   - Diễn đạt trôi chảy như ngôn ngữ đời thực hoặc phim điện ảnh chất lượng cao.\n"
            "   - Dùng các dấu câu cảm xúc: '!', '?', '...', '?!' và trợ từ cảm thán tự nhiên của tiếng Việt ('hả?', 'sao?', 'kìa!', 'đâu!', 'này!'...) để tạo ngữ điệu sống động cho giọng đọc AI.\n"
            "   - TUYỆT ĐỐI KHÔNG tự ý chèn từ chửi thề, thô tục, hoặc phóng đại nếu câu thoại gốc không có.\n"
            "3. KHỐNG CHẾ ĐỘ DÀI VÀ KHẨU HÌNH (TIMELINE PACING):\n"
            "   - ƯU TIÊN SỐ 1: BẢO TOÀN NGHĨA VÀ ĐỦ Ý. Câu thoại phải tự nhiên, dễ hiểu, người xem nghe là hiểu ngay. TUYỆT ĐỐI KHÔNG cắt xén bừa bãi làm què cụt câu văn!\n"
            "   - Diễn đạt gãy gọn, tránh các hư từ rườm rà (như 'về việc', 'tiến hành', 'một cách'...). Hướng tới mục tiêu khoảng 'max_words' khi có thể, nhưng luôn ưu tiên tính rõ nghĩa của câu thoại.\n"
            "   - Vẫn bảo toàn trọn vẹn thông tin trọng tâm và cảm xúc của câu thoại.\n"
            "4. BẮT BUỘC giữ nguyên 'id' của từng câu và trả về đúng JSON list các object: [{\"id\": \"...\", \"translation\": \"...\"}]"
        )
        prompt = (
            f"{('Tự nhận diện ngôn ngữ nguồn cho từng câu. ' if source_lang.lower() in ('auto', '', 'unknown') else f'Ngôn ngữ nguồn: {source_lang}. ')}"
            f"Dịch sang tiếng Việt ('{target_lang}') tự nhiên, trung thành với nghĩa gốc và phù hợp giới hạn từ:\n"
            + json.dumps(items, ensure_ascii=False)
        )
        payload = {
            "model": model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        result_map: Dict[str, str] = {}
        # 2. Try the full batch with a longer request timeout.
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(endpoint, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload)
                if resp.status_code == 200:
                    raw = cls.parse_response_text(resp.text)
                    result_map = cls._extract_id_translations(raw)
        except Exception as e:
            print(f"9router translate_segments_by_id (size={len(items)}, model={model_name}) chưa hoàn tất ({e}). Kích hoạt thích ứng...")

        # 3. Nếu model chính lỗi / timeout, thử ngay model dự phòng nhanh của 9router
        missing_count = len([it for it in items if str(it.get("id")) not in result_map or not result_map[str(it.get("id"))].strip()])
        if missing_count > 0:
            rec_model = cls.get_recommended_gemini_model() if (model_name and "gemini" in model_name.lower()) else cls.get_recommended_model()
            if rec_model != model_name:
                try:
                    payload["model"] = rec_model
                    with httpx.Client(timeout=35.0) as client:
                        resp = client.post(endpoint, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload)
                        if resp.status_code == 200:
                            raw = cls.parse_response_text(resp.text)
                            fb_map = cls._extract_id_translations(raw)
                            result_map.update(fb_map)
                except Exception:
                    pass

        # 4. Nếu vẫn còn thiếu và có Gemini API Key, thử fallback trực tiếp qua Gemini API
        missing_items = [it for it in items if str(it.get("id")) not in result_map or not result_map[str(it.get("id"))].strip()]
        if missing_items and cls.find_direct_gemini_key():
            try:
                g_res = cls.translate_segments_direct_gemini(missing_items, source_lang=source_lang, target_lang=target_lang)
                if g_res:
                    result_map.update(g_res)
            except Exception:
                pass

        # 5. Xử lý các câu còn thiếu bằng Adaptive Halving
        missing_items = [it for it in items if str(it.get("id")) not in result_map or not result_map[str(it.get("id"))].strip()]
        if missing_items:
            active_sub_model = cls.get_recommended_gemini_model() if (model_name and "gemini" in model_name.lower()) else cls.get_recommended_model()
            if len(missing_items) > 3:
                mid = len(missing_items) // 2
                left_map = cls.translate_segments_by_id(missing_items[:mid], source_lang=source_lang, target_lang=target_lang, url=url, api_key=api_key, model=active_sub_model)
                right_map = cls.translate_segments_by_id(missing_items[mid:], source_lang=source_lang, target_lang=target_lang, url=url, api_key=api_key, model=active_sub_model)
                result_map.update(left_map)
                result_map.update(right_map)
            else:
                try:
                    with httpx.Client(timeout=8.0) as client:
                        for it in missing_items:
                            s_id = str(it.get("id"))
                            if s_id in result_map and result_map[s_id].strip():
                                continue
                            if single_t and single_t.strip():
                                result_map[s_id] = single_t.strip()
                except Exception:
                    pass

        # 6. Bảo toàn lời thoại gốc nếu tất cả API thất bại
        for it in items:
            s_id = str(it.get("id"))
            if s_id not in result_map or not result_map[s_id].strip():
                result_map[s_id] = str(it.get("text", "")).strip()

        return result_map
