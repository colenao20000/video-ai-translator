"""Vision & Script-Based Character Recognition, Avatar Extraction & Dialogue Attribution Service."""

import os
import re
import json
import base64
from typing import List, Dict, Optional, Tuple
import cv2
import httpx
from ..domain.characters import Character, DEFAULT_PALETTE
from ..domain.enums import SpeakerGender, SpeakerAgeGroup, VoiceCharacter, CharacterMatchStatus
from ..domain.segments import SubtitleSegment
from ..domain.voices import VoiceLibrary
from ..ai.character_resolver import CharacterResolver
from ..services.nine_router_service import NineRouterService


class VisionCharacterService:
    @staticmethod
    def extract_scene_frames(video_path: str, timestamps: List[float]) -> List[Tuple[float, str]]:
        """Trích xuất ảnh khung hình tại các mốc thời gian chỉ định."""
        if not os.path.exists(video_path):
            return []

        out_dir = os.path.expanduser("~/.vtranslator/temp/frames")
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        extracted = []

        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(out_dir, f"frame_{int(ts)}s.jpg")
                cv2.imwrite(frame_path, frame)
                extracted.append((ts, frame_path))

        cap.release()
        return extracted

    @classmethod
    def analyze_characters_from_script(
        cls,
        segments: List[SubtitleSegment],
        model_name: str = "ag/gemini-3.8-flash-low",
        progress_callback=None
    ) -> List[Character]:
        """
        Đưa ra gợi ý nhân vật từ kịch bản; văn bản không xác định chắc chắn ai đang phát ngôn:
        - Đọc toàn bộ kịch bản theo thứ tự thời gian (không nhảy cóc để không bỏ sót lượt đối đáp).
        - Gợi ý tên, vai trò và đặc tính giọng dựa trên nội dung.
        - Gợi ý từ văn bản bổ sung hồ sơ; phân tách người nói từ âm thanh vẫn là căn cứ chính.
        """
        if not segments:
            return []

        if progress_callback:
            progress_callback(10, "Đang đọc kịch bản lời thoại để nhận diện các nhân vật...")

        # Giữ nguyên toàn bộ lời thoại tuần tự để bảo toàn cấu trúc hội thoại đối đáp
        valid_segments = [s for s in segments if (s.original_text or s.translated_text).strip()]
        # Lấy tối đa 350 câu liên tục đầu tiên (nơi toàn bộ nhân vật đều đã xuất hiện) mà KHÔNG nhảy cóc làm đứt mạch
        sample_subset = valid_segments[:350]

        lines_data = [
            {
                "id": s.segment_id,
                "time": f"{s.start_sec:.1f}s-{s.end_sec:.1f}s",
                "text": s.original_text.strip() or s.translated_text.strip()
            }
            for s in sample_subset
        ]

        prompt = (
            "Dưới đây là kịch bản lời thoại của video/phim theo trình tự thời gian:\n"
            + json.dumps(lines_data, ensure_ascii=False)
            + "\n\nNhiệm vụ: Gợi ý các nhân vật có thể xuất hiện trong kịch bản. Chỉ dựa vào văn bản thì không thể xác định chắc chắn người đang nói từng câu:\n"
            "QUY TẮC NHẬN DIỆN BẮT BUỘC:\n"
            "1. NHẬN DIỆN THEO ĐÚNG THỰC TẾ NỘI DUNG PHIM/VIDEO:\n"
            "   - Nhận diện đúng số lượng nhân vật dựa trên nội dung đối thoại (nếu video 1 người thuyết minh/vlogger thì chỉ tạo 1 nhân vật Thuyết minh; nếu có hội thoại nhiều người thì nhận diện đầy đủ từng nhân vật).\n"
            "   - Đặt tên nhân vật rõ ràng, chuẩn theo ngữ cảnh phim/video (dùng tên riêng nếu có trong kịch bản, hoặc vai trò: 'Nam chính', 'Nữ chính', 'Mẹ', 'Bác sĩ', 'Thầy giáo', 'Người dẫn chuyện'...). TUYỆT ĐỐI không bịa đặt những vai không có trong thoại.\n"
            "2. GỢI Ý GIỚI TÍNH ('male', 'female' hoặc 'unknown'):\n"
            "   - Chỉ dùng xưng hô và ngữ cảnh khi có căn cứ; nếu chưa rõ, trả về 'unknown'.\n"
            "3. GỢI Ý GIỌNG CLONE CÓ SẴN PHÙ HỢP NHẤT ('voice_preset'):\n"
            "   - Nhân vật Nam BẮT BUỘC chọn 1 trong: 'Gemini Puck' (ấm áp, tự nhiên), 'Gemini Charon' (trầm ấm, dõng dạc, uy nghiêm), 'Gemini Fenrir' (trẻ trung, năng động), 'Gemini Orus' (dày, uy lực, từng trải), 'Gemini Gacrux' (chững chạc, sâu lắng, lớn tuổi).\n"
            "   - Nhân vật Nữ BẮT BUỘC chọn 1 trong: 'Gemini Zephyr' (trong trẻo, tươi sáng, trẻ trung), 'Gemini Kore' (thanh lịch, trang nhã, điềm đạm), 'Gemini Aoede' (nhẹ nhàng, ngây thơ, trong sáng), 'Gemini Sulafat' (ấm áp, dịu dàng, truyền cảm), 'Gemini Leda' (tinh nghịch, đáng yêu).\n"
            "   - Ưu tiên gợi ý các giọng khác nhau khi danh mục có đủ lựa chọn; đây chỉ là gợi ý để người dùng rà soát.\n"
            "4. CHỈ DẪN DIỄN XUẤT ('acting_style'): Ghi ngắn gọn tâm lý, ngữ khí nhân vật theo kịch bản (ví dụ: 'tự tin, điềm đạm', 'lo lắng, van xin', 'nghiêm nghị, chỉ huy'...). Bám sát nội dung kịch bản.\n\n"
            "BẮT BUỘC chỉ trả về DUY NHẤT một JSON object theo định dạng:\n"
            "{\n"
            '  "characters": [\n'
            "    {\n"
            '      "id": "char_01",\n'
            '      "name": "Tên và vai trò nhân vật",\n'
            '      "gender": "male", "female" hoặc "unknown",\n'
            '      "age_group": "young_adult" hoặc "adult" hoặc "senior",\n'
            '      "voice_character": "deep" hoặc "bright" hoặc "soft" hoặc "energetic" hoặc "calm" hoặc "normal",\n'
            '      "voice_preset": "Gemini Puck" hoặc "Gemini Zephyr"...,\n'
            '      "acting_style": "chỉ dẫn diễn xuất theo ngữ cảnh phim",\n'
            '      "role": "main" hoặc "minor"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        raw_json = None
        is_9router_ok, _, _ = NineRouterService.check_connection()

        # 1. Prefer the configured local router when available.
        if is_9router_ok:
            key = NineRouterService.find_local_key()
            url = NineRouterService.DEFAULT_URL
            rec_model = NineRouterService.get_recommended_model()
            models_to_try = [model_name, rec_model, "ag/gemini-3.8-flash-low", "ag/gemini-3.8-flash", "cx/gpt-5.6-terra"]
            seen = set()
            clean_models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

            for m in clean_models:
                if m.startswith("gemini-") and "/" not in m:
                    continue
                payload = {
                    "model": m,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": "Bạn là chuyên gia phân tích kịch bản và đạo diễn lồng tiếng phim ảnh. Chỉ trả về đúng JSON object."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1
                }
                try:
                    with httpx.Client(timeout=45.0) as client:
                        resp = client.post(f"{url}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload)
                        if resp.status_code == 200:
                            text_resp = NineRouterService.parse_response_text(resp.text)
                            match = re.search(r"\{.*\}", text_resp, re.DOTALL)
                            if match:
                                raw_json = json.loads(match.group(0))
                                if raw_json.get("characters"):
                                    break
                except Exception as e:
                    print(f"Lỗi phân tích kịch bản ({m}): {e}")

        # 2. Dự phòng trực tiếp Google Gemini API nếu 9router chưa trả về kết quả
        gemini_key = NineRouterService.find_direct_gemini_key()
        if not raw_json and gemini_key:
            g_model = model_name if (model_name and model_name.startswith("gemini-") and "/" not in model_name) else "gemini-3.8-flash"
            try:
                with httpx.Client(timeout=30.0) as client:
                    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={gemini_key}"
                    resp = client.post(endpoint, json={"contents": [{"parts": [{"text": prompt}]}]})
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            text_resp = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []))
                            match = re.search(r"\{.*\}", text_resp, re.DOTALL)
                            if match:
                                raw_json = json.loads(match.group(0))
            except Exception as e:
                print(f"Lỗi phân tích kịch bản trực tiếp qua Gemini API ({g_model}): {e}")

        if not raw_json or not raw_json.get("characters"):
            return []

        result_characters: List[Character] = []
        for idx, item in enumerate(raw_json["characters"]):
            c_id = str(item.get("id") or f"char_{idx+1:02d}").strip()
            if not c_id or c_id == "char_unknown" or any(c.id == c_id for c in result_characters):
                c_id = f"char_{idx+1:02d}"
                suffix = 2
                while c_id == "char_unknown" or any(c.id == c_id for c in result_characters):
                    c_id = f"char_{idx+1:02d}_{suffix}"
                    suffix += 1
            name = item.get("name") or f"Nhân vật {idx+1}"
            g_str = str(item.get("gender", "unknown")).lower()
            gender = (
                SpeakerGender.FEMALE if "female" in g_str else
                SpeakerGender.MALE if "male" in g_str else
                SpeakerGender.UNKNOWN
            )
            age_str = item.get("age_group", "adult").lower()
            age_group = SpeakerAgeGroup.YOUNG_ADULT if "young" in age_str else (
                SpeakerAgeGroup.SENIOR if "senior" in age_str else SpeakerAgeGroup.ADULT
            )
            v_char_str = item.get("voice_character", "normal").lower()
            v_char = VoiceCharacter.DEEP if "deep" in v_char_str else (
                VoiceCharacter.BRIGHT if "bright" in v_char_str else (
                    VoiceCharacter.SOFT if "soft" in v_char_str else (
                        VoiceCharacter.ENERGETIC if "energetic" in v_char_str else (
                            VoiceCharacter.CALM if "calm" in v_char_str else VoiceCharacter.NORMAL
                        )
                    )
                )
            )

            # Chuẩn hóa voice_id sang canonical ID trong VoiceLibrary (có tiền tố Gemini)
            g_voice = str(item.get("voice_preset") or item.get("gemini_voice") or "").strip()
            canonical_voice = VoiceLibrary.resolve(g_voice)
            if canonical_voice is None:
                candidates = VoiceLibrary.filter_by_gender(gender) if gender != SpeakerGender.UNKNOWN else []
                canonical_voice = candidates[0] if candidates else VoiceLibrary.get_all()[0]
            if gender != SpeakerGender.UNKNOWN and canonical_voice.gender != gender:
                gender_voices = VoiceLibrary.filter_by_gender(gender)
                canonical_voice = gender_voices[0] if gender_voices else canonical_voice

            acting_style = item.get("acting_style") or item.get("style", "")
            color = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]

            c = Character(
                id=c_id,
                speaker_id=f"MODEL_{idx:02d}",
                name=name,
                gender=gender,
                age_group=age_group,
                voice_character=v_char,
                voice_id=canonical_voice.id,
                confidence=0.45,
                match_status=CharacterMatchStatus.PROBABLE,
                custom_prompt=acting_style,
                color=color
            )
            result_characters.append(c)

        # Suggest voices from the catalog; model-created speaker identities remain unconfirmed.
        result_characters = CharacterResolver.auto_assign_voices(result_characters)

        if progress_callback:
            progress_callback(30, f"Đã tạo {len(result_characters)} gợi ý hồ sơ nhân vật.")

        return result_characters


    @classmethod
    def analyze_characters_with_vision(
        cls,
        video_path: str,
        model_name: str = "ag/gemini-3.8-flash-low",
        sample_timestamps: Optional[List[float]] = None,
        progress_callback=None
    ) -> List[Character]:
        """Nhận diện nhân vật kết hợp thị giác và ảnh đại diện."""
        if progress_callback:
            progress_callback(10, "Đang trích xuất khung hình đối thoại từ video...")

        ts_list = sample_timestamps or [15.0, 40.0, 75.0]
        frames = cls.extract_scene_frames(video_path, ts_list)
        if not frames:
            raise RuntimeError("Không thể trích xuất khung hình từ video.")

        primary_frame_path = frames[1][1] if len(frames) > 1 else frames[0][1]

        with open(primary_frame_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        if progress_callback:
            progress_callback(30, f"Đang nhận diện nhân vật bằng Vision AI ({model_name})...")

        key = NineRouterService.find_local_key()
        url = NineRouterService.DEFAULT_URL

        system_prompt = (
            "Bạn là hệ thống AI phân tích nhân vật phim ảnh chuyên nghiệp qua thị giác.\n"
            "Hãy quan sát hình ảnh khung hình phim và trích xuất danh sách các nhân vật chính đang xuất hiện.\n"
            "QUY TẮC:\n"
            "1. Ước lượng các nhân vật nhìn thấy; không khẳng định danh tính từ một khung hình đơn lẻ.\n"
            "2. Gợi ý giới tính ('male', 'female' hoặc 'unknown') và độ tuổi nếu có căn cứ.\n"
            "3. Chỉ định vị trí tương đối trên ảnh ('left', 'right', 'center') để hệ thống cắt ảnh avatar.\n"
            "BẮT BUỘC chỉ trả về DUY NHẤT một JSON object theo cấu trúc:\n"
            "{\n"
            '  "characters": [\n'
            "    {\n"
            '      "id": "char_01",\n'
            '      "name": "Tên nhân vật",\n'
            '      "gender": "male", "female" hoặc "unknown",\n'
            '      "age_group": "young_adult" hoặc "adult" hoặc "senior",\n'
            '      "position": "left" hoặc "right" hoặc "center"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        models_to_try = [model_name, "ag/gemini-3.8-flash-low", "ag/gemini-3.8-flash"]
        raw_text = None
        for m in models_to_try:
            payload = {
                "model": m,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hãy quan sát ảnh và phân tích các nhân vật:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                        ]
                    }
                ],
                "temperature": 0.2
            }
            try:
                with httpx.Client(timeout=40.0) as client:
                    resp = client.post(f"{url}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload)
                    if resp.status_code == 200:
                        raw_text = NineRouterService.parse_response_text(resp.text)
                        if raw_text and "{" in raw_text:
                            break
            except Exception as e:
                print(f"Lỗi Vision AI ({m}): {e}")

        if not raw_text:
            raise RuntimeError("Không thể kết nối Vision AI.")

        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise RuntimeError("Vision AI không trả về JSON hợp lệ.")

        parsed_json = json.loads(match.group(0))
        raw_chars = parsed_json.get("characters", [])
        if not raw_chars:
            raise RuntimeError("Vision AI không tìm thấy nhân vật nào trong ảnh.")

        avatar_dir = os.path.expanduser("~/.vtranslator/avatars")
        os.makedirs(avatar_dir, exist_ok=True)
        img_cv = cv2.imread(primary_frame_path)
        h, w, _ = img_cv.shape

        result_characters: List[Character] = []
        for idx, item in enumerate(raw_chars):
            char_id = f"visual_char_{idx+1:02d}"
            name = item.get("name") or f"Nhân vật {idx+1}"
            gender_str = str(item.get("gender", "unknown")).lower()
            gender = (
                SpeakerGender.FEMALE if "female" in gender_str else
                SpeakerGender.MALE if "male" in gender_str else
                SpeakerGender.UNKNOWN
            )
            age_str = item.get("age_group", "adult").lower()
            age_group = SpeakerAgeGroup.YOUNG_ADULT if "young" in age_str else SpeakerAgeGroup.ADULT

            avatar_file = os.path.join(avatar_dir, f"visual_{idx+1}.jpg")
            pos = item.get("position", "left").lower()
            try:
                if len(raw_chars) == 2:
                    crop = img_cv[int(h * 0.08) : int(h * 0.85), int(w * 0.10) : int(w * 0.50)] if (idx == 0 or "left" in pos) else img_cv[int(h * 0.08) : int(h * 0.85), int(w * 0.50) : int(w * 0.90)]
                else:
                    crop = img_cv[int(h * 0.1) : int(h * 0.8), int(w * 0.25) : int(w * 0.75)]
                if crop is not None and crop.size > 0:
                    cv2.imwrite(avatar_file, crop)
            except Exception:
                pass

            color = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
            c = Character(
                id=char_id,
                speaker_id=f"VISUAL_{idx:02d}",
                name=name,
                gender=gender,
                age_group=age_group,
                voice_id=(
                    "Gemini Puck" if gender != SpeakerGender.FEMALE else "Gemini Kore"
                ),
                confidence=0.4,
                match_status=CharacterMatchStatus.PROBABLE,
                color=color,
                avatar_path=avatar_file if os.path.exists(avatar_file) else None
            )
            result_characters.append(c)

        result_characters = CharacterResolver.auto_assign_voices(result_characters)
        return result_characters

    @classmethod
    def assign_dialogue_smart(
        cls,
        segments: List[SubtitleSegment],
        characters: List[Character],
        model_name: str = "ag/gemini-3.8-flash-low",
        progress_callback=None
    ) -> Tuple[List[SubtitleSegment], List[Character]]:
        """
        Phân vai câu thoại hội thoại đối đáp thông minh (Context Dialogue Attribution):
        - Nếu số nhân vật hiện có chưa đủ hoặc chỉ có tên chung chung, tự động phát hiện đầy đủ nhân vật từ kịch bản.
        - Phân tích sâu toàn bộ kịch bản và ngữ cảnh từng phân cảnh: tách bạch rõ lời thoại của nhân vật chính và từng nhân vật phụ.
        - Dùng kết quả diarization làm nguồn nhận diện người nói khi có dữ liệu.
        - Các gợi ý từ văn bản cần được người dùng rà soát trước khi tạo TTS.
        """
        if not segments:
            return segments, characters

        has_diarized_speakers = any(
            s.speaker_source in ("diarization", "heuristic") for s in segments
        )

        # 1. Tự động trích xuất danh sách nhân vật chi tiết nếu danh sách hiện có chưa đủ hoặc tên chung chung
        has_generic_names = not characters or all(
            c.name.startswith("Nhân vật") or c.name.startswith("Character") or c.name.startswith("Speaker")
            for c in characters
        )
        has_visual_profiles = any(c.id.startswith("visual_char_") for c in characters)
        if not has_diarized_speakers and not has_visual_profiles and (
            not characters or len(characters) <= 2 or has_generic_names
        ):
            try:
                script_chars = cls.analyze_characters_from_script(
                    segments, model_name=model_name, progress_callback=progress_callback
                )
                if script_chars and len(script_chars) >= len(characters):
                    characters = script_chars
            except Exception as e:
                print(f"Lỗi trích xuất nhân vật từ kịch bản: {e}")

        if not characters:
            if has_diarized_speakers:
                unique_speakers = list(dict.fromkeys(
                    s.speaker_id for s in segments if s.speaker_source in ("diarization", "heuristic")
                ))
                characters = [
                    Character(
                        id=f"char_{idx + 1:02d}", speaker_id=speaker_id,
                        name=f"Người nói {idx + 1}", gender=SpeakerGender.UNKNOWN,
                        voice_id="Gemini Puck", confidence=0.0,
                        match_status=CharacterMatchStatus.PROBABLE,
                        color=DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
                    )
                    for idx, speaker_id in enumerate(unique_speakers)
                ]
            else:
                characters = [
                    Character(id="char_01", speaker_id="SPEAKER_00", name="Nhân vật 1", voice_id="Gemini Puck", gender=SpeakerGender.MALE)
                ]

        total = len(segments)
        char_info = [
            {"id": c.id, "speaker_id": c.speaker_id, "name": c.name, "gender": c.gender.value, "voice": c.voice_id}
            for c in characters
        ]
        char_map = {c.id: c for c in characters}
        char_by_speaker = {c.speaker_id: c for c in characters}
        char_options = ", ".join([f'"{c.id}" ({c.name} - {c.gender.value})' for c in characters])

        is_9router_ok, _, _ = NineRouterService.check_connection()
        gemini_key = NineRouterService.find_direct_gemini_key()
        key = NineRouterService.find_local_key()
        url = NineRouterService.DEFAULT_URL
        rec_model = NineRouterService.get_recommended_model()
        models_to_try = [model_name, rec_model, "ag/gemini-3.8-flash-low", "ag/gemini-3.8-flash", "cx/gpt-5.6-terra"]
        seen = set()
        clean_models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

        mapping: Dict[str, str] = {}
        style_mapping: Dict[str, str] = {}
        speaker_name_votes: Dict[str, Dict[str, int]] = {}
        segment_by_id = {s.segment_id: s for s in segments}

        def record_assignments(items) -> bool:
            if not isinstance(items, list):
                return False
            accepted_any = False
            for item in items:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                segment_id = str(item["id"])
                if segment_id not in segment_by_id:
                    continue
                accepted_any = True
                mapping[segment_id] = item.get("character_id")
                style = item.get("speech_style") or item.get("style")
                if style:
                    style_mapping[segment_id] = str(style)

                segment = segment_by_id[segment_id]
                proposed_name = str(item.get("character_name") or item.get("speaker_name") or "").strip()
                normalized_name = proposed_name.casefold()
                generic_names = (
                    "unknown", "unknown speaker", "char_unknown", "chưa xác định",
                    "chưa rõ", "không rõ", "người nói", "speaker"
                )
                if (
                    segment.speaker_source in ("diarization", "heuristic")
                    and proposed_name
                    and not any(normalized_name.startswith(generic) for generic in generic_names)
                ):
                    votes = speaker_name_votes.setdefault(segment.speaker_id, {})
                    votes[proposed_name] = votes.get(proposed_name, 0) + 1
            return accepted_any

        # Smaller chunks reduce the chance of request timeouts and output truncation.
        # A short context overlap can help preserve dialogue continuity.
        chunk_size = 25
        overlap = 4
        num_chunks = (total + chunk_size - 1) // chunk_size

        for c_idx in range(num_chunks):
            start_i = c_idx * chunk_size
            end_i = min(total, (c_idx + 1) * chunk_size)
            sub_slice = segments[start_i:end_i]

            # Ngữ cảnh các câu trước đó để nối liền mạch hội thoại
            context_prefix = []
            if start_i > 0:
                ctx_start = max(0, start_i - overlap)
                for prev_s in segments[ctx_start:start_i]:
                    prev_cid = mapping.get(prev_s.segment_id, prev_s.character_id)
                    prev_c = char_map.get(prev_cid)
                    prev_cname = prev_c.name if prev_c else "Chưa xác định"
                    context_prefix.append({
                        "id": prev_s.segment_id,
                        "speaker": prev_cname,
                        "text": prev_s.original_text or prev_s.translated_text
                    })

            lines_data = [
                {
                    "id": s.segment_id,
                    "speaker_id": s.speaker_id if s.speaker_source in ("diarization", "heuristic") else "unresolved",
                    "time": f"{s.start_sec:.1f}s-{s.end_sec:.1f}s",
                    "text": s.original_text.strip() or s.translated_text.strip()
                }
                for s in sub_slice
            ]

            prompt_chunk = (
                f"Bối cảnh: Danh sách tất cả nhân vật hợp lệ:\n"
                + json.dumps(char_info, ensure_ascii=False, indent=2)
                + (f"\n\nNgữ cảnh hội thoại ngay trước đó:\n{json.dumps(context_prefix, ensure_ascii=False)}" if context_prefix else "")
                + f"\n\nDanh sách các câu thoại cần phân vai ({start_i+1} đến {end_i}/{total}):\n"
                + json.dumps(lines_data, ensure_ascii=False)
                + "\n\nNhiệm vụ: Gợi ý character_id, character_name và speech_style cho từng câu thoại.\n"
                f"QUY TẮC PHÂN VAI: Dùng bằng chứng trong lời thoại và dữ liệu người nói đã có; nếu không chắc, chọn unknown.\n"
                f"1. character_id BẮT BUỘC là một trong các ID hợp lệ: {char_options}, hoặc 'char_unknown' khi chưa đủ căn cứ.\n"
                "2. Không suy luận rằng các câu liền nhau luôn do cùng một người nói hoặc luân phiên theo mẫu. Chỉ dùng ngữ cảnh khi có bằng chứng rõ; nếu không chắc, chọn 'char_unknown'.\n"
                "3. Việc một câu nhắc tên nhân vật không có nghĩa nhân vật đó đang nói. Đại từ và giới tính cũng không đủ căn cứ nếu thiếu ngữ cảnh.\n"
                "4. Nếu speaker_id đã được xác định từ âm thanh, character_id phải khớp hồ sơ có speaker_id đó; không đổi danh tính theo kịch bản.\n"
                "5. Với speaker_id đã có, thêm character_name khi có bằng chứng rõ từ lời thoại (tự giới thiệu hoặc ngữ cảnh chắc chắn); không đoán tên từ một câu chỉ nhắc đến tên người khác. Nếu chưa rõ, để rỗng; hệ thống vẫn tự gán preset giọng mặc định để tạo TTS.\n"
                "6. speech_style chỉ mô tả sắc thái thể hiện trong câu gốc; nếu không rõ, để chuỗi rỗng.\n"
                'BẮT BUỘC chỉ trả về DUY NHẤT một JSON array: [{"id": "...", "character_id": "...", "character_name": "...", "speech_style": "..."}]'
            )

            chunk_assigned = False

            # 1. Try the configured local router first when available.
            if is_9router_ok:
                for m in clean_models:
                    if m.startswith("gemini-") and "/" not in m:
                        continue
                    payload = {
                        "model": m,
                        "stream": False,
                        "messages": [
                            {"role": "system", "content": "Bạn là đạo diễn phân vai lồng tiếng phim. Chỉ trả về JSON array."},
                            {"role": "user", "content": prompt_chunk}
                        ],
                        "temperature": 0.1
                    }
                    try:
                        with httpx.Client(timeout=45.0) as client:
                            resp = client.post(f"{url}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload)
                            if resp.status_code == 200:
                                content_str = NineRouterService.parse_response_text(resp.text)
                                match = re.search(r"\[.*\]", content_str, re.DOTALL)
                                if match:
                                    parsed = json.loads(match.group(0))
                                    chunk_assigned = record_assignments(parsed)
                                    break
                    except Exception as e:
                        print(f"Lỗi phân vai chunk {c_idx+1}/{num_chunks} ({m}): {e}")

            # 2. Thử trực tiếp qua Gemini API nếu 9router chưa thành công và có key
            if not chunk_assigned and gemini_key:
                g_model = model_name if (model_name and model_name.startswith("gemini-") and "/" not in model_name) else "gemini-3.8-flash"
                try:
                    with httpx.Client(timeout=30.0) as client:
                        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={gemini_key}"
                        resp = client.post(endpoint, json={"contents": [{"parts": [{"text": prompt_chunk}]}]})
                        if resp.status_code == 200:
                            data = resp.json()
                            candidates = data.get("candidates", [])
                            if candidates:
                                text_resp = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []))
                                match = re.search(r"\[.*\]", text_resp, re.DOTALL)
                                if match:
                                    parsed = json.loads(match.group(0))
                                    chunk_assigned = record_assignments(parsed)
                except Exception as e:
                    print(f"Lỗi phân vai trực tiếp qua Gemini API: {e}")

            if progress_callback:
                p = int((end_i / total) * 100)
                progress_callback(p, f"Đang phân vai hội thoại: {end_i}/{total} câu...")

        # Audio diarization is the source of speaker identity. Script analysis
        # may add a tentative display name, but cannot remap a measured speaker.
        for speaker_id, votes in speaker_name_votes.items():
            target_c = char_by_speaker.get(speaker_id)
            if not target_c or target_c.match_status == CharacterMatchStatus.MANUAL:
                continue
            if target_c.name.casefold().startswith(("người nói", "speaker", "character")):
                target_c.name = max(votes.items(), key=lambda item: item[1])[0]

        for s in segments:
            if s.speaker_source == "manual":
                target_c = char_map.get(s.character_id)
                if target_c:
                    s.sync_voice_from_character(target_c.voice_id)
                else:
                    s.character_id = "char_unknown"
                    s.speaker_id = "SPEAKER_UNKNOWN"
                    s.speaker_source = "unassigned"
                    s.confidence = 0.0
                    s.voice_id = ""
            elif has_diarized_speakers:
                if s.speaker_source in ("diarization", "heuristic"):
                    target_c = char_by_speaker.get(s.speaker_id)
                    if target_c:
                        s.character_id = target_c.id
                        s.sync_voice_from_character(target_c.voice_id)
                        if s.speaker_source == "heuristic":
                            s.confidence = min(s.confidence, 0.55)
                    else:
                        s.character_id = "char_unknown"
                        s.voice_id = ""
                else:
                    target_c = None
                    s.character_id = "char_unknown"
                    s.speaker_id = "SPEAKER_UNKNOWN"
                    s.speaker_source = "unassigned"
                    s.confidence = 0.0
                    s.voice_id = ""
            else:
                assigned_id = mapping.get(s.segment_id)
                target_c = char_map.get(assigned_id)

                # Tự động nắn câu thoại đặc trưng nữ sang nhân vật nữ nếu có
                text_content = (s.original_text + " " + s.translated_text).lower()
                female_cues = ["đừng mà", "cứu tôi với", "đồ biến thái", "buông tôi ra", "tha cho tôi", "đừng động vào tôi", "anh yêu", "chàng ơi", "thiếp"]
                if any(cue in text_content for cue in female_cues):
                    females = [c for c in characters if c.gender == SpeakerGender.FEMALE]
                    if females:
                        target_c = females[0]

                if target_c:
                    s.character_id = target_c.id
                    s.speaker_id = target_c.speaker_id
                    s.speaker_source = "model_guess"
                    s.confidence = min(s.confidence, 0.55)
                    target_c.match_status = CharacterMatchStatus.PROBABLE
                    canonical_v = VoiceLibrary.get_by_id(target_c.voice_id)
                    target_c.voice_id = canonical_v.id
                    s.sync_voice_from_character(canonical_v.id)
                else:
                    s.character_id = "char_unknown"
                    s.speaker_source = "unassigned"
                    s.confidence = min(s.confidence, 0.25)
                    s.voice_id = ""
            if s.speaker_source != "manual":
                s.speech_style = style_mapping.get(s.segment_id) or None

        if progress_callback:
            progress_callback(100, f"Đã tự động phân vai cho {len(segments)} câu thoại.")

        return segments, characters
