"""Voice domain models and the configured voice catalog."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from .enums import SpeakerGender, SpeakerAgeGroup, VoiceStyle, VoiceCharacter


class VoiceOption(BaseModel):
    id: str
    name: str
    gender: SpeakerGender
    region: str = "Bắc"  # "Bắc" | "Trung" | "Nam"
    style: VoiceStyle = VoiceStyle.TU_NHIEN
    age_group: SpeakerAgeGroup = SpeakerAgeGroup.ADULT
    voice_character: VoiceCharacter = VoiceCharacter.NORMAL
    description: str = ""
    pitch: str = "+0Hz"
    rate: str = "+0%"
    base_voice: str = "vi-VN-NamMinhNeural"
    gemini_voice: str = "Puck"
    gemini_style: str = "tự nhiên và rõ ràng"


# Gemini voice presets available to the configured VieNeu model.
CLONED_GEMINI_VOICES: List[VoiceOption] = [
    VoiceOption(
        id="Gemini Puck", name="Gemini Puck", gender=SpeakerGender.MALE, region="Bắc",
        style=VoiceStyle.TU_NHIEN, age_group=SpeakerAgeGroup.ADULT, voice_character=VoiceCharacter.NORMAL,
        description="Nam · VieNeu On-Device (Gemini Clone) · Ấm áp, tự nhiên, sinh động", pitch="+0Hz", rate="+0%", base_voice="vi-VN-NamMinhNeural",
        gemini_voice="Puck", gemini_style="ấm áp, tự nhiên và thân thiện"
    ),
    VoiceOption(
        id="Gemini Charon", name="Gemini Charon", gender=SpeakerGender.MALE, region="Bắc",
        style=VoiceStyle.TIN_TUC, age_group=SpeakerAgeGroup.ADULT, voice_character=VoiceCharacter.DEEP,
        description="Nam · VieNeu On-Device (Gemini Clone) · Trầm ấm, dõng dạc, uy lực", pitch="+0Hz", rate="+0%", base_voice="vi-VN-NamMinhNeural",
        gemini_voice="Charon", gemini_style="chuyên nghiệp, dõng dạc, tin tức"
    ),
    VoiceOption(
        id="Gemini Kore", name="Gemini Kore", gender=SpeakerGender.FEMALE, region="Bắc",
        style=VoiceStyle.TIN_TUC, age_group=SpeakerAgeGroup.ADULT, voice_character=VoiceCharacter.CALM,
        description="Nữ · VieNeu On-Device (Gemini Clone) · Thanh lịch, chuẩn mực, rõ ràng", pitch="+0Hz", rate="+0%", base_voice="vi-VN-HoaiMyNeural",
        gemini_voice="Kore", gemini_style="thanh lịch, trang nhã và rõ ràng"
    ),
    VoiceOption(
        id="Gemini Zephyr", name="Gemini Zephyr", gender=SpeakerGender.FEMALE, region="Bắc",
        style=VoiceStyle.TU_NHIEN, age_group=SpeakerAgeGroup.YOUNG_ADULT, voice_character=VoiceCharacter.BRIGHT,
        description="Nữ · VieNeu On-Device (Gemini Clone) · Trong trẻo, tươi sáng, linh hoạt", pitch="+0Hz", rate="+0%", base_voice="vi-VN-HoaiMyNeural",
        gemini_voice="Zephyr", gemini_style="trong trẻo, tươi sáng và tự nhiên"
    ),
    VoiceOption(
        id="Gemini Fenrir", name="Gemini Fenrir", gender=SpeakerGender.MALE, region="Bắc",
        style=VoiceStyle.TU_NHIEN, age_group=SpeakerAgeGroup.YOUNG_ADULT, voice_character=VoiceCharacter.ENERGETIC,
        description="Nam · VieNeu On-Device (Gemini Clone) · Trẻ trung, sôi nổi, năng động", pitch="+0Hz", rate="+0%", base_voice="vi-VN-NamMinhNeural",
        gemini_voice="Fenrir", gemini_style="sôi nổi, tràn đầy năng lượng"
    ),
    VoiceOption(
        id="Gemini Aoede", name="Gemini Aoede", gender=SpeakerGender.FEMALE, region="Bắc",
        style=VoiceStyle.TU_NHIEN, age_group=SpeakerAgeGroup.YOUNG_ADULT, voice_character=VoiceCharacter.SOFT,
        description="Nữ · VieNeu On-Device (Gemini Clone) · Nhẹ nhàng, tự nhiên, trong sáng", pitch="+0Hz", rate="+0%", base_voice="vi-VN-HoaiMyNeural",
        gemini_voice="Aoede", gemini_style="nhẹ nhàng, trong sáng"
    ),
    VoiceOption(
        id="Gemini Sulafat", name="Gemini Sulafat", gender=SpeakerGender.FEMALE, region="Nam",
        style=VoiceStyle.DOC_TRUYEN, age_group=SpeakerAgeGroup.ADULT, voice_character=VoiceCharacter.SOFT,
        description="Nữ · VieNeu On-Device (Gemini Clone) · Ấm áp, truyền cảm kể chuyện", pitch="+0Hz", rate="+0%", base_voice="vi-VN-HoaiMyNeural",
        gemini_voice="Sulafat", gemini_style="dịu dàng, truyền cảm kể chuyện"
    ),
    VoiceOption(
        id="Gemini Orus", name="Gemini Orus", gender=SpeakerGender.MALE, region="Nam",
        style=VoiceStyle.DOC_TRUYEN, age_group=SpeakerAgeGroup.ADULT, voice_character=VoiceCharacter.DEEP,
        description="Nam · VieNeu On-Device (Gemini Clone) · Dày, uy lực, điềm đạm", pitch="+0Hz", rate="+0%", base_voice="vi-VN-NamMinhNeural",
        gemini_voice="Orus", gemini_style="trầm ấm, sâu lắng kể chuyện"
    ),
    VoiceOption(
        id="Gemini Gacrux", name="Gemini Gacrux", gender=SpeakerGender.MALE, region="Bắc",
        style=VoiceStyle.DOC_TRUYEN, age_group=SpeakerAgeGroup.ADULT, voice_character=VoiceCharacter.DEEP,
        description="Nam · VieNeu On-Device (Gemini Clone) · Trưởng thành, già dặn, sư phụ", pitch="+0Hz", rate="+0%", base_voice="vi-VN-NamMinhNeural",
        gemini_voice="Gacrux", gemini_style="trưởng thành, sâu lắng"
    ),
    VoiceOption(
        id="Gemini Leda", name="Gemini Leda", gender=SpeakerGender.FEMALE, region="Bắc",
        style=VoiceStyle.TU_NHIEN, age_group=SpeakerAgeGroup.YOUNG_ADULT, voice_character=VoiceCharacter.BRIGHT,
        description="Nữ · VieNeu On-Device (Gemini Clone) · Trẻ trung, đáng yêu, tinh nghịch", pitch="+0Hz", rate="+0%", base_voice="vi-VN-HoaiMyNeural",
        gemini_voice="Leda", gemini_style="tinh nghịch, đáng yêu"
    ),
]

CLONED_GEMINI_VOICE_IDS = {v.id for v in CLONED_GEMINI_VOICES}

# The application catalog currently exposes these ten presets.
AVAILABLE_VOICES: List[VoiceOption] = CLONED_GEMINI_VOICES

# Bản đồ chuyển đổi toàn bộ tên giọng gốc cũ sang giọng VieNeu Gemini Clone tương ứng
LEGACY_VOICE_TO_CLONE: Dict[str, str] = {
    "minh đức": "Gemini Charon",
    "phạm tuyên": "Gemini Puck",
    "thái sơn": "Gemini Orus",
    "xuân vĩnh": "Gemini Puck",
    "thanh bình": "Gemini Charon",
    "trúc ly": "Gemini Zephyr",
    "ngọc linh": "Gemini Sulafat",
    "đoan trang": "Gemini Kore",
    "mai anh": "Gemini Zephyr",
    "thục đoan": "Gemini Sulafat",
    "minh triết": "Gemini Puck",
    "thùy dung": "Gemini Kore",
    "quang sơn": "Gemini Puck",
    "ngọc trân": "Gemini Kore",
    "mỹ duyên": "Gemini Sulafat",
    "quỳnh anh": "Gemini Aoede",
    "đức trí": "Gemini Gacrux",
    "kim thanh": "Gemini Sulafat",
    "ngọc huyền": "Gemini Zephyr",
    "adam": "Gemini Fenrir",
    "mạnh dũng": "Gemini Orus",
    "minh quân": "Gemini Fenrir",
    "anh khôi": "Gemini Gacrux",
    "hải đăng": "Gemini Fenrir",
    "thiện minh": "Gemini Charon",
    "quốc tuấn": "Gemini Orus",
    "thiền tâm đức": "Gemini Gacrux",
    "adam bựa": "Gemini Fenrir",
}

VOICE_MAP: Dict[str, VoiceOption] = {v.id: v for v in AVAILABLE_VOICES}
VOICE_MAP.update({v.name.lower(): v for v in AVAILABLE_VOICES})
for v in AVAILABLE_VOICES:
    if hasattr(v, 'gemini_voice') and v.gemini_voice:
        g_name = v.gemini_voice.lower()
        VOICE_MAP[g_name] = v
        VOICE_MAP[f"{g_name} (gemini)"] = v
        VOICE_MAP[f"gemini {g_name}"] = v
        VOICE_MAP[f"gemini_{g_name}"] = v

# Ánh xạ cả tên giọng cũ sang VoiceOption Clone mới
for leg_name, target_clone_id in LEGACY_VOICE_TO_CLONE.items():
    if target_clone_id in VOICE_MAP:
        VOICE_MAP[leg_name] = VOICE_MAP[target_clone_id]


class VoiceLibrary:
    """Quản lý tra cứu và phân loại thư viện giọng nói (Chỉ gồm 10 giọng Gemini Clone)."""

    @staticmethod
    def get_all() -> List[VoiceOption]:
        return AVAILABLE_VOICES

    @staticmethod
    def resolve(voice_id: str) -> Optional[VoiceOption]:
        """Resolve a known current or legacy voice name without silently choosing a default."""
        if not voice_id:
            return None
        if voice_id in VOICE_MAP:
            return VOICE_MAP[voice_id]
        clean_id = voice_id.strip().lower()
        if clean_id in VOICE_MAP:
            return VOICE_MAP[clean_id]
        target_id = LEGACY_VOICE_TO_CLONE.get(clean_id)
        return VOICE_MAP.get(target_id) if target_id else None

    @staticmethod
    def get_by_id(voice_id: str, default: Optional[VoiceOption] = None) -> VoiceOption:
        return VoiceLibrary.resolve(voice_id) or default or AVAILABLE_VOICES[0]

    @staticmethod
    def filter_by_gender(gender: SpeakerGender) -> List[VoiceOption]:
        return [v for v in AVAILABLE_VOICES if v.gender == gender]

    @staticmethod
    def filter_by_region(region: str) -> List[VoiceOption]:
        return [v for v in AVAILABLE_VOICES if v.region == region]
