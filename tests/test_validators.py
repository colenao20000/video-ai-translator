"""Unit Tests for TimelineValidator and ProjectIntegrityValidator (Section 8)."""

import os
import pytest
from src.vtranslator.domain.project import Project
from src.vtranslator.domain.characters import Character
from src.vtranslator.domain.segments import SubtitleSegment
from src.vtranslator.domain.enums import CharacterMatchStatus
from src.vtranslator.domain.validators import TimelineValidator, ProjectIntegrityValidator


def test_timeline_validator_duplicate_id():
    seg1 = SubtitleSegment(segment_id="dup_id", start_ms=1000, end_ms=2000, original_text="A")
    seg2 = SubtitleSegment(segment_id="dup_id", start_ms=3000, end_ms=4000, original_text="B")
    res = TimelineValidator.validate_timeline_data([seg1, seg2])
    assert not res.is_valid
    assert any("Trùng lặp segment_id" in err.message for err in res.errors)


def test_timeline_validator_invalid_timestamps():
    seg = SubtitleSegment(segment_id="s1", start_ms=5000, end_ms=2000, original_text="A")
    res = TimelineValidator.validate_timeline_data([seg])
    assert not res.is_valid
    assert any("start_ms" in err.message for err in res.errors)


def test_project_integrity_validator_unregistered_character():
    char = Character(id="char_01", voice_id="Minh Đức")
    seg = SubtitleSegment(segment_id="s1", character_id="char_nonexistent", start_ms=1000, end_ms=2000)
    proj = Project(characters=[char], segments=[seg])

    res = ProjectIntegrityValidator.validate_project(proj)
    assert not res.is_valid
    assert any("không tồn tại" in err.message for err in res.errors)


def test_project_integrity_validator_confirmed_char_missing_voice():
    char = Character(id="char_01", voice_id="", match_status=CharacterMatchStatus.CONFIRMED)
    seg = SubtitleSegment(segment_id="s1", character_id="char_01", start_ms=1000, end_ms=2000)
    proj = Project(characters=[char], segments=[seg])

    res = ProjectIntegrityValidator.validate_project(proj)
    assert not res.is_valid
    assert any("chưa có Voice ID" in err.message for err in res.errors)
