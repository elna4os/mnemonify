"""
This module defines the KanjiSubject class and related types for handling kanji characters and their readings.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.subjects.ABCSubject import SubjectABC
from enum import StrEnum


__all__ = [
    "KanjiReadingType",
    "KanjiReading",
    "KanjiSubject",
]


class KanjiReadingType(StrEnum):
    """
    Enum representing the type of kanji reading.
    """

    KUNYOMI = "kunyomi"
    NANORI = "nanori"
    ONYOMI = "onyomi"


@dataclass
class KanjiReading:
    """
    Represents a reading of a kanji character.
    """

    reading: str
    primary: bool
    accepted_answer: bool
    type: KanjiReadingType


@dataclass
class KanjiSubject(SubjectABC):
    """
    Represents a kanji subject with its readings and meanings.
    """

    amalgamation_subject_ids: List[int]
    component_subject_ids: List[int]
    meaning_hint: Optional[str]
    reading_hint: Optional[str]
    reading_mnemonic: str
    readings: List[KanjiReading]
    visually_similar_subject_ids: List[int]
