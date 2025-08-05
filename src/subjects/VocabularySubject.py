"""
This module defines the VocabularySubject class and related data structures.
"""

from dataclasses import dataclass
from typing import List

from src.subjects.ABCSubject import SubjectABC


__all__ = [
    "ContextSentence",
    "VocabularyReading",
    "VocabularySubject",
]


@dataclass
class ContextSentence:
    """
    Represents a context sentence for a vocabulary subject.
    """

    en: str
    ja: str


@dataclass
class VocabularyReading:
    """
    Represents a reading for a vocabulary subject.
    """

    accepted_answer: bool
    primary: bool
    reading: str


@dataclass
class VocabularySubject(SubjectABC):
    """
    Represents a vocabulary subject with its properties and context.
    """

    component_subject_ids: List[int]
    context_sentences: List[ContextSentence]
    parts_of_speech: List[str]
    readings: List[VocabularyReading]
    reading_mnemonic: str
