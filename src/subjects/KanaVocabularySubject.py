"""
This module defines the KanaVocabularySubject class, which is used to represent a kana vocabulary
"""

from dataclasses import dataclass
from typing import List

from src.subjects.ABCSubject import SubjectABC
from src.subjects.VocabularySubject import ContextSentence

__all__ = [
    "KanaVocabularySubject",
]


@dataclass
class KanaVocabularySubject(SubjectABC):
    """Represents a vocabulary subject for Kana characters
    """

    context_sentences: List[ContextSentence]
    parts_of_speech: List[str]
