"""
This module defines the base class for subjects.
"""

from abc import ABC
from dataclasses import dataclass
from enum import StrEnum
from typing import List, Optional

__all__ = [
    "SubjectType",
    "AuxiliaryMeaningType",
    "Meaning",
    "AuxiliaryMeaning",
    "SubjectABC",
]


class SubjectType(StrEnum):
    """
    Enum representing the type of subject object.
    """

    KANA_VOCABULARY = "kana_vocabulary"
    KANJI = "kanji"
    RADICAL = "radical"
    VOCABULARY = "vocabulary"


class AuxiliaryMeaningType(StrEnum):
    """
    Enum representing the type of auxiliary meaning.
    """

    WHITELIST = "whitelist"
    BLACKLIST = "blacklist"


@dataclass
class Meaning:
    """
    Represents a meaning of a subject.
    """

    meaning: str
    primary: bool
    accepted_answer: bool


@dataclass
class AuxiliaryMeaning:
    """
    Represents an auxiliary meaning of a subject.
    """

    text: str
    type: AuxiliaryMeaningType


@dataclass
class SubjectABC(ABC):
    """
    Base class for all subjects.
    """

    id: int
    object: SubjectType
    characters: Optional[str]
    meanings: List[Meaning]
    auxiliary_meanings: List[AuxiliaryMeaning]
    meaning_mnemonic: str
