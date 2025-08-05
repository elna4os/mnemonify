"""
This module defines the RadicalSubject class, which represents a radical subject in the mnemonify application.
"""

from dataclasses import dataclass
from typing import List

from src.subjects.ABCSubject import SubjectABC

__all__ = [
    "RadicalSubject",
]


@dataclass
class RadicalSubject(SubjectABC):
    """
    Represents a radical subject.
    """

    amalgamation_subject_ids: List[int]
