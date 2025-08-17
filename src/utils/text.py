"""Utility functions for reading and processing text data.
"""

import re
from typing import Dict, List

KANJI_REGEX = r"^[\u4e00-\u9fff]$"


def remove_tags(text: str) -> str:
    """Remove HTML tags from a string.

    Args:
        text (str): String containing HTML tags

    Returns:
        str: String with HTML tags removed
    """

    return re.sub(r'<[^>]+>', '', text)


def is_kanji(char: str) -> bool:
    """Check if a character is a kanji character.

    Args:
        char (str): Character to check

    Returns:
        bool: True if the character is a kanji, False otherwise
    """

    return re.match(KANJI_REGEX, char) is not None


def read_krad_file(
    path: str,
    encoding: str = "euc-jp"
) -> Dict[str, List[str]]:
    """Get data from KRAD file.

    Args:
        path (str): Path to the KRAD file
        encoding (str, optional): Encoding of the file. Defaults to "euc-jp".

    Returns:
        Dict[str, List[str]]: Dictionary mapping kanji to a list of radicals
    """

    res = dict()
    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()
    for line in lines:
        if not line.startswith("#"):
            kanji, radicals = line.split(":")
            kanji = kanji.strip()
            radicals = radicals.strip().split(" ")
            res[kanji] = radicals

    return res
