"""Utility functions for processing text data.
"""

import re


def remove_tags(text: str) -> str:
    """Remove HTML tags from a string.

    Args:
        text (str): String containing HTML tags

    Returns:
        str: String with HTML tags removed
    """

    return re.sub(r'<[^>]+>', '', text)
