"""This module contains functions to convert data to training/inference samples.
"""

from typing import Any, Dict

from src.templates.input import INPUT_DEFAULT
from src.templates.instructions import INSTRUCTION_DEFAULT
from src.utils.text import remove_tags

__all__ = ["wk_to_train_sample"]


def wk_to_train_sample(
    wk_sample: Dict[str, Any],
    wk_idx2data: Dict[int, Dict[str, Any]]
) -> Dict[str, str]:
    """Convert WaniKani subject data to a training sample format.

    Args:
        wk_sample (Dict[str, Any]): WaniKani subject data
        wk_idx2data (Dict[int, Dict[str, Any]]): Dictionary mapping subject IDs to subject data

    Returns:
        Dict[str, str]: Training sample in the format required by the model
    """

    k2v = dict()
    k2v["characters"] = wk_sample["data"]["characters"]
    k2v["type"] = wk_sample["object"]
    # Get meanings (including auxiliary)
    k2v["meanings"] = []
    for meaning in wk_sample["data"]["meanings"]:
        k2v["meanings"].append(meaning["meaning"])
    for meaning in wk_sample["data"]["auxiliary_meanings"]:
        if meaning["type"] != "blacklist":
            k2v["meanings"].append(meaning["meaning"])
    k2v["meanings"] = ", ".join(k2v["meanings"])
    # Get compounding subjects
    k2v["parts"] = []
    for part_id in wk_sample["data"]["component_subject_ids"]:
        k2v["parts"].append(f'{wk_idx2data[part_id]["data"]["characters"]}/{wk_idx2data[part_id]["data"]["meanings"][0]["meaning"]}')
    k2v["parts"] = ", ".join(k2v["parts"])

    res = {
        "instruction": INSTRUCTION_DEFAULT,
        "input": INPUT_DEFAULT.format(**k2v),
        "output": remove_tags(wk_sample["data"]["meaning_mnemonic"]),
    }

    return res
