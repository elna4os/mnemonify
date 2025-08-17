"""This module contains functions to convert data
"""

import unicodedata
from typing import Any, Dict, List

from loguru import logger

from src.templates.input import INPUT_DEFAULT
from src.templates.instructions import INSTRUCTION_DEFAULT
from src.utils.text import remove_tags

__all__ = [
    "wk_to_sample",
    "prepare_inference_sample"
]


def wk_to_sample(
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
    k2v["kanji"] = wk_sample["data"]["characters"]
    # Get meanings (including auxiliary)
    k2v["primary_meaning"] = wk_sample["data"]["meanings"][0]["meaning"]
    k2v["other_meanings"] = []
    for meaning in wk_sample["data"]["meanings"][1:]:
        k2v["other_meanings"].append(meaning["meaning"])
    for meaning in wk_sample["data"]["auxiliary_meanings"]:
        if meaning["type"] != "blacklist":
            k2v["other_meanings"].append(meaning["meaning"])
    k2v["other_meanings"] = ", ".join(k2v["other_meanings"])
    # Get compounding subjects
    k2v["radicals"] = []
    for part_id in wk_sample["data"]["component_subject_ids"]:
        k2v["radicals"].append(str((wk_idx2data[part_id]["data"]["characters"], wk_idx2data[part_id]["data"]["meanings"][0]["meaning"])))
    k2v["radicals"] = ", ".join(k2v["radicals"])

    res = {
        "instruction": INSTRUCTION_DEFAULT,
        "input": INPUT_DEFAULT.format(**k2v),
        "output": remove_tags(wk_sample["data"]["meaning_mnemonic"]),
    }

    return res


def prepare_with_krad(
    kanji: str,
    kanji2meanings: Dict[str, List[str]],
    radical2meanings: Dict[str, List[str]],
    krad_data: Dict[str, List[str]],
    wk_char2subject: Dict[str, Dict[str, Any]],
    normalization_form: str = "NFC"
) -> Dict[str, str]:
    """Prepare a sample for inference using KRAD decomposition.

    Args:
        kanji (str): Kanji character to prepare the sample for
        kanji2meanings (Dict[str, List[str]]): KanjiAlive mapping of kanji to their meanings
        radical2meanings (Dict[str, List[str]]): KanjiAlive mapping of radicals to their meanings
        krad_data (Dict[str, List[str]]): KRAD data mapping kanji to their radicals
        wk_char2subject (Dict[str, Dict[str, Any]]): Mapping of WaniKani kanji to their data
        normalization_form (str, optional): Normalization form for radicals. Defaults to "NFC".

    Returns:
        Dict[str, str]: Prepared sample data for inference
    """

    k2v = dict()
    # Fill character
    k2v["kanji"] = kanji
    # Fill primary/other meanings
    k2v["other_meanings"] = []
    if kanji in wk_char2subject:
        k2v["primary_meaning"] = wk_char2subject[kanji]["data"]["meanings"][0]["meaning"]
        k2v["other_meanings"].extend([meaning["meaning"] for meaning in wk_char2subject[kanji]["data"]["meanings"][1:]])
        k2v["other_meanings"].extend([meaning["meaning"] for meaning in wk_char2subject[kanji]["data"]["auxiliary_meanings"] if meaning["type"] != "blacklist"])
    elif kanji in kanji2meanings:
        k2v["primary_meaning"] = kanji2meanings[kanji][0]
        k2v["other_meanings"].extend(kanji2meanings[kanji][1:])
    else:
        k2v["primary_meaning"] = ""
    k2v["other_meanings"] = ", ".join(k2v["other_meanings"])
    # Fill radicals
    k2v["radicals"] = []
    if kanji in krad_data:
        radicals = [unicodedata.normalize(normalization_form, x.strip()) for x in krad_data[kanji]]
        for radical in radicals:
            if radical in wk_char2subject:
                k2v["radicals"].append(str((radical, wk_char2subject[radical]["data"]["meanings"][0]["meaning"])))
            elif radical in radical2meanings:
                k2v["radicals"].append(str((radical, radical2meanings[radical][0])))
            elif radical in kanji2meanings:
                k2v["radicals"].append(str((radical, kanji2meanings[radical][0])))
            else:
                k2v["radicals"].append(str((radical, "")))
    k2v["radicals"] = ", ".join(k2v["radicals"])

    return {
        "instruction": INSTRUCTION_DEFAULT,
        "input": INPUT_DEFAULT.format(**k2v)
    }


def prepare_inference_sample(
    kanji: str,
    wk_char2subject: Dict[str, Dict[str, Any]],
    wk_id2subject: Dict[int, Dict[str, Any]],
    kanji2meanings: Dict[str, List[str]],
    radical2meanings: Dict[str, List[str]],
    krad_data: Dict[str, List[str]]
) -> Dict[str, str]:
    """Prepare a sample for inference based on the kanji character.

    Args:
        kanji (str): Kanji character to prepare the sample for
        wk_char2subject (Dict[str, Dict[str, Any]]): Mapping of WaniKani kanji to their data
        wk_id2subject (Dict[int, Dict[str, Any]]): Mapping of WaniKani subject IDs to their data
        kanji2meanings (Dict[str, List[str]]): KanjiAlive mapping of kanji to their meanings
        radical2meanings (Dict[str, List[str]]): KanjiAlive mapping of radicals to their meanings
        krad_data (Dict[str, List[str]]): KRAD data mapping kanji to their radicals

    Returns:
        Dict[str, str]: Prepared sample data for inference
    """

    # Look for kanji in WaniKani subjects
    if kanji in wk_char2subject:
        logger.info(f"Found kanji {kanji} in WaniKani subjects.")
        subject_data = wk_char2subject[kanji]
        sample_data = wk_to_sample(wk_sample=subject_data, wk_idx2data=wk_id2subject)
        del sample_data["output"]
    # If not found, use KRAD to decompose kanji and watch for radicals meanings in KanjiAlive or WaniKani
    else:
        logger.info(f"Kanji {kanji} not found in WaniKani subjects, using fallback method.")
        sample_data = prepare_with_krad(
            kanji=kanji,
            kanji2meanings=kanji2meanings,
            radical2meanings=radical2meanings,
            krad_data=krad_data,
            wk_char2subject=wk_char2subject
        )

    return sample_data
