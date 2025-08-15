"""This script prepares prompts from WaniKani subjects data.
"""

import json
from typing import Any, Dict, List

import yaml
from loguru import logger
from tqdm import tqdm

from src.utils.convert import wk_to_train_sample


def prepare_prompts(data_path: str) -> List[Dict[str, Any]]:
    """Prepare prompts from WaniKani subjects data.

    Args:
        data_path (str): Path to the JSON file containing WaniKani subjects data

    Returns:
        List[Dict[str, Any]]: List of prepared prompts
    """

    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # For faster search, we create a dictionary mapping subject IDs to subject data
    id2subject = dict()
    logger.info(f"Preprocessing {len(data)} subjects from {data_path}")
    for subject in tqdm(data):
        id2subject[subject["id"]] = subject
    prompts = []
    logger.info("Preparing prompts")
    # Main loop
    for subject in tqdm(id2subject.values()):
        # Process only kanji
        if subject["object"] == "kanji":
            processed_sample = wk_to_train_sample(
                wk_sample=subject,
                wk_idx2data=id2subject
            )
            prompts.append(processed_sample)
    logger.info(f"Prepared {len(prompts)} prompts")

    return prompts


if __name__ == "__main__":
    try:
        with open("params.yaml", "r") as file:
            params = yaml.safe_load(file)
        prompts = prepare_prompts(data_path=params["fetch"]["out_file"])
        with open(params["prepare_prompts"]["out_file"], "w", encoding="utf-8") as file:
            json.dump(prompts, file, ensure_ascii=False)
        logger.info(
            f"Prompts prepared and saved to {params['prepare_prompts']['out_file']}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e
