import json
from typing import Any, Dict, List

import jaconv
import yaml
from loguru import logger
from tqdm import tqdm

from src.subjects.ABCSubject import SubjectType
from src.subjects.KanjiSubject import KanjiReadingType
from src.templates.input import INPUT_DEFAULT
from src.templates.instructions import INSTRUCTION_DEFAULT
from src.utils.text import remove_tags


def prepare_prompts(
    data_path: str,
    instruction_template: str,
    input_template: str
) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # For faster search, we create a dictionary mapping subject IDs to subject data
    id2subject = dict()
    logger.info(f"Preprocessing {len(data)} subjects from {data_path}")
    for subject in tqdm(data):
        id2subject[subject["id"]] = subject
    prompts = []
    logger.info("Preparing prompts")
    for subject in tqdm(id2subject.values()):
        if subject["object"] == SubjectType.KANJI:
            readings = []
            for x in subject["data"]["readings"]:
                if x["type"] == KanjiReadingType.ONYOMI:
                    reading = jaconv.hira2kata(x["reading"])
                else:
                    reading = x["reading"]
                readings.append(str((reading, x["type"])))
            input_data = {
                "characters": subject["data"]["characters"],
                "type": subject["object"],
                "primary_meanings": ", ".join([x["meaning"] for x in subject["data"]["meanings"]]),
                "auxiliary_meanings": ", ".join([x["meaning"] for x in subject["data"]["auxiliary_meanings"] if x["type"] != "blacklist"]),
                "readings": ", ".join(readings),
                "compound_subjects": ", ".join([
                    str((
                        id2subject[idx]["data"]["characters"],
                        id2subject[idx]["data"]["meanings"][0]["meaning"]
                    )) for idx in subject["data"]["component_subject_ids"]]),
                "similar_subjects": ", ".join([
                    str((
                        id2subject[idx]["data"]["characters"],
                        id2subject[idx]["data"]["meanings"][0]["meaning"],
                    )) for idx in subject["data"]["visually_similar_subject_ids"]])
            }
            curr_prompt = {
                "instruction": instruction_template,
                "input": input_template.format(**input_data),
                "output": remove_tags(subject["data"]["meaning_mnemonic"])
            }
            prompts.append(curr_prompt)
    logger.info(f"Prepared {len(prompts)} prompts")

    return prompts


if __name__ == "__main__":
    try:
        with open("params.yaml", "r") as file:
            params = yaml.safe_load(file)
        prompts = prepare_prompts(
            data_path=params["fetch"]["out_file"],
            instruction_template=INSTRUCTION_DEFAULT,
            input_template=INPUT_DEFAULT
        )
        with open(params["prepare_prompts"]["out_file"], "w", encoding="utf-8") as file:
            json.dump(prompts, file, ensure_ascii=False)
        logger.info(
            f"Prompts prepared and saved to {params['prepare_prompts']['out_file']}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e
