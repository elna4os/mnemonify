"""Fetch WaniKani subjects and save them to a JSON file.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml
from dotenv import load_dotenv
from loguru import logger


def fetch_subjects(
    base_url: str,
    headers: Dict[str, Any],
    sleep_time: int = 3
) -> List[Dict[str, Any]]:
    """Fetch subjects from WaniKani API.

    Args:
        base_url (str): Base URL for the WaniKani API endpoint to fetch subjects.
        headers (Dict[str, Any]): Headers for the API request, including authorization.
        sleep_time (int, optional): Sleep time between requests to avoid hitting rate limits. Defaults to 3.

    Returns:
        List[Dict[str, Any]]: List of subjects fetched from the API.
    """

    subjects_list = []
    url = base_url
    while url:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Fetched {len(data['data'])} subjects from {url}")
        subjects_list.extend(data["data"])
        url = data["pages"]["next_url"]
        if url:
            time.sleep(sleep_time)
    logger.info(f"Total subjects fetched: {len(subjects_list)}")

    return subjects_list


if __name__ == "__main__":
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        load_dotenv()
        api_token = os.getenv("WANIKANI_API_TOKEN")
        headers = {"Authorization": f"Bearer {api_token}"}
        subjects_list = fetch_subjects(
            base_url=params["fetch"]["base_url"],
            headers=headers
        )
        Path(params["fetch"]["out_file"]).parent.mkdir(
            parents=True, exist_ok=True)
        with open(params["fetch"]["out_file"], "w", encoding="utf-8") as f:
            json.dump(subjects_list, f, ensure_ascii=False)
        logger.info(f"Subjects saved to {params['fetch']['out_file']}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e
