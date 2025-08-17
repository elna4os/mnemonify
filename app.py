"""This is the main application file for the Kanji Mnemonic Generator using Streamlit and llama.cpp.
"""

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from loguru import logger

from scripts.fetch_wanikani import fetch_subjects
from src.utils.convert import prepare_inference_sample
from src.utils.inference import build_prompt
from src.utils.text import is_kanji, read_krad_file


def ensure_model(
    model_repo: str,
    model_filename: str,
    local_dir: str
) -> str:
    """Ensure the model file exists locally by downloading it from the Hugging Face Hub if necessary.

    Args:
        model_repo (str): Hugging Face repository name
        model_filename (str): Filename of the model to download
        local_dir (str): Local directory to save the model file

    Returns:
        str: Path to the local model file
    """

    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, model_filename)
    if not os.path.exists(local_path):
        print(f"Downloading {model_filename} from {model_repo}...")
        hf_hub_download(
            repo_id=model_repo,
            filename=model_filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

    return local_path


@st.cache_resource
def get_krad_data(krad_data_path: str, krad2_data_path: str) -> Dict[str, List[str]]:
    """Read KRAD/KRAD2 files and return a combined dictionary of kanji radicals.

    Args:
        krad_data_path (str): Path to the KRAD data file
        krad2_data_path (str): Path to the KRAD2 data file

    Returns:
        Dict[str, List[str]]: Combined dictionary of kanji radicals
    """

    krad_1 = read_krad_file(path=krad_data_path)
    logger.info(f"Loaded {len(krad_1)} kanji from {krad_data_path}")
    krad_2 = read_krad_file(path=krad2_data_path)
    logger.info(f"Loader {len(krad_2)} kanji from {krad2_data_path}")

    return krad_1 | krad_2


@st.cache_resource
def get_radicals_data(data_path: str) -> Dict[str, List[str]]:
    """Get radicals with corresponding meanings from KanjiAlive file.

    Args:
        data_path (str): Path to the radicals data file

    Returns:
        Dict[str, List[str]]: Dictionary mapping radicals to their meanings
    """

    df = pd.read_csv(data_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} radical entries from {data_path}")
    logger.info("Drop repeating radicals")
    df.drop_duplicates(subset=["Radical"], inplace=True)
    logger.info(f"Remaining radicals: {len(df)}")
    radical2meanings = {str(row.Radical): str(row.Meaning).split(", ") for row in df.itertuples(index=False)}

    return radical2meanings


@st.cache_resource
def get_kanji_data(data_path: str) -> Dict[str, List[str]]:
    """Get kanji with corresponding meanings from KanjiAlive file.

    Args:
        data_path (str): Path to the kanji data file

    Returns:
        Dict[str, List[str]]: Dictionary mapping kanji to their meanings
    """

    df = pd.read_csv(data_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} kanji entries from {data_path}")
    logger.info("Drop repeating kanji")
    df.drop_duplicates(subset=["kanji"], inplace=True)
    logger.info(f"Remaining kanji: {len(df)}")
    kanji2meanings = {str(row.kanji): str(row.kmeaning).split(", ") for row in df.itertuples(index=False)}

    return kanji2meanings


@st.cache_resource
def get_wanikani_data(
    data_path: str,
    wanikani_api_url: Optional[str] = None,
    headers: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Fetch WaniKani subjects from the API or load from a local file.

    Args:
        data_path (str): Path to the WaniKani data file
        wanikani_api_url (Optional[str], optional): Base URL for the WaniKani API. Defaults to None.
        headers (Optional[Dict[str, Any]], optional): Headers for the API request. Defaults to None.

    Returns:
        List[Dict[str, Any]]: List of WaniKani subjects
    """

    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} WaniKani subjects from {data_path}")
    else:
        logger.info(f"Fetching WaniKani subjects from {wanikani_api_url}")
        data = fetch_subjects(base_url=wanikani_api_url, headers=headers)

    return data


@st.cache_resource
def load_model(
    model_path: str,
    n_ctx: int,
    n_threads: int,
    n_gpu_layers: int
) -> Any:
    """Load model with llama.cpp.

    Args:
        model_path (str): Model path
        n_ctx (int): llama.cpp context size
        n_threads (int): llama.cpp number of threads
        n_gpu_layers (int): llama.cpp number of GPU layers

    Returns:
        Any: Loaded model
    """

    return Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_threads=n_threads
    )


def generate_mnemonic(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float
) -> str:
    """Generate a mnemonic using the LLM.

    Args:
        prompt (str): Prompt to generate the mnemonic
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling probability
        repeat_penalty (float): Repeat penalty for the generation

    Returns:
        str: Generated mnemonic text
    """

    output = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        stop=["<|user|>:", "<|endoftext|>", "\n"])

    return output["choices"][0]["text"].strip()


if __name__ == "__main__":
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    load_dotenv()

    # Load KanjiAlive kanji and radicals meanings
    kanji2meanings = get_kanji_data(params["app"]["kanji_data_path"])
    radical2meanings = get_radicals_data(params["app"]["radicals_data_path"])

    # Load KRAD/KRAD2 data
    krad_data = get_krad_data(
        krad_data_path=params["app"]["krad_data_path"],
        krad2_data_path=params["app"]["krad2_data_path"]
    )

    # Load WaniKani subjects
    api_token = os.getenv("WANIKANI_API_TOKEN")
    headers = {"Authorization": f"Bearer {api_token}"}
    wk_subjects = get_wanikani_data(
        data_path=params["app"]["wanikani_data_path"],
        wanikani_api_url=params["fetch"]["base_url"],
        headers=headers
    )
    wk_char2subject = {s["data"]["characters"]: s for s in wk_subjects if s["object"] == "kanji"}
    wk_id2subject = {s["id"]: s for s in wk_subjects}

    # Download model
    model_path = ensure_model(
        model_repo=params["app"]["model_repo"],
        model_filename=params["app"]["model_filename"],
        local_dir=params["app"]["local_dir"]
    )
    st.title("Kanji Mnemonic Generator (Phi3 Mini 4K + llama.cpp + Streamlit)")
    kanji_input = st.text_input("Enter a single kanji character:")
    if st.button("Generate Mnemonic"):
        if not kanji_input:
            st.warning("Please enter a kanji character.")
        elif not is_kanji(kanji_input):
            st.error("Input is not a valid kanji character. Please enter a single kanji.")
        else:
            # Load the model
            llm = load_model(
                model_path=model_path,
                n_ctx=params["app"]["n_ctx"],
                n_threads=params["app"]["n_threads"],
                n_gpu_layers=params["app"]["n_gpu_layers"]
            )
            # Prepare sample
            sample_data = prepare_inference_sample(
                kanji=kanji_input,
                wk_char2subject=wk_char2subject,
                wk_id2subject=wk_id2subject,
                kanji2meanings=kanji2meanings,
                radical2meanings=radical2meanings,
                krad_data=krad_data
            )
            prompt = build_prompt(sample=sample_data)
            # Generate mnemonic
            with st.spinner("Generating mnemonic..."):
                mnemonic = generate_mnemonic(
                    prompt=prompt,
                    max_new_tokens=params["app"]["max_new_tokens"],
                    temperature=params["app"]["temperature"],
                    top_p=params["app"]["top_p"],
                    repeat_penalty=params["app"]["repeat_penalty"]
                )
                mnemonic = ". ".join(mnemonic.split(".")[:params["app"]["mnemonic_max_sentences"]])
            st.markdown(mnemonic)
