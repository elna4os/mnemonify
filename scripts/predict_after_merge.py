"""This script is used to make predictions using a converted and quantized model (for debugging purposes).
"""

import json

import yaml
from llama_cpp import Llama
from loguru import logger
from tqdm import tqdm
from transformers import set_seed


def predict_after_merge(
    data_path: str,
    model_path: str,
    out_file: str,
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int
) -> None:
    """Make predictions (on debugging purpose) using a converted and quantized model.

    Args:
        data_path (str): Path to the dataset file
        model_path (str): Model path for the converted and quantized model
        out_file (str): Path to save the predictions
        n_ctx (int): llama.cpp context size
        n_gpu_layers (int): llama.cpp number of GPU layers
        n_threads (int): llama.cpp number of threads
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling probability
        max_new_tokens (int): Maximum number of new tokens to generate
        seed (int): Random seed for reproducibility
    """

    set_seed(seed)
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads
    )
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for sample in tqdm(data):
        prediction = llm(
            prompt=sample["prompt"],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop="\n"
        )['choices'][0]['text']
        sample['prediction_after'] = prediction.strip()
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == "__main__":
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        predict_after_merge(
            data_path=params["predict_before_merge"]["output_path"],
            model_path=params["predict_after_merge"]["model_path"],
            out_file=params["predict_after_merge"]["output_path"],
            n_ctx=params["predict_after_merge"]["n_ctx"],
            n_gpu_layers=params["predict_after_merge"]["n_gpu_layers"],
            n_threads=params["predict_after_merge"]["n_threads"],
            temperature=params["predict_before_merge"]["temperature"],
            top_p=params["predict_before_merge"]["top_p"],
            max_new_tokens=params["predict_before_merge"]["max_new_tokens"],
            seed=params["predict_before_merge"]["seed"]
        )
    except Exception as e:
        logger.error(f"Exception during prediction: {e}")
        raise e
