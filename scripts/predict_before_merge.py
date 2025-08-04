"""Make predictions on a random subset of the dataset using a pretrained model (mainly, for debugging purposes, e.g. to compare predictions before and after weights merging and conversion to GGUF).
"""

import json
import random

import torch
import yaml
from loguru import logger

from src.utils.inference import (batch_predict, build_prompt,
                                 load_model_and_tokenizer)


def predict_before_merge(
    num_samples: int,
    model_dir: str,
    data_path: str,
    output_path: str,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int
) -> None:
    """Make predictions on a random subset of the dataset using a pretrained model.

    Args:
        num_samples (int): Num of samples to predict
        model_dir (str): Pretrained model directory
        data_path (str): Path to the dataset file
        output_path (str): Path to save the predictions
        seed (int): Random seed for reproducibility
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling probability
        max_new_tokens (int): Maximum number of new tokens to generate
        batch_size (int): Batch size for prediction
    """

    torch.manual_seed(seed)
    random.seed(seed)
    model, tokenizer = load_model_and_tokenizer(model_dir, use_bnb=True, device_map="auto")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data) > num_samples:
        indices = random.sample(range(len(data)), num_samples)
    else:
        indices = list(range(len(data)))
    prompts = []
    idx_map = []
    for idx in indices:
        sample = data[idx]
        prompt = build_prompt(sample)
        prompts.append(prompt)
        idx_map.append(idx)
    predictions = batch_predict(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size
    )
    results = []
    for i, idx in enumerate(idx_map):
        sample = data[idx]
        results.append({
            "index": idx,
            "prompt": prompts[i],
            "prediction": predictions[i],
            "output": sample["output"]
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        pbm = params["predict_before_merge"]
        predict_before_merge(
            num_samples=pbm["num_samples"],
            model_dir=pbm["model_dir"],
            data_path=pbm["data_path"],
            output_path=pbm["output_path"],
            seed=pbm["seed"],
            temperature=pbm["temperature"],
            top_p=pbm["top_p"],
            max_new_tokens=pbm["max_new_tokens"],
            batch_size=pbm["batch_size"]
        )
    except Exception as e:
        logger.error(f"Exception during prediction: {e}")
        raise e
