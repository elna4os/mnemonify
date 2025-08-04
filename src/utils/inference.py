"""This module provides utilities for loading a pretrained model and tokenizer, building prompts, and making predictions using the model.
"""

from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

__all__ = [
    "load_model_and_tokenizer",
    "build_prompt",
    "batch_predict"
]


def load_model_and_tokenizer(
    model_dir: str,
    use_bnb: bool = True,
    device_map: str = "auto"
) -> Tuple[Any, Any]:
    """Load the model and tokenizer from the specified directory.

    Args:
        model_dir (str): Path to the model directory
        use_bnb (bool, optional): Whether to use BitsAndBytes for quantization. Defaults to True.
        device_map (str, optional): Device map for model loading. Defaults to "auto".

    Returns:
        Tuple[Any, Any]: Loaded model and tokenizer
    """

    if use_bnb:
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map, quantization_config=bnb_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    return model, tokenizer


def build_prompt(sample: Dict[str, str]) -> str:
    """Build a prompt from the sample data.

    Args:
        sample (Dict[str, str]): Sample data containing instruction and input

    Returns:
        str: Formatted prompt string ready for model input
    """

    messages = [
        {"role": "user", "content": sample["instruction"] + "\n" + sample["input"]}
    ]
    prompt = ""
    for msg in messages:
        prompt += f"<|{msg['role']}|>: {msg['content']}\n"
    prompt += "<|assistant|>:"

    return prompt


def batch_predict(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int = 4
) -> List[str]:
    """Make predictions in batches using the model and tokenizer.

    Args:
        model (Any): Model to use for predictions
        tokenizer (Any): Tokenizer for encoding prompts
        prompts (List[str]): List of prompts to predict
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling probability
        max_new_tokens (int): Maximum number of new tokens to generate
        batch_size (int, optional): Batch size for prediction. Defaults to 4.

    Returns:
        List[str]: List of predictions corresponding to the prompts
    """

    results = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(prompts), batch_size), total=num_batches, desc="Predicting"):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )
        for output in outputs:
            pred = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            results.append(pred.strip())

    return results
