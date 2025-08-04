"""This script merges and unloads a PEFT (Parameter-Efficient Fine-Tuning) model into a base model and saves it to the specified output directory.
"""

import os
from shutil import rmtree

import yaml
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_unload(
    base_model_name: str,
    model_dir: str,
    output_dir: str
) -> None:
    """Merge and unload a PEFT model into a base model and save it to the output directory.

    Args:
        base_model_name (str): Base model name or path
        model_dir (str): Path to the PEFT model directory
        output_dir (str): Directory to save the merged model
    """

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto")
    model = PeftModel.from_pretrained(base_model, model_dir)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        merge_unload(
            base_model_name=params["training"]["model_name"],
            model_dir=params["merge_unload"]["model_dir"],
            output_dir=params["merge_unload"]["out_dir"]
        )
    except Exception as e:
        if os.path.exists(params["merge_unload"]["out_dir"]):
            rmtree(params["merge_unload"]["out_dir"])
        logger.error(f"Error during merge and unload: {e}")
        raise e
