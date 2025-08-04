"""Train Phi-3 Mini 4K model using QLoRA.
"""

import json
import os
import shutil
from typing import Any, Dict

import torch
import yaml
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


def preprocess(x: Dict[str, str]) -> Dict[str, str]:
    """Convert a single sample from the dataset into conversational prompt-completion format.

    Args:
        x (Dict[str, str]): A single sample from the dataset

    Returns:
        Dict[str, str]: Preprocessed sample
    """

    return {
        "messages": [
            {"role": "user", "content": x["instruction"] + "\n" + x["input"]},
            {"role": "assistant", "content": x["output"]}
        ]
    }


def train_phi3_mini_4k(
    data_path: str,
    output_dir: str,
    model_name: str,
    max_length: int,
    val_frac: float,
    qlora_config: Dict[str, Any],
    train_bs: int,
    gradient_accumulation_steps: int,
    use_gradient_checkpointing: bool,
    eval_bs: int,
    max_epochs: int,
    lr: float,
    optimizer: str,
    warmup_ratio: float,
    logging_steps: int,
    save_total_limit: int,
    seed: int
) -> None:
    """Train a Phi-3 Mini 4K model using QLoRA.

    Args:
        data_path (str): Path to the dataset file in JSON format
        output_dir (str): Directory to save the trained model and logs
        model_name (str): Name of the base model to use, e.g., "microsoft/phi-3-mini-4k-instruct"
        max_length (int): Maximum sequence length for the model
        val_frac (float): Fraction of data to use for validation
        qlora_config (Dict[str, Any]): Configuration for QLoRA, including r, lora_alpha, target_modules, and dropout
        train_bs (int): Batch size for training
        gradient_accumulation_steps (int): Number of gradient accumulation steps
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing
        eval_bs (int): Batch size for evaluation
        max_epochs (int): Maximum number of training epochs
        lr (float): Learning rate
        optimizer (str): Optimizer to use, e.g., "adamw_torch" or "adamw_bnb_4bit"
        warmup_ratio (float): Fraction of total steps for learning rate warmup
        logging_steps (int): Number of steps between logging training metrics
        save_total_limit (int): Maximum number of checkpoints to keep
        seed (int): Random seed for reproducibility
    """

    set_seed(seed)
    # Load data, split into train and validation sets, preprocess
    with open(data_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} examples from {data_path}")
    data = list(map(preprocess, data))
    logger.info("Preprocessed data first sample:")
    logger.info(data[0])
    train_data, val_data = train_test_split(data, test_size=val_frac)
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Validation size: {len(val_data)}")
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Model/tokenizer
    logger.info("Loading model and tokenizer")

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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_gradient_checkpointing
    )

    # QLoRA config
    logger.info("Configuring QLoRA")
    peft_config = LoraConfig(
        r=qlora_config["r"],
        lora_alpha=qlora_config["lora_alpha"],
        target_modules=qlora_config["target_modules"],
        lora_dropout=qlora_config["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Training
    logger.info("Training model")
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=train_bs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=max_epochs,
        learning_rate=lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        save_strategy="epoch",
        eval_strategy="steps",
        logging_steps=logging_steps,
        optim=optimizer,
        report_to="tensorboard",
        lr_scheduler_type="linear",
        completion_only_loss=True,
        overwrite_output_dir=True,
        save_total_limit=save_total_limit,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"Model trained and saved to {output_dir}")


if __name__ == "__main__":
    try:
        with open("params.yaml", "r") as file:
            params = yaml.safe_load(file)
        train_phi3_mini_4k(
            data_path=params["prepare_prompts"]["out_file"],
            output_dir=params["training"]["out_dir"],
            model_name=params["training"]["model_name"],
            max_length=params["training"]["max_len"],
            val_frac=params["training"]["val_frac"],
            qlora_config=params["training"]["qlora_config"],
            train_bs=params["training"]["train_bs"],
            gradient_accumulation_steps=params["training"]["gradient_accumulation_steps"],
            use_gradient_checkpointing=params["training"]["use_gradient_checkpointing"],
            eval_bs=params["training"]["eval_bs"],
            max_epochs=params["training"]["max_epochs"],
            lr=float(params["training"]["lr"]),
            optimizer=params["training"]["optimizer"],
            warmup_ratio=params["training"]["warmup_ratio"],
            logging_steps=params["training"]["logging_steps"],
            save_total_limit=params["training"]["save_total_limit"],
            seed=params["training"]["seed"]
        )
    except Exception as e:
        if os.path.exists(params["training"]["out_dir"]):
            shutil.rmtree(params["training"]["out_dir"])
            logger.info(f"Removed output directory {params['training']['out_dir']}")
        logger.error(f"An error occurred: {e}")
        raise e
