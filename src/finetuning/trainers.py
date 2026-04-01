"""
Fine-Tuning Trainers
"""

import os
import gc
import torch
from typing import Tuple, Any
from datasets import DatasetDict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

from trl import SFTTrainer, SFTConfig

from .finetune_config import (
    LORA_CONFIG,
    SFT_CONFIG,
    get_adapter_path,
)
from ..config import DEVICE


def cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def create_lora_config(task_type: str = "CAUSAL_LM") -> LoraConfig:
    """Create LoRA configuration."""
    task = TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else TaskType.SEQ_CLS

    config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias=LORA_CONFIG["bias"],
        task_type=task,
    )

    print(f"LoRA config: r={config.r}, alpha={config.lora_alpha}, targets={config.target_modules}")

    return config


def setup_tokenizer(model_name: str) -> AutoTokenizer:
    """Setup tokenizer with proper padding token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
    print(f"Tokenizer ready: {model_name} (pad_token='{tokenizer.pad_token}')")

    return tokenizer


def setup_base_model(model_name: str, training: bool) -> Tuple[Any, AutoTokenizer]:
    """Load base model and tokenizer."""
    print(f"\nLoading base model: {model_name}")

    cleanup_gpu()

    tokenizer = setup_tokenizer(model_name)

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if training else DEVICE,
        "torch_dtype": torch.bfloat16,
    }

    if "gemma" in model_name.lower():
        load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    print("Model loaded successfully")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return model, tokenizer


def load_finetuned_model(
    base_model_name: str, adapter_path: str
) -> Tuple[Any, AutoTokenizer]:
    """Load base model with LoRA adapter."""
    print(f"\nLoading fine-tuned model: base={base_model_name}, adapter={adapter_path}")

    model, tokenizer = setup_base_model(base_model_name, training=False)
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Fine-tuned model loaded")
    return model, tokenizer


def train_sft(
    model_name: str,
    dataset: DatasetDict,
    output_dir: str,
    model_key: str,
    dataset_name: str,
    pattern_name: str = "vanilla_qa",
) -> str:
    """Train with SFT using LoRA for specific pattern."""

    print(f"\nSFT training: {model_key} - {pattern_name}")

    cleanup_gpu()

    adapter_path = get_adapter_path(model_key, "sft", dataset_name)
    os.makedirs(adapter_path, exist_ok=True)

    model, tokenizer = setup_base_model(model_name, training=True)

    peft_config = create_lora_config("CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"  Trainable parameters: {trainable_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    sft_config = SFTConfig(
        # Only compute loss on the completion tokens
        completion_only_loss=True,
        output_dir=output_dir,
        num_train_epochs=SFT_CONFIG["num_train_epochs"],
        per_device_train_batch_size=SFT_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=SFT_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=SFT_CONFIG["gradient_accumulation_steps"],
        learning_rate=SFT_CONFIG["learning_rate"],
        weight_decay=SFT_CONFIG["weight_decay"],
        warmup_ratio=SFT_CONFIG["warmup_ratio"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=SFT_CONFIG["logging_steps"],
        eval_steps=SFT_CONFIG["logging_steps"],
        save_strategy=SFT_CONFIG["save_strategy"],
        eval_strategy=SFT_CONFIG["evaluation_strategy"],
        report_to="none",
        optim="adamw_torch",
        max_length=SFT_CONFIG["max_length"],
        max_grad_norm=1,
        save_total_limit=5,
        save_only_model=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_first_step=True,
    )

    print("Starting SFT training")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.0
            )
        ],
    )

    trainer.train()

    print(f"Saving adapter to {adapter_path}")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    del model, trainer
    cleanup_gpu()

    print(f"SFT training complete for {model_key}")
    return adapter_path
