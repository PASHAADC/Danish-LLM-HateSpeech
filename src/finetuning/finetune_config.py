"""
Finetuning Configuration
"""

import os

from ..config import BASE_OUTPUT_DIR
from ..config import PROMPT_PATTERNS

# LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into the model
LORA_CONFIG = {
    "r": 16,
    # Scaling factor: effective learning rate = alpha / r
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# SFT config -- loss: standard cross-entropy on completion tokens
SFT_CONFIG = {
    # Let early stopping decide when to stop
    "num_train_epochs": 10,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "gradient_checkpointing": True,
    "logging_steps": 10,
    "save_strategy": "steps",
    "evaluation_strategy": "steps",
    "max_length": 1024,
}

# Splits are now handled by prepare_splits.py (no on-the-fly splitting)
# Kept for backward compatibility with encoder_models.py and run_cross_prompt.py
SPLIT_CONFIG = {
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "seed": 42,
}

# Output directories
FINETUNE_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "finetuned_models")


def get_adapter_path(model_key: str, method: str, dataset_name: str) -> str:
    return os.path.join(FINETUNE_OUTPUT_DIR, dataset_name, model_key, method, "adapter")


RUN_FINETUNED_PATTERN_ANALYSIS = True

# Prompt templates for fine-tuning

SFT_PROMPT_TEMPLATE2 = """### Instruktion:
Analyser f\u00f8lgende tekst og afg\u00f8r om det er hadefuld tale.
Svar kun med "Ja" (hadefuld) eller "Nej" (ikke hadefuld).

### Tekst:
{text}

### Svar:
"""


def get_responses(pattern_name: str = "vanilla_qa") -> tuple:
    """Get the positive/negative responses for a pattern."""
    pattern = PROMPT_PATTERNS[pattern_name]
    return pattern["positive"], pattern["negative"]


def get_prompt_template(model_key: str, pattern_name: str = "vanilla_qa") -> str:

    base = PROMPT_PATTERNS[pattern_name]["template"]

    if model_key == "llama":
        # Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{base}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    elif model_key == "mistral":
        # Reference: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
        # Mistral does NOT natively support system prompts
        return f"<s>[INST] {base} [/INST]"

    elif model_key == "gemma":
        # Reference: https://ai.google.dev/gemma/docs/core/prompt-structure
        # Gemma does NOT support system prompts
        return (
            "<bos><start_of_turn>user\n"
            f"{base}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    elif model_key == "qwen":
        # Reference: https://qwen.readthedocs.io/en/latest/getting_started/concepts.html
        return (
            "<|im_start|>user\n"
            f"{base}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    else:
        return base
