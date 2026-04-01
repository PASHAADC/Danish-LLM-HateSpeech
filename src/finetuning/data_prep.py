"""
Data Preparation for Fine-Tuning
"""

import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Any, Dict, Tuple

from .finetune_config import (
    SFT_PROMPT_TEMPLATE2,
    get_prompt_template,
    get_responses,
)


def prepare_sft_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_key: str = None,
    pattern_name: str = "vanilla_qa",
) -> Tuple[DatasetDict, pd.DataFrame]:
    """Format pre-split data for SFT training."""

    prompt_template = get_prompt_template(model_key, pattern_name) if model_key else SFT_PROMPT_TEMPLATE2
    positive_response, negative_response = get_responses(pattern_name)
    print(f"  Using template for: {model_key or 'default'}")

    def format_sft_sample(row: pd.Series) -> Dict[str, str]:
        label_text = positive_response if row["label"] == 1 else negative_response
        prompt = prompt_template.format(text=row["text"])
        return {"prompt": prompt, "completion": " " + label_text}

    train_data = [format_sft_sample(row) for _, row in train_df.iterrows()]
    val_data = [format_sft_sample(row) for _, row in val_df.iterrows()]
    test_data = [format_sft_sample(row) for _, row in test_df.iterrows()]

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data),
        }
    )

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return dataset, test_df


def get_data_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    stats = {
        "total_samples": len(df),
        "hate_samples": int((df["label"] == 1).sum()),
        "non_hate_samples": int((df["label"] == 0).sum()),
        "hate_ratio": float((df["label"] == 1).mean()),
        "avg_text_length": float(df["text"].str.len().mean()),
        "max_text_length": int(df["text"].str.len().max()),
        "min_text_length": int(df["text"].str.len().min()),
    }

    return stats
