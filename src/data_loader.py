"""
Dataset loading and management
"""

import os
import pandas as pd
from .config import DATASET_OPTIONS, DATA_DIR


def select_dataset():
    """Interactive dataset selection menu."""
    print("SELECT DATASET")

    for key, dataset in DATASET_OPTIONS.items():
        print(f"  [{key}] {dataset['description']}")
        print(f"      File: {dataset['file']}")

    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        if choice in DATASET_OPTIONS:
            selected = DATASET_OPTIONS[choice]
            print(f"\nSelected: {selected['description']}")
            return selected
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


def load_dataset(dataset_config):
    """Load specified dataset."""
    filepath = os.path.join(DATA_DIR, dataset_config["file"])
    print(f"\nLoading dataset from {filepath}")

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)

    off_count = sum(df["label"] == 1)
    not_count = sum(df["label"] == 0)
    print(f"Loaded {len(df)} samples")
    print(f"OFF (hate): {off_count} ({100 * off_count / len(df):.1f}%)")
    print(f"NOT (normal): {not_count} ({100 * not_count / len(df):.1f}%)")

    return df
