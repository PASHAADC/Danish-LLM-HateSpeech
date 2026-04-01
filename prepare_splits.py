"""
Pre-split datasets into train/val/test for balanced, imbalanced,
and cross_balanced (balanced train, imbalanced test) experiments.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

DATA_DIR = "data/dk_hate_processed"
OUTPUT_DIR = os.path.join(DATA_DIR, "splits")
SEED = 42

SPLIT_CONFIGS = {
    "80_10_10": {"test": 0.10, "val": 0.10},
    "60_10_30": {"test": 0.30, "val": 0.10},
}


def create_split(df, test_ratio, val_ratio):
    remainder, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df["label"], random_state=SEED
    )

    adjusted_val = val_ratio / (1.0 - test_ratio)
    train_df, val_df = train_test_split(
        remainder, test_size=adjusted_val, stratify=remainder["label"], random_state=SEED
    )

    return train_df, val_df, test_df


def balance_downsample(df):
    minority_count = df["label"].value_counts().min()
    hate = df[df["label"] == 1].sample(n=minority_count, random_state=SEED)
    non_hate = df[df["label"] == 0].sample(n=minority_count, random_state=SEED)
    return pd.concat([hate, non_hate]).sample(frac=1, random_state=SEED).reset_index(drop=True)


def verify_no_leakage(train_df, val_df, test_df, name):
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])

    assert train_ids.isdisjoint(test_ids), f"{name}: train/test overlap"
    assert val_ids.isdisjoint(test_ids), f"{name}: val/test overlap"
    assert train_ids.isdisjoint(val_ids), f"{name}: train/val overlap"
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(train_df) + len(val_df) + len(test_df), \
        f"{name}: duplicate IDs within a split"


def save_split(train_df, val_df, test_df, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"  {label}")
    print(f"    Train: {len(train_df)} (hate={sum(train_df['label']==1)}, non-hate={sum(train_df['label']==0)})")
    print(f"    Val:   {len(val_df)} (hate={sum(val_df['label']==1)}, non-hate={sum(val_df['label']==0)})")
    print(f"    Test:  {len(test_df)} (hate={sum(test_df['label']==1)}, non-hate={sum(test_df['label']==0)})")


if __name__ == "__main__":
    complete_df = pd.read_csv(os.path.join(DATA_DIR, "dkhate_complete.csv"))
    balanced_df = pd.read_csv(os.path.join(DATA_DIR, "dkhate_balanced_1000.csv"))

    for config_name, ratios in SPLIT_CONFIGS.items():
        print(f"\nSplit: {config_name}")

        # 1. Imbalanced: split complete dataset independently
        imb_train, imb_val, imb_test = create_split(complete_df, ratios["test"], ratios["val"])
        verify_no_leakage(imb_train, imb_val, imb_test, f"{config_name}/imbalanced")
        save_split(imb_train, imb_val, imb_test,
                   os.path.join(OUTPUT_DIR, config_name, "imbalanced"), "imbalanced")

        # 2. Balanced: split balanced dataset independently
        bal_train, bal_val, bal_test = create_split(balanced_df, ratios["test"], ratios["val"])
        verify_no_leakage(bal_train, bal_val, bal_test, f"{config_name}/balanced")
        save_split(bal_train, bal_val, bal_test,
                   os.path.join(OUTPUT_DIR, config_name, "balanced"), "balanced")

        # 3. Cross-balanced: balanced train/val from imbalanced remainder, imbalanced test
        #    Hold out imbalanced test first, then downsample remainder to create balanced train/val
        imb_remainder = pd.concat([imb_train, imb_val])
        cross_bal_pool = balance_downsample(imb_remainder)
        adjusted_val = ratios["val"] / (1.0 - ratios["test"])
        cross_train, cross_val = train_test_split(
            cross_bal_pool, test_size=adjusted_val,
            stratify=cross_bal_pool["label"], random_state=SEED
        )
        verify_no_leakage(cross_train, cross_val, imb_test, f"{config_name}/cross_balanced")
        save_split(cross_train, cross_val, imb_test,
                   os.path.join(OUTPUT_DIR, config_name, "cross_balanced"), "cross_balanced")
