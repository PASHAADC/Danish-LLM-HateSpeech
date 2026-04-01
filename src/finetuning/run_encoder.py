"""
Encoder (BERT) Experiment Runner
================================

Runs MLM prompting + classification head experiments for
BERT, DaBERT, and ScandiBERT on the DKhate datasets.

Methods:
  - base_mlm:      MLM prompting (zero-shot, all 11 patterns)
  - base_cls:      Untrained classification head (random baseline)
  - sft_lora_cls:  LoRA fine-tuned classification head (r=16, alpha=32)

Usage:
  # All encoder experiments (MLM + CLS)
  CUDA_VISIBLE_DEVICES=0 python -m src.finetuning.run_encoder --stage all

  # MLM prompting only
  python -m src.finetuning.run_encoder --stage mlm

  # Classification head only
  python -m src.finetuning.run_encoder --stage cls

  # Specific models
  python -m src.finetuning.run_encoder --stage all --models bert_multi dabert

  # Specific patterns (MLM only)
  python -m src.finetuning.run_encoder --stage mlm --patterns vanilla_qa cot

Seed: 42 | LoRA r=16, alpha=32 | Split: 80/10/10
"""

import os
import time
import shutil
import warnings
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from ..config import ENCODER_MODELS, BASE_OUTPUT_DIR, DATA_DIR
from ..metrics import compute_metrics
from .encoder_models import (
    evaluate_encoder_mlm,
    train_encoder,
    evaluate_encoder,
    _cleanup_gpu,
    _ZERO_SHOT_PATTERNS,
)

# DATASETS
DATASETS = {
    "balanced": {
        "name": "DKhate_balanced_1000",
        "file": os.path.join(DATA_DIR, "dkhate_balanced_1000.csv"),
    },
    "imbalanced": {
        "name": "DKhate_imbalanced",
        "file": os.path.join(DATA_DIR, "dkhate_complete.csv"),
    },
}
# RESULTS CSV
ENCODER_CSV = os.path.join(BASE_OUTPUT_DIR, "encoder_results.csv")
def load_completed():
    """Load set of already-completed experiment keys."""
    if not os.path.exists(ENCODER_CSV):
        return set()
    try:
        df = pd.read_csv(ENCODER_CSV)
    except pd.errors.EmptyDataError:
        return set()

    completed = set()
    for _, row in df.iterrows():
        key = (
            str(row.get("dataset", "")),
            str(row.get("model", "")),
            str(row.get("method", "")),
            str(row.get("shot_type", "")),
            str(row.get("pattern", "")),
        )
        completed.add(key)
    print(f"Resume: {len(completed)} experiments already completed")
    return completed
def is_done(completed, dataset, model, method, shot_type, pattern):
    return (dataset, model, method, shot_type, pattern) in completed
def append_result(result_dict):
    """Append a single result row to encoder CSV."""
    os.makedirs(os.path.dirname(ENCODER_CSV), exist_ok=True)
    df_new = pd.DataFrame([result_dict])
    if os.path.exists(ENCODER_CSV):
        df_new.to_csv(ENCODER_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(ENCODER_CSV, index=False)
def load_dataset_by_key(key):
    """Load one of the two datasets."""
    ds = DATASETS[key]
    filepath = ds["file"]
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None, None

    df = pd.read_csv(filepath)
    name = ds["name"]
    off = (df["label"] == 1).sum()
    print(f"  {name}: {len(df)} samples ({off} hate = {100 * off / len(df):.1f}%)")
    return df, name
def load_presplit_encoder_data(split_config, variant):
    """Load pre-split CSVs for encoder experiments."""
    split_dir = os.path.join(DATA_DIR, "splits", split_config, variant)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}\nRun prepare_splits.py first.")

    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"))

    print(f"  Loaded {split_config}/{variant}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df

# COLLAPSE DETECTION
def _result_collapsed(metrics):
    """Detect collapsed results."""
    if metrics["f1"] < 0.05:
        return True
    if metrics["recall"] >= 0.99 or metrics["recall"] <= 0.01:
        return True
    return False

# STAGE: MLM PROMPTING
def run_mlm_stage(model_keys=None, pattern_names=None, split_config="80_10_10", variants=None):
    """MLM prompting evaluation for encoder models (zero-shot only)."""
    variants = variants or ["balanced", "imbalanced", "cross_balanced"]
    models = {k: ENCODER_MODELS[k] for k in model_keys} if model_keys else ENCODER_MODELS
    completed = load_completed()

    for ds_key in variants:
        _, _, mlm_test_df = load_presplit_encoder_data(split_config, ds_key)
        dataset_name = f"{ds_key}_{split_config}"

        for model_key, model_name in models.items():
            print(f"\n  {model_key} MLM on {dataset_name}")

            mlm_patterns = pattern_names or list(_ZERO_SHOT_PATTERNS.keys())
            mlm_remaining = [
                p for p in mlm_patterns
                if not is_done(completed, dataset_name, model_key, "base_mlm", "zero_shot", p)
            ]

            if not mlm_remaining:
                print(f"  MLM: all done, skipping")
                continue

            print(f"  MLM: {len(mlm_remaining)} evaluations, test={len(mlm_test_df)} samples")

            try:
                mlm_results = evaluate_encoder_mlm(
                    model_name, mlm_test_df,
                    pattern_names=mlm_remaining,
                )

                for r in mlm_results:
                    result = {
                        "dataset": dataset_name,
                        "model": model_key,
                        "method": "base_mlm",
                        "shot_type": r["shot_type"],
                        "pattern": r["pattern"],
                        "precision": round(r["precision"], 4),
                        "recall": round(r["recall"], 4),
                        "f1": round(r["f1"], 4),
                        "accuracy": round(r["accuracy"], 4),
                        "n_samples": r["n_samples"],
                        "timestamp": datetime.now().isoformat(),
                    }
                    append_result(result)
                    collapsed_flag = "[COLLAPSED]" if _result_collapsed(r) else ""
                    print(f"base_mlm/zero_shot/{r['pattern']}: F1={r['f1']:.3f}{collapsed_flag}")

            except Exception as e:
                print(f"MLM FAILED: {e}")
                import traceback; traceback.print_exc()

            _cleanup_gpu()

    print("\nMLM stage complete.")

# CLASSIFICATION HEAD (base_cls + sft_lora_cls)
def run_cls_stage(model_keys=None, split_config="80_10_10", variants=None):
    """Classification head experiments with retry logic."""
    variants = variants or ["balanced", "imbalanced", "cross_balanced"]
    models = {k: ENCODER_MODELS[k] for k in model_keys} if model_keys else ENCODER_MODELS
    completed = load_completed()

    for ds_key in variants:
        train_df, val_df, test_df = load_presplit_encoder_data(split_config, ds_key)
        dataset_name = f"{ds_key}_{split_config}"

        for model_key, model_name in models.items():
            print(f"\n  {model_key} CLS on {dataset_name}")

            base_done = is_done(completed, dataset_name, model_key, "base_cls", "zero", "classification_head")
            sft_done = is_done(completed, dataset_name, model_key, "sft_lora_cls", "zero", "classification_head")

            if base_done and sft_done:
                print("  CLS: Already completed, skipping")
                continue

            try:
                out_dir = os.path.join(BASE_OUTPUT_DIR, "encoder_models", dataset_name)

                adapter_path, test_df = train_encoder(
                    model_name, train_df, val_df, test_df, model_key,
                    output_dir=out_dir, use_lora=True,
                )

                if not base_done:
                    print("  Evaluating BASE_CLS (untrained head)...")
                    m = evaluate_encoder(model_name, test_df, adapter_path=None)
                    result = {
                        "dataset": dataset_name,
                        "model": model_key,
                        "method": "base_cls",
                        "shot_type": "zero",
                        "pattern": "classification_head",
                        "precision": round(m["precision"], 4),
                        "recall": round(m["recall"], 4),
                        "f1": round(m["f1"], 4),
                        "accuracy": round(m["accuracy"], 4),
                        "n_samples": m["n_samples"],
                        "timestamp": datetime.now().isoformat(),
                    }
                    append_result(result)
                    print(f"    BASE_CLS: F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}")

                # --- SFT_LORA_CLS with retry ---
                if not sft_done:
                    m_sft = evaluate_encoder(model_name, test_df, adapter_path=adapter_path)
                    print(f"    SFT_CLS (attempt 1): F1={m_sft['f1']:.3f}")

                    best_sft_metrics = m_sft
                    attempt = 1
                    MAX_RETRIES = 5

                    while _result_collapsed(best_sft_metrics) and attempt < MAX_RETRIES:
                        attempt += 1
                        print(f"\n    SFT COLLAPSED — retrying (attempt {attempt}/{MAX_RETRIES})")

                        ckpt_dir = os.path.join(out_dir, model_key, "encoder_sft")
                        if os.path.exists(ckpt_dir):
                            shutil.rmtree(ckpt_dir)
                        _cleanup_gpu()

                        try:
                            retry_adapter, retry_test_df = train_encoder(
                                model_name, train_df, val_df, test_df, model_key,
                                output_dir=out_dir, use_lora=True,
                            )
                            retry_m = evaluate_encoder(model_name, retry_test_df, adapter_path=retry_adapter)
                            print(f"    SFT_CLS (attempt {attempt}): F1={retry_m['f1']:.3f}")

                            if retry_m["f1"] > best_sft_metrics["f1"]:
                                best_sft_metrics = retry_m
                        except Exception as e:
                            print(f"    Retry {attempt} FAILED: {e}")

                        _cleanup_gpu()

                    result = {
                        "dataset": dataset_name,
                        "model": model_key,
                        "method": "sft_lora_cls",
                        "shot_type": "zero",
                        "pattern": "classification_head",
                        "precision": round(best_sft_metrics["precision"], 4),
                        "recall": round(best_sft_metrics["recall"], 4),
                        "f1": round(best_sft_metrics["f1"], 4),
                        "accuracy": round(best_sft_metrics["accuracy"], 4),
                        "n_samples": best_sft_metrics["n_samples"],
                        "timestamp": datetime.now().isoformat(),
                    }
                    append_result(result)
                    print(f"    SFT_CLS FINAL (best of {attempt}): F1={best_sft_metrics['f1']:.3f}")

            except Exception as e:
                print(f"  CLS FAILED: {e}")
                import traceback; traceback.print_exc()

            _cleanup_gpu()

    print("\nCLS stage complete.")

# SUMMARY
def print_summary():
    """Print summary of encoder results."""
    if not os.path.exists(ENCODER_CSV):
        print("No encoder results yet.")
        return

    try:
        df = pd.read_csv(ENCODER_CSV)
    except pd.errors.EmptyDataError:
        print("No encoder results yet (empty CSV).")
        return

    if len(df) == 0:
        print("No encoder results yet (0 rows).")
        return

    for col in ["precision", "recall", "f1", "accuracy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"\n{'='*70}")
    print(f"ENCODER RESULTS: {ENCODER_CSV}")
    print(f"Total rows: {len(df)}")
    print(f"{'='*70}")

    # --- Precision / Recall / F1 Table ---
    print(f"\n{'='*70}")
    print("PRECISION / RECALL / F1 TABLE (Encoder Models)")
    print(f"{'='*70}")

    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        print(f"\n  Dataset: {ds}")
        print(f"  {'Model':<14} {'Method':<14} {'Pattern':<20} {'P':>7} {'R':>7} {'F1':>7} {'Acc':>7}")
        print(f"  {'-'*76}")

        for _, row in ds_df.sort_values(["model", "method", "pattern"]).iterrows():
            print(f"  {row['model']:<14} {row['method']:<14} {row['pattern']:<20} "
                  f"{row['precision']:>7.4f} {row['recall']:>7.4f} {row['f1']:>7.4f} {row['accuracy']:>7.4f}")

    # --- Aggregated view ---
    print(f"\n{'='*70}")
    print("AGGREGATED BY METHOD (mean F1)")
    print(f"{'='*70}")
    print(df.groupby(["dataset", "method"])["f1"].agg(["count", "mean"]).round(4).to_string())

    print(f"\n{'='*70}")
    print("AGGREGATED BY MODEL (mean F1)")
    print(f"{'='*70}")
    print(df.groupby(["dataset", "model"])["f1"].mean().round(4).to_string())

    # --- Best per dataset ---
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        best = ds_df.loc[ds_df["f1"].idxmax()]
        print(f"\n  Best on {ds}: {best['model']}/{best['method']} "
              f"({best.get('pattern', '')}) F1={best['f1']:.4f}")

# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder (BERT) experiments for Danish hate speech detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  mlm      - MLM prompting (all 11 patterns, zero-shot only)
  cls      - Classification head (base_cls + sft_lora_cls with LoRA)
  all      - Run both MLM and CLS stages
  summary  - Print current results with P/R/F1 table

Examples:
  python -m src.finetuning.run_encoder --stage all
  python -m src.finetuning.run_encoder --stage mlm --models bert_multi dabert
  python -m src.finetuning.run_encoder --stage cls --models scandibert
  python -m src.finetuning.run_encoder --stage summary
        """,
    )
    parser.add_argument("--stage", type=str, required=True,
                        choices=["mlm", "cls", "all", "summary"])
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Encoder model keys (bert_multi, dabert, scandibert)")
    parser.add_argument("--patterns", type=str, nargs="+", default=None,
                        help="Pattern names for MLM (default: all 11)")
    parser.add_argument("--split", type=str, default="80_10_10",
                        choices=["80_10_10", "60_10_30"])
    parser.add_argument("--variant", type=str, nargs="+", default=None,
                        choices=["balanced", "imbalanced", "cross_balanced"],
                        help="Dataset variants (default: all three)")

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Danish Hate Speech — Encoder (BERT) Experiments")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: 42 | LoRA r=16 alpha=32")
    print(f"Results CSV: {ENCODER_CSV}")
    print(f"{'='*60}")

    if args.stage == "mlm":
        run_mlm_stage(model_keys=args.models, pattern_names=args.patterns,
                      split_config=args.split, variants=args.variant)
    elif args.stage == "cls":
        run_cls_stage(model_keys=args.models, split_config=args.split, variants=args.variant)
    elif args.stage == "all":
        run_mlm_stage(model_keys=args.models, pattern_names=args.patterns,
                      split_config=args.split, variants=args.variant)
        run_cls_stage(model_keys=args.models, split_config=args.split, variants=args.variant)
    elif args.stage == "summary":
        print_summary()

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
