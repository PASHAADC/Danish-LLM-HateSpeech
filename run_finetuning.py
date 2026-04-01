"""
Fine-Tuning Experiments

Usage:
  python run_finetuning.py --split 60_10_30 --variant imbalanced
  python run_finetuning.py --split 80_10_10 --variant balanced --models llama gemma
  python run_finetuning.py --split 60_10_30 --variant balanced --patterns vanilla_qa cot
"""

import argparse
from src.finetuning.run_finetuning import run_finetuning_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning pipeline")
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--patterns", type=str, nargs="+", default=None)
    parser.add_argument("--clear-master", action="store_true")
    parser.add_argument("--split", type=str, default="80_10_10", choices=["80_10_10", "60_10_30"])
    parser.add_argument("--variant", type=str, default="imbalanced", choices=["balanced", "imbalanced", "cross_balanced"])
    args = parser.parse_args()

    run_finetuning_pipeline(
        model_keys=args.models,
        pattern_names=args.patterns,
        clear_master=args.clear_master,
        split_config=args.split,
        variant=args.variant,
    )
