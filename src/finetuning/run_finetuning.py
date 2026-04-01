"""
Fine-Tuning Pipeline
Pattern-by-pattern execution

Flow:
1. For each pattern:
   a. Train all models (base eval + SFT)
   b. Save pattern_results.csv
   c. Create pattern plot
   d. Append to master_results.csv
   e. Clear GPU/storage
2. After all patterns create master plot
"""

import os
import warnings
import logging
import pandas as pd
import shutil
from datetime import datetime

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from ..config import MODELS, BASE_OUTPUT_DIR, PROMPT_PATTERNS, DATA_DIR
from ..data_loader import select_dataset, load_dataset

from .finetune_config import FINETUNE_OUTPUT_DIR
from .data_prep import prepare_sft_data, get_data_statistics
from .trainers import train_sft, cleanup_gpu
from .evaluation import evaluate_all_methods


def load_presplit_data(split_config, variant):
    """
    Load pre-split CSVs from data/dk_hate_processed/splits/.
    split_config: "80_10_10" or "60_10_30"
    variant: "balanced" or "imbalanced"
    """
    split_dir = os.path.join(DATA_DIR, "splits", split_config, variant)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            f"Run prepare_splits.py first."
        )

    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"))

    print(f"  Loaded {split_config}/{variant}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


# Plotting

def create_pattern_plot(csv_path, output_path, pattern_name):
    """Create precision-recall plot for a single pattern."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    MODEL_COLORS = {
        "llama": "#3498DB",
        "mistral": "#E74C3C",
        "gemma": "#9B59B6",
        "qwen": "#2ECC71",
    }

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"No data for {pattern_name}, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    for _, row in df.iterrows():
        model = row['model']
        method = row.get('method', 'base')
        p = row['precision']
        r = row['recall']

        color = MODEL_COLORS.get(model, 'gray')
        marker = 'o' if method == 'base' else '^'
        size = 150 if method == 'base' else 200
        facecolor = 'white' if method == 'base' else color

        ax.scatter(p, r, c=facecolor, marker=marker, s=size,
                   edgecolors=color, linewidths=2, alpha=0.9, zorder=5)

        offset = 0.02
        ax.annotate(f"{model}", (p + offset, r + offset), fontsize=8)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.3)

    ax.text(0.25, 0.85, "OVERSENSITIVE", fontsize=9, ha='center', alpha=0.7)
    ax.text(0.75, 0.25, "CONSERVATIVE", fontsize=9, ha='center', alpha=0.7)
    ax.text(0.25, 0.25, "POOR", fontsize=9, ha='center', alpha=0.7)
    ax.text(0.75, 0.85, "GOOD", fontsize=9, ha='center', alpha=0.7)

    patches = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()]
    base_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                              markeredgecolor='gray', markersize=10, label='Base')
    sft_marker = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                             markeredgecolor='gray', markersize=10, label='SFT')
    ax.legend(handles=patches + [base_marker, sft_marker], loc='lower right')

    ax.set_xlabel('Precision', fontsize=11)
    ax.set_ylabel('Recall', fontsize=11)
    ax.set_title(f'Precision vs Recall: {pattern_name}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")


def create_master_plot(master_csv_path, output_path):
    """
    Master plot with ALL patterns.
    Color = Model, Shape = Pattern, Fill = Method (hollow=base, filled=SFT)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    MODEL_COLORS = {
        "llama": "#3498DB",
        "mistral": "#E74C3C",
        "gemma": "#9B59B6",
        "qwen": "#2ECC71",
    }

    PATTERN_MARKERS = {
        "vanilla_qa": "o",
        "choice_qa": "s",
        "cloze": "D",
        "cot": "^",
        "target": "v",
        "illocutionary": "p",
        "functional": "h",
        "definition": "*",
        "victim_perspective": "X",
        "expert_moderator": "P",
    }

    df = pd.read_csv(master_csv_path)
    if df.empty:
        print("Master CSV empty, skipping master plot")
        return

    fig, ax = plt.subplots(figsize=(14, 12))

    for _, row in df.iterrows():
        model = row['model']
        method = row.get('method', 'base')
        pattern = row['pattern']
        p = row['precision']
        r = row['recall']

        color = MODEL_COLORS.get(model, 'gray')
        marker = PATTERN_MARKERS.get(pattern, 'o')

        if method == 'base':
            facecolor = 'white'
            size = 120
        else:
            facecolor = color
            size = 150

        ax.scatter(p, r, c=facecolor, marker=marker, s=size,
                   edgecolors=color, linewidths=1.5, alpha=0.8, zorder=5)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.3)

    ax.text(0.25, 0.92, "OVERSENSITIVE", fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.6))
    ax.text(0.75, 0.08, "CONSERVATIVE", fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='#D5F5E3', alpha=0.6))
    ax.text(0.25, 0.08, "POOR", fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='#D6DBDF', alpha=0.6))
    ax.text(0.75, 0.92, "GOOD", fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='#ABEBC6', alpha=0.6))

    model_patches = [mpatches.Patch(color=c, label=m.capitalize())
                     for m, c in MODEL_COLORS.items()]
    legend1 = ax.legend(handles=model_patches, loc='upper left',
                        title='Model', fontsize=8, title_fontsize=9)
    ax.add_artist(legend1)

    pattern_handles = []
    for pattern, marker in PATTERN_MARKERS.items():
        if pattern in df['pattern'].values:
            handle = Line2D([0], [0], marker=marker, color='w',
                           markerfacecolor='gray', markeredgecolor='black',
                           markersize=8, label=pattern)
            pattern_handles.append(handle)
    legend2 = ax.legend(handles=pattern_handles, loc='lower left',
                        title='Pattern', fontsize=7, title_fontsize=9,
                        ncol=2)
    ax.add_artist(legend2)

    base_handle = Line2D([0], [0], marker='o', color='w',
                         markerfacecolor='white', markeredgecolor='gray',
                         markersize=10, label='Base (hollow)')
    sft_handle = Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='gray', markeredgecolor='gray',
                        markersize=10, label='SFT (filled)')
    legend3 = ax.legend(handles=[base_handle, sft_handle], loc='center left',
                        bbox_to_anchor=(0, 0.5), title='Method', fontsize=8)

    ax.set_xlabel('Precision', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Master Results: All Patterns x All Models',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Master plot saved: {output_path}")


# Master file management

def get_master_file_path():
    """Get path to master results file."""
    return os.path.join(BASE_OUTPUT_DIR, "metrics", "master_results.csv")


def append_to_master(pattern_results_df):
    """Append pattern results to master CSV, avoiding duplicates."""
    master_path = get_master_file_path()
    os.makedirs(os.path.dirname(master_path), exist_ok=True)

    if os.path.exists(master_path):
        master_df = pd.read_csv(master_path)

        for _, row in pattern_results_df.iterrows():
            mask = (
                (master_df['model'] == row['model']) &
                (master_df['method'] == row['method']) &
                (master_df['pattern'] == row['pattern'])
            )
            master_df = master_df[~mask]

        master_df = pd.concat([master_df, pattern_results_df], ignore_index=True)
    else:
        master_df = pattern_results_df

    master_df.to_csv(master_path, index=False)
    print(f"Master file updated: {master_path} ({len(master_df)} total rows)")
    return master_df


def clear_master_file():
    """Clear master file to start fresh."""
    master_path = get_master_file_path()
    if os.path.exists(master_path):
        os.remove(master_path)
        print(f"Cleared master file: {master_path}")


# Single pattern execution

def run_single_pattern(
    pattern_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    models_to_train: dict,
    results_dir: str,
) -> pd.DataFrame:
    """Run training + evaluation for a single pattern on all models."""
    import time

    print(f"\nPattern: {pattern_name}")

    pattern_results = []

    for model_key, model_name in models_to_train.items():
        print(f"Model: {model_key}")

        adapter_paths = {}
        training_log_dir = os.path.join(results_dir, "training_logs", pattern_name, model_key)
        os.makedirs(training_log_dir, exist_ok=True)

        # Train SFT
        try:
            print(f"\nPreparing data for {model_key}...")
            sft_data, _ = prepare_sft_data(
                train_df, val_df, test_df,
                model_key=model_key,
                pattern_name=pattern_name,
            )

            print(f"Training SFT for {model_key}...")
            try:
                adapter_paths["sft"] = train_sft(
                    model_name=model_name,
                    dataset=sft_data,
                    output_dir=training_log_dir,
                    model_key=model_key,
                    dataset_name=dataset_name,
                    pattern_name=pattern_name,
                )
                print(f"SFT training complete")
            except Exception as e:
                print(f"SFT training failed: {e}")
                adapter_paths["sft"] = None

        except Exception as e:
            print(f"Data prep error: {e}")
            import traceback
            traceback.print_exc()
            cleanup_gpu()
            continue

        # Clear GPU before evaluation
        cleanup_gpu()
        time.sleep(3)
        cleanup_gpu()

        # Evaluate base + SFT on test set only (no data leakage)
        print(f"Evaluating {model_key} (base + SFT) on test set ({len(test_df)} samples)")

        try:
            valid_adapters = {
                k: v for k, v in adapter_paths.items()
                if v is not None
            }

            results = evaluate_all_methods(
                model_key=model_key,
                model_name=model_name,
                data=test_df,
                dataset_name=dataset_name,
                adapter_paths=valid_adapters,
                output_dir=results_dir,
                pattern_name=pattern_name,
            )

            if results.get("pattern"):
                pattern_results.extend(results["pattern"])
                print(f"Evaluation complete: {len(results['pattern'])} results")

        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

        # Clear GPU before next model
        cleanup_gpu()
        time.sleep(2)
        cleanup_gpu()

        print(f"\n{model_key} complete!")

    # Save pattern results
    if pattern_results:
        pattern_df = pd.DataFrame(pattern_results)

        pattern_csv_path = os.path.join(
            results_dir, "metrics", f"{pattern_name}_results.csv"
        )
        pattern_df.to_csv(pattern_csv_path, index=False)
        print(f"Saved: {pattern_csv_path}")

        pattern_plot_path = os.path.join(
            results_dir, "plots", f"{pattern_name}_precision_recall.png"
        )
        create_pattern_plot(pattern_csv_path, pattern_plot_path, pattern_name)

        append_to_master(pattern_df)

        return pattern_df
    else:
        print(f"No results for {pattern_name}")
        return pd.DataFrame()


# Storage cleanup

def cleanup_pattern_artifacts(results_dir, pattern_name):
    """Clean up training checkpoints to save storage."""
    checkpoint_dir = os.path.join(results_dir, "training_logs", pattern_name)

    if os.path.exists(checkpoint_dir):
        for model_dir in os.listdir(checkpoint_dir):
            model_path = os.path.join(checkpoint_dir, model_dir)
            if os.path.isdir(model_path):
                for item in os.listdir(model_path):
                    item_path = os.path.join(model_path, item)
                    if item.startswith("checkpoint-") and os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        print(f"  Cleaned: {item_path}")


# Main pipeline

def run_finetuning_pipeline(
    model_keys=None,
    pattern_names=None,
    clear_master=False,
    split_config="80_10_10",
    variant="imbalanced",
):
    """
    Main pipeline: pattern-by-pattern execution.
    split_config: "80_10_10" or "60_10_30"
    variant: "balanced" or "imbalanced"
    """
    print(f"\nFine-tuning pipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    models_to_train = {k: MODELS[k] for k in model_keys} if model_keys else MODELS
    patterns_to_run = pattern_names if pattern_names else list(PROMPT_PATTERNS.keys())

    print(f"Models: {list(models_to_train.keys())}")
    print(f"Patterns: {patterns_to_run}")
    print(f"Split: {split_config}, Variant: {variant}")

    if clear_master:
        clear_master_file()

    # Load pre-split data
    train_df, val_df, test_df = load_presplit_data(split_config, variant)
    dataset_name = f"{variant}_{split_config}"

    stats = get_data_statistics(train_df)
    print(f"\nDataset: {dataset_name}")
    print(f"  Train: {len(train_df)} samples (hate: {sum(train_df['label']==1)})")
    print(f"  Val:   {len(val_df)} samples (hate: {sum(val_df['label']==1)})")
    print(f"  Test:  {len(test_df)} samples (hate: {sum(test_df['label']==1)})")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        BASE_OUTPUT_DIR, "finetuning_results", f"{dataset_name}_{timestamp}"
    )

    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "training_logs"), exist_ok=True)

    print(f"Output: {results_dir}")

    # Pattern-by-pattern training and evaluation
    all_pattern_results = []

    for i, pattern_name in enumerate(patterns_to_run):
        print(f"\n[{i+1}/{len(patterns_to_run)}] Processing: {pattern_name}")

        try:
            pattern_df = run_single_pattern(
                pattern_name=pattern_name,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                dataset_name=dataset_name,
                models_to_train=models_to_train,
                results_dir=results_dir,
            )

            if not pattern_df.empty:
                all_pattern_results.append(pattern_df)

            cleanup_pattern_artifacts(results_dir, pattern_name)
            cleanup_gpu()

            print(f"Pattern {pattern_name} complete")

        except Exception as e:
            print(f"Pattern {pattern_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create master plot
    master_path = get_master_file_path()
    if os.path.exists(master_path):
        master_plot_path = os.path.join(results_dir, "plots", "master_precision_recall.png")
        create_master_plot(master_path, master_plot_path)

        f1_plot_path = os.path.join(results_dir, "plots", "f1_by_pattern.png")
        plot_f1_by_pattern(master_path, f1_plot_path)

        master_plot_copy = os.path.join(BASE_OUTPUT_DIR, "master_precision_recall.png")
        shutil.copy(master_plot_path, master_plot_copy)
        print(f"Master plot also saved to: {master_plot_copy}")

    # Final summary
    print(f"\nPipeline complete")
    print(f"Results: {results_dir}")
    print(f"Master CSV: {master_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if os.path.exists(master_path):
        master_df = pd.read_csv(master_path)
        print(f"\nSummary: {len(master_df)} total results")
        print("\nBy Pattern:")
        print(master_df.groupby('pattern')['f1'].mean().sort_values(ascending=False).to_string())
        print("\nBy Model (SFT only):")
        sft_df = master_df[master_df['method'] == 'sft']
        if not sft_df.empty:
            print(sft_df.groupby('model')['f1'].mean().sort_values(ascending=False).to_string())


# Evaluation only mode

def run_evaluation_only(
    adapter_base_dir: str,
    train_dataset_name: str,
    eval_dataset_name: str = None,
    model_keys: list = None,
    pattern_names: list = None,
):
    """
    Run evaluation only (skip training).
    Use this when you have already trained adapters and want to
    evaluate on a different dataset.
    """
    from ..config import DATASET_OPTIONS

    print("\nEvaluation only mode")

    if eval_dataset_name is None:
        eval_dataset_name = train_dataset_name

    dataset_config = None
    for key, config in DATASET_OPTIONS.items():
        if config["name"] == eval_dataset_name:
            dataset_config = config
            break

    if dataset_config is None:
        print(f"ERROR: Dataset '{eval_dataset_name}' not found")
        print(f"Available: {[c['name'] for c in DATASET_OPTIONS.values()]}")
        return

    data = load_dataset(dataset_config)
    if data is None:
        print("ERROR: Could not load dataset")
        return

    stats = get_data_statistics(data)
    print(f"\nEval Dataset: {eval_dataset_name}")
    print(f"  Total: {stats['total_samples']} samples")
    print(f"  Hate: {stats['hate_samples']} ({stats['hate_ratio']*100:.1f}%)")

    models_to_eval = {k: MODELS[k] for k in model_keys} if model_keys else MODELS
    patterns_to_eval = pattern_names if pattern_names else list(PROMPT_PATTERNS.keys())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        BASE_OUTPUT_DIR, "eval_results", f"{eval_dataset_name}_{timestamp}"
    )
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    print(f"Output: {results_dir}")
    print(f"Looking for adapters in: {adapter_base_dir}/{train_dataset_name}/")

    all_results = []

    for pattern_name in patterns_to_eval:
        print(f"\nPattern: {pattern_name}")

        for model_key, model_name in models_to_eval.items():
            adapter_path = os.path.join(
                adapter_base_dir, train_dataset_name, model_key, "sft", "adapter"
            )

            pattern_adapter_path = os.path.join(
                adapter_base_dir, train_dataset_name, pattern_name, model_key, "sft", "adapter"
            )

            if os.path.exists(pattern_adapter_path):
                adapter_path = pattern_adapter_path

            adapter_paths = {}
            if os.path.exists(adapter_path):
                adapter_paths["sft"] = adapter_path
                print(f"Found adapter: {model_key}")
            else:
                print(f"No adapter found for {model_key} at {adapter_path}")

            try:
                results = evaluate_all_methods(
                    model_key=model_key,
                    model_name=model_name,
                    data=data,
                    dataset_name=eval_dataset_name,
                    adapter_paths=adapter_paths,
                    output_dir=results_dir,
                    pattern_name=pattern_name,
                )

                if results.get("pattern"):
                    all_results.extend(results["pattern"])

            except Exception as e:
                print(f"Evaluation failed: {e}")
                import traceback
                traceback.print_exc()

            cleanup_gpu()

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(results_dir, "metrics", "eval_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved: {results_path}")

        plot_path = os.path.join(results_dir, "plots", "eval_precision_recall.png")
        create_master_plot(results_path, plot_path)

    print(f"\nEvaluation complete!")


def plot_f1_by_pattern(master_csv_path, output_path=None):
    """Create horizontal bar chart comparing Base vs SFT F1 by pattern."""
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = pd.read_csv(master_csv_path)

    pattern_method = df.groupby(['pattern', 'method'])['f1'].mean().unstack()
    pattern_method = pattern_method.sort_values('sft', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    x = np.arange(len(pattern_method.index))
    width = 0.35

    bars1 = ax.barh(x - width/2, pattern_method['base'], width,
                    label='Base', color='#95A5A6', edgecolor='black')
    bars2 = ax.barh(x + width/2, pattern_method['sft'], width,
                    label='SFT', color='#27AE60', edgecolor='black')

    ax.set_yticks(x)
    ax.set_yticklabels(pattern_method.index, fontsize=10)
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('Average F1 by Pattern: Base vs SFT', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    for bar in bars1:
        width_val = bar.get_width()
        ax.text(width_val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width_val:.2f}', va='center', fontsize=8, color='gray')
    for bar in bars2:
        width_val = bar.get_width()
        ax.text(width_val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width_val:.2f}', va='center', fontsize=8, color='darkgreen')

    plt.tight_layout()

    if output_path is None:
        output_path = master_csv_path.replace('.csv', '_f1_by_pattern.png')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    return output_path


# Entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tuning pipeline with cross-prompt evaluation support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
  # Standard finetuning (train and eval on same pattern)
  python -m src.finetuning.run_finetuning --models llama --patterns vanilla_qa

  # Cross-prompt: train on one, eval on many
  python -m src.finetuning.run_finetuning --cross-prompt \\
      --models llama \\
      --train-patterns vanilla_qa \\
      --eval-patterns vanilla_qa definition cot target

  # Cross-prompt: full NxN matrix
  python -m src.finetuning.run_finetuning --cross-prompt \\
      --models llama \\
      --train-patterns vanilla_qa definition cot \\
      --eval-patterns vanilla_qa definition cot

  # Evaluation only mode
  python -m src.finetuning.run_finetuning --eval-only \\
      --train-dataset dkhate_1000_cleaned
        """
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to train/evaluate (e.g., llama qwen mistral gemma)"
    )

    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=None,
        help="Patterns for standard mode - same pattern for train & eval"
    )

    parser.add_argument(
        "--clear-master",
        action="store_true",
        help="Clear master results file before starting"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="80_10_10",
        choices=["80_10_10", "60_10_30"],
        help="Which pre-split to use (default: 80_10_10)"
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="imbalanced",
        choices=["balanced", "imbalanced", "cross_balanced"],
        help="Dataset variant (default: imbalanced)"
    )

    parser.add_argument(
        "--cross-prompt",
        action="store_true",
        help="Enable cross-prompt evaluation mode"
    )

    parser.add_argument(
        "--train-patterns",
        type=str,
        nargs="+",
        default=None,
        help="Patterns to use for TRAINING in cross-prompt mode"
    )

    parser.add_argument(
        "--eval-patterns",
        type=str,
        nargs="+",
        default=None,
        help="Patterns to use for EVALUATION in cross-prompt mode"
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (skip training)"
    )

    parser.add_argument(
        "--train-dataset",
        type=str,
        default="dkhate_1000_cleaned",
        help="Dataset name used for training (to find adapters)"
    )

    parser.add_argument(
        "--eval-dataset",
        type=str,
        default=None,
        help="Dataset name to evaluate on (defaults to train-dataset)"
    )

    args = parser.parse_args()

    if args.cross_prompt:
        from .run_cross_prompt import run_cross_prompt_experiment

        run_cross_prompt_experiment(
            model_keys=args.models,
            train_patterns=args.train_patterns,
            eval_patterns=args.eval_patterns,
            clear_master=args.clear_master,
        )

    elif args.eval_only:
        run_evaluation_only(
            adapter_base_dir=FINETUNE_OUTPUT_DIR,
            train_dataset_name=args.train_dataset,
            eval_dataset_name=args.eval_dataset,
            model_keys=args.models,
            pattern_names=args.patterns,
        )

    else:
        run_finetuning_pipeline(
            model_keys=args.models,
            pattern_names=args.patterns,
            clear_master=args.clear_master,
            split_config=args.split,
            variant=args.variant,
        )
