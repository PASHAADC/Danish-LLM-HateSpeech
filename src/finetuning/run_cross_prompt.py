"""
Cross-Prompt Evaluation Module
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

from ..config import MODELS, BASE_OUTPUT_DIR, PROMPT_PATTERNS
from ..models import classify_with_pattern
from ..metrics import compute_metrics

from .data_prep import prepare_sft_data, get_data_statistics, split_dataframe
from .finetune_config import SPLIT_CONFIG
from .trainers import train_sft, cleanup_gpu
from .evaluation import load_model_with_adapter, save_detailed_predictions


def create_cross_prompt_split(df, train_ratio=0.8, seed=42):
    """
    Create 80/20 train/val split for cross-prompt experiments.
    No test split here because we test on the held-out set (A-B).
    """
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        stratify=df["label"],
        random_state=seed,
    )

    return train_df, val_df


def evaluate_with_pattern(
    model,
    tokenizer,
    test_data,
    eval_pattern_name,
    train_pattern_name,
    model_key,
    method,
    dataset_name,
    output_dir,
):
    """Evaluate a model using a specific pattern."""
    print(f"Evaluating with {eval_pattern_name}")

    predictions, raw_outputs = classify_with_pattern(
        model,
        tokenizer,
        test_data["text"].tolist(),
        eval_pattern_name,
        temperature=1.0,
        top_p=0.9,
        batch_size=4,
    )

    y_true = test_data["label"].values
    y_pred = np.array(predictions)
    metrics = compute_metrics(y_true, y_pred)

    result = {
        "dataset": dataset_name,
        "model": model_key,
        "method": method,
        "train_pattern": train_pattern_name,
        "eval_pattern": eval_pattern_name,
        "same_pattern": train_pattern_name == eval_pattern_name,
        **metrics,
    }

    print(f"F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")

    detail_dir = os.path.join(output_dir, "detailed_predictions")
    os.makedirs(detail_dir, exist_ok=True)

    detailed = pd.DataFrame({
        "text": test_data["text"].values[:len(predictions)],
        "true_label": y_true,
        "predicted": y_pred,
        "raw_llm_output": raw_outputs,
        "correct": (y_true == y_pred),
        "train_pattern": train_pattern_name,
        "eval_pattern": eval_pattern_name,
    })

    filename = f"{model_key}_{method}_train-{train_pattern_name}_eval-{eval_pattern_name}.csv"
    detailed.to_csv(os.path.join(detail_dir, filename), index=False)

    return result


def run_cross_prompt_experiment(
    model_keys=None,
    train_patterns=None,
    eval_patterns=None,
    clear_master=False,
):
    """
    For each training pattern:
        1. Finetune model on that pattern
        2. Evaluate on ALL patterns
    """
    from ..data_loader import select_dataset, load_dataset
    from .finetune_config import FINETUNE_OUTPUT_DIR

    print("Cross-prompt experiment")

    models_to_train = {k: MODELS[k] for k in model_keys} if model_keys else MODELS
    train_patterns = train_patterns if train_patterns else list(PROMPT_PATTERNS.keys())
    eval_patterns = eval_patterns if eval_patterns else list(PROMPT_PATTERNS.keys())

    print(f"\nModels: {list(models_to_train.keys())}")
    print(f"Train patterns: {train_patterns}")
    print(f"Eval patterns: {eval_patterns}")
    print(f"Total evaluations per model: {len(train_patterns)} x {len(eval_patterns)} = {len(train_patterns) * len(eval_patterns)}")

    # Dataset selection
    dataset_config = select_dataset()
    dataset_name = dataset_config["name"]

    data = load_dataset(dataset_config)
    if data is None:
        print("Could not load dataset")
        return

    stats = get_data_statistics(data)
    print(f"\nDataset: {dataset_name}")
    print(f"  Total: {stats['total_samples']} samples")
    print(f"  Hate: {stats['hate_samples']} ({stats['hate_ratio']*100:.1f}%)")

    # Split: 80% for finetuning (train/val), 20% held-out test
    from sklearn.model_selection import train_test_split

    finetune_data, test_data = train_test_split(
        data,
        test_size=0.2,
        stratify=data["label"],
        random_state=SPLIT_CONFIG["seed"],
    )

    print(f"Finetune set: {len(finetune_data)} samples (will be split 80/20 for train/val)")
    print(f"Held-out test set: {len(test_data)} samples (used for ALL evaluations)")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        BASE_OUTPUT_DIR, "cross_prompt_results", f"{dataset_name}_{timestamp}"
    )

    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "training_logs"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "detailed_predictions"), exist_ok=True)

    print(f"Output: {results_dir}")

    # Run cross-prompt experiment
    all_results = []

    for model_key, model_name in models_to_train.items():
        print(f"\nModel: {model_key}")

        for train_pattern in train_patterns:
            print(f"\nTraining pattern: {train_pattern}")

            training_log_dir = os.path.join(
                results_dir, "training_logs", train_pattern, model_key
            )
            os.makedirs(training_log_dir, exist_ok=True)

            # Train on this pattern
            try:
                print(f"Preparing data with {train_pattern}")
                sft_data, _ = prepare_sft_data(
                    finetune_data,
                    model_key=model_key,
                    pattern_name=train_pattern,
                )

                print(f"Training SFT for {model_key} with {train_pattern}...")
                adapter_path = train_sft(
                    model_name=model_name,
                    dataset=sft_data,
                    output_dir=training_log_dir,
                    model_key=model_key,
                    dataset_name=dataset_name,
                    pattern_name=train_pattern,
                )
                print(f"Training complete. Adapter: {adapter_path}")

            except Exception as e:
                print(f"Training failed: {e}")
                import traceback
                traceback.print_exc()
                cleanup_gpu()
                continue

            # Evaluate on all patterns
            cleanup_gpu()
            time.sleep(3)

            # Evaluate base model
            print(f"\nEvaluating base model on all patterns")
            try:
                base_model, base_tokenizer = load_model_with_adapter(model_name, None)

                for eval_pattern in eval_patterns:
                    result = evaluate_with_pattern(
                        model=base_model,
                        tokenizer=base_tokenizer,
                        test_data=test_data,
                        eval_pattern_name=eval_pattern,
                        train_pattern_name=train_pattern,
                        model_key=model_key,
                        method="base",
                        dataset_name=dataset_name,
                        output_dir=results_dir,
                    )
                    all_results.append(result)

                del base_model, base_tokenizer
                cleanup_gpu()

            except Exception as e:
                print(f"Base evaluation failed: {e}")
                import traceback
                traceback.print_exc()

            # Evaluate SFT model
            print(f"\nEvaluating SFT model (trained on {train_pattern}) on all patterns")
            try:
                sft_model, sft_tokenizer = load_model_with_adapter(model_name, adapter_path)

                for eval_pattern in eval_patterns:
                    result = evaluate_with_pattern(
                        model=sft_model,
                        tokenizer=sft_tokenizer,
                        test_data=test_data,
                        eval_pattern_name=eval_pattern,
                        train_pattern_name=train_pattern,
                        model_key=model_key,
                        method="sft",
                        dataset_name=dataset_name,
                        output_dir=results_dir,
                    )
                    all_results.append(result)

                del sft_model, sft_tokenizer
                cleanup_gpu()

            except Exception as e:
                print(f"SFT evaluation failed: {e}")
                import traceback
                traceback.print_exc()

            # Save intermediate results
            if all_results:
                interim_df = pd.DataFrame(all_results)
                interim_path = os.path.join(results_dir, "metrics", "cross_prompt_results_interim.csv")
                interim_df.to_csv(interim_path, index=False)
                print(f"Interim results saved: {interim_path}")

            cleanup_gpu()
            time.sleep(2)

    # Save final results
    print("\nSaving final results")

    if all_results:
        results_df = pd.DataFrame(all_results)

        results_path = os.path.join(results_dir, "metrics", "cross_prompt_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Full results saved: {results_path}")

        create_cross_prompt_heatmap(results_df, results_dir)

        print_cross_prompt_summary(results_df)

    print(f"Results: {results_dir}")

    return results_df


def create_cross_prompt_heatmap(results_df, output_dir):
    """Heatmap showing F1 scores for train_pattern x eval_pattern matrix."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sft_df = results_df[results_df['method'] == 'sft']

    if sft_df.empty:
        print("No SFT results to plot")
        return

    for model in sft_df['model'].unique():
        model_df = sft_df[sft_df['model'] == model]

        pivot = model_df.pivot_table(
            values='f1',
            index='train_pattern',
            columns='eval_pattern',
            aggfunc='mean'
        )

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={'label': 'F1 Score'}
        )

        ax.set_xlabel('Evaluation Pattern', fontsize=12)
        ax.set_ylabel('Training Pattern', fontsize=12)
        ax.set_title(f'Cross-Prompt F1 Scores: {model.upper()} (SFT)\n'
                     f'Diagonal = Same pattern for train & eval',
                     fontsize=14, fontweight='bold')

        # Highlight diagonal
        for i in range(min(len(pivot.index), len(pivot.columns))):
            if pivot.index[i] in pivot.columns:
                j = list(pivot.columns).index(pivot.index[i])
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                           edgecolor='blue', linewidth=3))

        plt.tight_layout()

        plot_path = os.path.join(output_dir, "plots", f"cross_prompt_heatmap_{model}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Heatmap saved: {plot_path}")

    create_base_vs_sft_comparison(results_df, output_dir)


def create_base_vs_sft_comparison(results_df, output_dir):
    """Bar chart comparing Base vs SFT for same vs different prompt scenarios."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    same_pattern = results_df[results_df['same_pattern'] == True]
    same_grouped = same_pattern.groupby(['model', 'method'])['f1'].mean().unstack()

    if not same_grouped.empty:
        same_grouped.plot(kind='bar', ax=axes[0], color=['#95A5A6', '#27AE60'])
        axes[0].set_title('Same Pattern (Train = Eval)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('F1 Score')
        axes[0].set_xlabel('Model')
        axes[0].legend(title='Method')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)

    diff_pattern = results_df[results_df['same_pattern'] == False]
    diff_grouped = diff_pattern.groupby(['model', 'method'])['f1'].mean().unstack()

    if not diff_grouped.empty:
        diff_grouped.plot(kind='bar', ax=axes[1], color=['#95A5A6', '#E74C3C'])
        axes[1].set_title('Different Pattern (Train != Eval)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_xlabel('Model')
        axes[1].legend(title='Method')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle('Cross-Prompt Generalization: Does Finetuning Transfer?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "plots", "base_vs_sft_generalization.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()


def print_cross_prompt_summary(results_df):

    sft_df = results_df[results_df['method'] == 'sft']

    same = sft_df[sft_df['same_pattern'] == True]['f1'].mean()
    diff = sft_df[sft_df['same_pattern'] == False]['f1'].mean()

    print(f"\nSFT Performance:")
    print(f"Same pattern (train=eval): F1 = {same:.3f}")
    print(f"Different pattern (train!=eval): F1 = {diff:.3f}")
    print(f"Generalization gap: {same - diff:+.3f}")

    if not sft_df.empty:
        gen_by_train = sft_df[sft_df['same_pattern'] == False].groupby('train_pattern')['f1'].mean()
        if not gen_by_train.empty:
            best_train = gen_by_train.idxmax()
            print(f"\nBest training pattern for generalization: {best_train} (F1={gen_by_train[best_train]:.3f})")

    if not sft_df.empty:
        by_eval = sft_df.groupby('eval_pattern')['f1'].mean()
        if not by_eval.empty:
            best_eval = by_eval.idxmax()
            print(f"Most robust evaluation pattern: {best_eval} (F1={by_eval[best_eval]:.3f})")


# Entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-prompt experiment")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to use (e.g., llama qwen)"
    )
    parser.add_argument(
        "--train-patterns",
        type=str,
        nargs="+",
        default=None,
        help="Patterns to train on (default: all 11)"
    )
    parser.add_argument(
        "--eval-patterns",
        type=str,
        nargs="+",
        default=None,
        help="Patterns to evaluate on (default: all 11)"
    )

    args = parser.parse_args()

    run_cross_prompt_experiment(
        model_keys=args.models,
        train_patterns=args.train_patterns,
        eval_patterns=args.eval_patterns,
    )
