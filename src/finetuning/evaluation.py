"""
Evaluation Module for Fine-Tuned Models
- Instruct models (Base refers to no fine-tuning)
- LoRA + SFT fine-tuned models
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from peft import PeftModel

from ..config import (
    PROMPT_PATTERNS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from ..models import classify_with_pattern
from ..metrics import compute_metrics

from .trainers import setup_base_model, cleanup_gpu

from .finetune_config import (
    RUN_FINETUNED_PATTERN_ANALYSIS,
)


def save_detailed_predictions(data, predictions, raw_outputs, model_key, method, pattern_name, dataset_name, output_dir):

    import os

    y_true = data["label"].values[:len(predictions)]
    y_pred = np.array(predictions)

    detailed = pd.DataFrame({
        "text": data["text"].values[:len(predictions)],
        "true_label": y_true,
        "predicted": y_pred,
        "raw_llm_output": raw_outputs,
        "correct": (y_true == y_pred),
    })

    detailed["error_type"] = "Correct"
    detailed.loc[(y_true == 1) & (y_pred == 0), "error_type"] = "FN - Missed Hate"
    detailed.loc[(y_true == 0) & (y_pred == 1), "error_type"] = "FP - False Alarm"

    pattern = PROMPT_PATTERNS[pattern_name]

    def check_parse_quality(raw):
        raw_lower = raw.strip().lower()
        pos = pattern["positive"]
        neg = pattern["negative"]
        if raw_lower.startswith(pos) or raw_lower.startswith(neg):
            return "Clean"
        elif pos in raw_lower[:50] or neg in raw_lower[:50]:
            return "Parseable"
        else:
            return "May be wrong"

    detailed["parse_quality"] = detailed["raw_llm_output"].apply(check_parse_quality)

    detailed = detailed.sort_values(["correct", "parse_quality"], ascending=[True, False])

    detail_dir = os.path.join(output_dir, "detailed_predictions")
    os.makedirs(detail_dir, exist_ok=True)

    filename = f"{model_key}_{method}_{pattern_name}_detailed.csv"
    filepath = os.path.join(detail_dir, filename)
    detailed.to_csv(filepath, index=False)

    fn_count = (detailed["error_type"].str.contains("FN")).sum()
    fp_count = (detailed["error_type"].str.contains("FP")).sum()
    check_count = (detailed["parse_quality"] == "May be wrong").sum()

    print(f"Saved {filepath}")
    print(f"Errors: {fn_count} FN, {fp_count} FP | Parse issues: {check_count}")

    return detailed


def load_model_with_adapter(
    base_model_name: str, adapter_path: Optional[str] = None
) -> Tuple[Any, Any]:

    model, tokenizer = setup_base_model(base_model_name, training=False)

    if adapter_path is not None and os.path.exists(adapter_path):
        print(f"Loading adapter from: {adapter_path}")

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

        print("Adapter merged successfully")

    return model, tokenizer


def evaluate_single_model(
    model: Any,
    tokenizer: Any,
    data: pd.DataFrame,
    model_key: str,
    method: str,
    dataset_name: str,
    output_dir: str,
    pattern_name: str = "vanilla_qa",
) -> Dict[str, List[Dict]]:
    results = {
        "pattern": [],
    }

    if RUN_FINETUNED_PATTERN_ANALYSIS:
        print(f"\nPattern Analysis ({method}): {pattern_name}")

        try:
            predictions, raw_outputs = classify_with_pattern(
                model,
                tokenizer,
                data["text"].tolist(),
                pattern_name,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                batch_size=4,
            )

            y_true = data["label"].values
            y_pred = np.array(predictions)
            metrics = compute_metrics(y_true, y_pred)

            results["pattern"].append(
                {
                    "dataset": dataset_name,
                    "model": model_key,
                    "method": method,
                    "pattern": pattern_name,
                    **metrics,
                }
            )

            print(
                f"    F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, "
                f"R={metrics['recall']:.3f}"
            )

            save_detailed_predictions(
                data, predictions, raw_outputs, model_key, method,
                pattern_name, dataset_name, output_dir
            )

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    return results


def evaluate_all_methods(
    model_key: str,
    model_name: str,
    data: pd.DataFrame,
    dataset_name: str,
    adapter_paths: Dict[str, Optional[str]],
    output_dir: str,
    pattern_name: str = "vanilla_qa",
) -> Dict[str, List[Dict]]:
    """
    Evaluate the base model (no fine-tuning) and SFT fine-tuned model.
    """
    all_results = {
        "pattern": [],
    }

    methods = ["base"] + list(adapter_paths.keys())

    for method in methods:
        print(f"Loading model: {model_key} ({method})")

        adapter_path = adapter_paths.get(method) if method != "base" else None

        model, tokenizer = load_model_with_adapter(model_name, adapter_path)

        results = evaluate_single_model(
            model=model,
            tokenizer=tokenizer,
            data=data,
            model_key=model_key,
            method=method,
            dataset_name=dataset_name,
            output_dir=output_dir,
            pattern_name=pattern_name,
        )

        for key in all_results:
            all_results[key].extend(results[key])

        del model, tokenizer
        cleanup_gpu()

    return all_results


def generate_comparison_report(
    results: Dict[str, pd.DataFrame], dataset_name: str, output_dir: str
) -> str:

    report_lines = []
    report_lines.append(f"Dataset: {dataset_name}")

    if "pattern" in results and not results["pattern"].empty:
        pattern_df = results["pattern"]

        method_f1 = pattern_df.groupby("method")["f1"].mean()
        best_method = method_f1.idxmax()

        report_lines.append(
            f"\nBest method by F1: {best_method.upper()} "
            f"(F1={method_f1[best_method]:.3f})"
        )
        for method in method_f1.index:
            report_lines.append(f"  {method.upper():8s}: F1={method_f1[method]:.3f}")

        if "base" in method_f1.index:
            base_f1 = method_f1["base"]
            report_lines.append("\nImprovement over base:")
            for method in method_f1.index:
                if method != "base":
                    improvement = (method_f1[method] - base_f1) / base_f1 * 100
                    report_lines.append(f"  {method.upper():8s}: {improvement:+.1f}%")
