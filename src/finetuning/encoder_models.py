"""
Encoder Model Support (BERT, DaBERT, ScandiBERT)
=================================================

Three evaluation methods:

1. base_mlm:      MLM prompting — insert [MASK], compare P(ja) vs P(nej)
                   Works with all 11 patterns (zero-shot only — BERT cannot
                   do in-context learning; 8-shot examples dominate the
                   512-token window causing all patterns to collapse)
2. base_cls:      Untrained classification head (random baseline)
3. sft_lora_cls:  LoRA fine-tuned classification head

Multi-token handling (e.g., cloze with "hadefuldt"/"neutralt"):
  - If a candidate word splits into N subwords, insert N [MASK] tokens
  - Run separate forward passes for each candidate's mask count
  - Compute joint probability = product of per-position subword probs
  - Length-normalize to remove bias from different candidate lengths
  - Compare joint probabilities to classify
  - Falls back to single-token "ja"/"nej" if multi-token degenerates

Reproducibility: seed=42, LoRA r=16, alpha=32
"""

import os
import gc
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Any, Dict, List, Optional
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict

from ..config import ENCODER_MODELS, PROMPT_PATTERNS, DEVICE
from ..metrics import compute_metrics
from .finetune_config import SPLIT_CONFIG
# SEED (matches finetune_config.py)

SEED = SPLIT_CONFIG["seed"]  # 42
# ENCODER LORA CONFIGURATION
# Mirrors the causal LoRA config (r=16, alpha=32) but targets
# BERT attention layers instead of causal LM projections

ENCODER_LORA_R = 16
ENCODER_LORA_ALPHA = 32
ENCODER_LORA_DROPOUT = 0.05
# TRAINING CONFIGURATION

ENCODER_TRAIN_CONFIG = {
    "num_epochs": 10,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "early_stopping_patience": 3,
    "max_length": 512,
}

MAX_SFT_RETRIES = 5
# ZERO-SHOT PATTERNS FOR MLM

# BERT uses zero-shot only. We strip the 8-shot examples from PROMPT_PATTERNS
# and build zero-shot versions dynamically (just header + test prefix).

_ZERO_SHOT_PATTERNS = {
    "vanilla_qa": {
        "template": "Givet udsagnet, er dette udsagn hadefuld tale? Angiv venligst dit svar. Svaret skal indeholde Ja eller Nej.\n\nNu din tur:\nUdsagn: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "choice_qa": {
        "template": "Givet udsagnet, er dette udsagn hadefuld tale? A: Ja. B: Nej. Angiv venligst dit valg (A eller B).\n\nNu din tur:\nUdsagn: {text}\nSvar:",
        "positive": "a",
        "negative": "b",
    },
    "cloze": {
        "template": "Givet udsagnet, er dette udsagn hadefuldt eller neutralt? Forudsig venligst [MASK] token, du kan kun svare hadefuldt eller neutralt.\n\nNu din tur:\nUdsagn: {text}\nSvar: Det er et [MASK] udsagn.",
        "positive": "hadefuldt",
        "negative": "neutralt",
    },
    "cot": {
        "template": "Er dette udsagn hadefuld tale? Dit svar skal kun indeholde Ja eller Nej, og giv derefter din forklaring trin for trin.\n\nNu din tur:\nUdsagn: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "target": {
        "template": "Givet udsagnet, er dette udsagn hadefuld tale? Dit svar skal kun indeholde Ja eller Nej, og identificer derefter målgruppen eller individet.\n\nNu din tur:\nUdsagn: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "illocutionary": {
        "template": "Analyser hvilken sproghandling dette udsagn udfører. Er det en påstand, en opfordring, et løfte, et udtryk for følelser, eller en erklæring? Hadefuld tale fungerer typisk som udtryk for foragt eller opfordring til skade. Givet udsagnet, er dette udsagn hadefuld tale? Svaret skal indeholde ja eller nej.\n\nNu din tur:\nUdsagn: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "functional": {
        "template": "Overvej tekstens sociale funktion: informerer den, underholder, opbygger fællesskab, udelukker andre, eller etablerer dominans? Tekster der udelukker eller dominerer er ofte hadefulde. Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.\n\nNu din tur:\nTekst: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "definition": {
        "template": "Hadefuld tale angriber en person eller gruppe baseret på race, religion, køn, seksualitet eller handicap, inkl. trusler og dehumanisering.\n\nNu din tur:\nTekst: {text}\nEr teksten hadefuld? Svaret skal indeholde ja eller nej.\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "victim_perspective": {
        "template": "Forestil dig du er den person eller gruppe teksten handler om. Ville du føle dig angrebet eller nedgjort? Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.\n\nNu din tur:\nTekst: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "expert_moderator": {
        "template": "Du er en erfaren moderator på sociale medier. Vurder om denne tekst overtræder retningslinjer mod hadefuld tale. Givet teksten, er dette hadefuld tale? Svaret skal indeholde ja eller nej.\n\nNu din tur:\nTekst: {text}\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
    "rules": {
        "template": "Hadefuld tale kræver: (1) angreb på en gruppe baseret på race, religion, køn eller nationalitet, OG (2) trusler, nedværdigende sprog, eller formål om at udelukke/dominere.\n\nNu din tur:\nTekst: {text}\nOpfylder teksten begge krav? Svaret skal indeholde Ja eller Nej.\nSvar:",
        "positive": "ja",
        "negative": "nej",
    },
}

# GPU CLEANUP 
def _cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# ENCODER MODEL SETUP
def setup_encoder(model_name, training=True, num_labels=2):
    """Load encoder model for sequence classification."""
    print(f"\nLoading encoder: {model_name}")
    _cleanup_gpu()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, trust_remote_code=True,
    )
    model = model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params:,} parameters")
    return model, tokenizer
def _setup_mlm(model_name):
    """Load encoder model for Masked Language Modeling."""
    print(f"\nLoading MLM encoder: {model_name}")
    _cleanup_gpu()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

    model = model.to(DEVICE)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MLM model loaded: {n_params:,} parameters")
    return model, tokenizer

# MULTI-TOKEN MLM SCORING
def _score_multi_token_candidate(
    model, tokenizer, prompts_with_placeholder, candidate_ids,
    mask_token, mask_token_id, batch_size=16,
):
    """
    Score a multi-token candidate across a batch of prompts.
    Uses length-normalized log probability (Salazar et al. 2020).
    """
    n_masks = len(candidate_ids)
    mask_string = " ".join([mask_token] * n_masks)
    prompts = [p.replace("<<ANSWER>>", mask_string) for p in prompts_with_placeholder]

    all_log_probs = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        input_ids = inputs["input_ids"]

        for j in range(len(batch)):
            mask_positions = (input_ids[j] == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) < n_masks:
                all_log_probs.append(float("-inf"))
                continue

            answer_positions = mask_positions[-n_masks:]
            log_prob = 0.0
            for k, (pos, token_id) in enumerate(zip(answer_positions, candidate_ids)):
                pos_logits = logits[j, pos.item()]
                pos_probs = pos_logits.softmax(dim=-1)
                p = pos_probs[token_id].item()
                log_prob += np.log(max(p, 1e-10))

            all_log_probs.append(log_prob / n_masks)

    return all_log_probs

# MLM EVALUATION
def evaluate_encoder_mlm(
    model_name: str,
    test_df: pd.DataFrame,
    pattern_names: List[str] = None,
    batch_size: int = 16,
) -> List[Dict]:
    """
    Evaluate encoder model using MLM prompting (zero-shot only).
    
    Three paths:
      A. Single-token candidates (ja/nej): one [MASK], compare probs
      B. Multi-token candidates (hadefuldt/neutralt): separate forward passes
         with length-normalized joint log-probs. Falls back to ja/nej if degenerate.
      C. choice_qa remapping: "a"/"b" -> "ja"/"nej"
    """
    pattern_names = pattern_names or list(_ZERO_SHOT_PATTERNS.keys())

    model, tokenizer = _setup_mlm(model_name)

    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.mask_token_id

    if mask_token is None or mask_token_id is None:
        print(f"  ERROR: {model_name} has no mask token — cannot do MLM")
        del model, tokenizer
        _cleanup_gpu()
        return []

    print(f"  Mask token: '{mask_token}' (id={mask_token_id})")

    texts = test_df["text"].tolist()
    labels = test_df["label"].values
    results = []

    for pattern_name in pattern_names:
        if pattern_name not in _ZERO_SHOT_PATTERNS:
            print(f"  SKIP: {pattern_name} not in zero-shot patterns")
            continue

        pattern = _ZERO_SHOT_PATTERNS[pattern_name]
        pos_word = pattern["positive"]
        neg_word = pattern["negative"]

        # MLM remapping for choice_qa
        if pattern_name == "choice_qa":
            pos_word = "ja"
            neg_word = "nej"
            print(f"    [MLM remap] choice_qa: 'a'/'b' -> 'ja'/'nej'")

        pos_ids = tokenizer.encode(pos_word, add_special_tokens=False)
        neg_ids = tokenizer.encode(neg_word, add_special_tokens=False)
        pos_cap_ids = tokenizer.encode(pos_word.capitalize(), add_special_tokens=False)
        neg_cap_ids = tokenizer.encode(neg_word.capitalize(), add_special_tokens=False)

        # Space-prefixed versions for SentencePiece/RoBERTa-based models (e.g. ScandiBERT).
        # In SentencePiece, "▁ja" (after a space) is a different token than "ja".
        # Non-cloze patterns append " " + mask_token, so the model predicts a
        # space-prefixed token — we must include those IDs in the comparison.
        pos_ids_sp = tokenizer.encode(" " + pos_word, add_special_tokens=False)
        neg_ids_sp = tokenizer.encode(" " + neg_word, add_special_tokens=False)
        pos_cap_ids_sp = tokenizer.encode(" " + pos_word.capitalize(), add_special_tokens=False)
        neg_cap_ids_sp = tokenizer.encode(" " + neg_word.capitalize(), add_special_tokens=False)

        is_multi_token = len(pos_ids) > 1 or len(neg_ids) > 1

        print(
            f"\n  MLM zero_shot/{pattern_name}: "
            f"pos='{pos_word}'({len(pos_ids)} subwords) ids={pos_ids}, "
            f"neg='{neg_word}'({len(neg_ids)} subwords) ids={neg_ids}"
            f"\n    space-prefixed: pos_sp={pos_ids_sp}, neg_sp={neg_ids_sp}"
            f"{' [MULTI-TOKEN]' if is_multi_token else ''}"
        )

        # ---- PATH A: Single-token ----
        if not is_multi_token:
            pos_id, neg_id = pos_ids[0], neg_ids[0]
            pos_cap_id, neg_cap_id = pos_cap_ids[0], neg_cap_ids[0]

            predictions = []
            truncated_count = 0

            for i in tqdm(range(0, len(texts), batch_size),
                          desc=f"  MLM {pattern_name}"):
                batch_texts = texts[i : i + batch_size]
                prompts = []
                for text in batch_texts:
                    prompt = pattern["template"].format(text=text)
                    if pattern_name == "cloze":
                        prompt = prompt.replace("[MASK]", mask_token)
                    else:
                        prompt = prompt.rstrip() + " " + mask_token
                    prompts.append(prompt)

                inputs = tokenizer(
                    prompts, return_tensors="pt",
                    padding=True, truncation=True, max_length=512,
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(**inputs).logits

                input_ids = inputs["input_ids"]
                for j in range(len(prompts)):
                    mask_positions = (input_ids[j] == mask_token_id).nonzero(as_tuple=True)[0]
                    if len(mask_positions) == 0:
                        predictions.append(0)
                        truncated_count += 1
                        continue
                    mask_pos = mask_positions[-1].item()
                    probs = logits[j, mask_pos].softmax(dim=-1)
                    p_pos = probs[pos_id].item() + probs[pos_cap_id].item()
                    p_neg = probs[neg_id].item() + probs[neg_cap_id].item()
                    # Add space-prefixed token probs (fixes ScandiBERT/SentencePiece models
                    # where "▁ja" is a different token than "ja")
                    if len(pos_ids_sp) == 1:
                        p_pos += probs[pos_ids_sp[0]].item()
                    if len(pos_cap_ids_sp) == 1:
                        p_pos += probs[pos_cap_ids_sp[0]].item()
                    if len(neg_ids_sp) == 1:
                        p_neg += probs[neg_ids_sp[0]].item()
                    if len(neg_cap_ids_sp) == 1:
                        p_neg += probs[neg_cap_ids_sp[0]].item()
                    # Debug first 3 samples of first batch
                    if i == 0 and j < 3:
                        print(f"    [DEBUG sample {j}] p_pos={p_pos:.6f} p_neg={p_neg:.6f} → pred={1 if p_pos > p_neg else 0}")
                    predictions.append(1 if p_pos > p_neg else 0)

            if truncated_count > 0:
                print(f"    WARNING: {truncated_count}/{len(texts)} truncated")

        # ---- PATH B: Multi-token ----
        else:
            prompts_with_placeholder = []
            for text in texts:
                prompt = pattern["template"].format(text=text)
                if pattern_name == "cloze":
                    prompt = prompt.replace("[MASK]", "<<ANSWER>>")
                else:
                    prompt = prompt.rstrip() + " <<ANSWER>>"
                prompts_with_placeholder.append(prompt)

            pos_scores = _score_multi_token_candidate(
                model, tokenizer, prompts_with_placeholder,
                pos_ids, mask_token, mask_token_id, batch_size,
            )
            pos_cap_scores = _score_multi_token_candidate(
                model, tokenizer, prompts_with_placeholder,
                pos_cap_ids, mask_token, mask_token_id, batch_size,
            )
            neg_scores = _score_multi_token_candidate(
                model, tokenizer, prompts_with_placeholder,
                neg_ids, mask_token, mask_token_id, batch_size,
            )
            neg_cap_scores = _score_multi_token_candidate(
                model, tokenizer, prompts_with_placeholder,
                neg_cap_ids, mask_token, mask_token_id, batch_size,
            )

            predictions = []
            truncated_count = 0
            for k in range(len(texts)):
                p_pos = max(pos_scores[k], pos_cap_scores[k])
                p_neg = max(neg_scores[k], neg_cap_scores[k])
                if p_pos == float("-inf") and p_neg == float("-inf"):
                    predictions.append(0)
                    truncated_count += 1
                else:
                    predictions.append(1 if p_pos > p_neg else 0)

            if truncated_count > 0:
                print(f"WARNING: {truncated_count}/{len(texts)} truncated")

            # FALLBACK: if degenerate, retry with single-token ja/nej
            n_pos = sum(predictions)
            if n_pos == 0 or n_pos == len(predictions):
                fb_pos_ids = tokenizer.encode("ja", add_special_tokens=False)
                fb_neg_ids = tokenizer.encode("nej", add_special_tokens=False)

                if len(fb_pos_ids) == 1 and len(fb_neg_ids) == 1:
                    print(f"    FALLBACK: multi-token degenerate → 'ja'/'nej'")
                    fb_pos_id = fb_pos_ids[0]
                    fb_neg_id = fb_neg_ids[0]
                    fb_pos_cap_id = tokenizer.encode("Ja", add_special_tokens=False)[0]
                    fb_neg_cap_id = tokenizer.encode("Nej", add_special_tokens=False)[0]

                    predictions = []
                    for i_fb in tqdm(range(0, len(texts), batch_size),
                                     desc=f"  FALLBACK {pattern_name}"):
                        batch_texts = texts[i_fb : i_fb + batch_size]
                        fb_prompts = []
                        for text in batch_texts:
                            prompt = pattern["template"].format(text=text)
                            if pattern_name == "cloze":
                                prompt = prompt.replace("[MASK]", mask_token)
                            else:
                                prompt = prompt.rstrip() + " " + mask_token
                            fb_prompts.append(prompt)

                        fb_inputs = tokenizer(
                            fb_prompts, return_tensors="pt",
                            padding=True, truncation=True, max_length=512,
                        )
                        fb_inputs = {kk: vv.to(DEVICE) for kk, vv in fb_inputs.items()}

                        with torch.no_grad():
                            fb_logits = model(**fb_inputs).logits

                        fb_input_ids = fb_inputs["input_ids"]
                        for j_fb in range(len(fb_prompts)):
                            fb_mask_pos = (fb_input_ids[j_fb] == mask_token_id).nonzero(as_tuple=True)[0]
                            if len(fb_mask_pos) == 0:
                                predictions.append(0)
                                continue
                            pos_idx = fb_mask_pos[-1].item()
                            fb_probs = fb_logits[j_fb, pos_idx].softmax(dim=-1)
                            p_p = fb_probs[fb_pos_id].item() + fb_probs[fb_pos_cap_id].item()
                            p_n = fb_probs[fb_neg_id].item() + fb_probs[fb_neg_cap_id].item()
                            predictions.append(1 if p_p > p_n else 0)

                    print(f"FALLBACK result: {sum(predictions)}/{len(predictions)} positive")

        # ---- Metrics ----
        metrics = compute_metrics(labels, predictions)
        results.append({
            "shot_type": "zero_shot",
            "pattern": pattern_name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "n_samples": metrics["n_samples"],
        })
        print(f"F1={metrics['f1']:.3f} P={metrics['precision']:.3f} "
              f"R={metrics['recall']:.3f} Acc={metrics['accuracy']:.3f}")

    del model, tokenizer
    _cleanup_gpu()
    print(f"\n  MLM complete: {len(results)} evaluations")
    return results

# CLASSIFICATION HEAD — data prep, training, evaluation
def prepare_encoder_data(train_df, val_df, test_df, tokenizer, max_length=512):
    """Format pre-split data for encoder classification."""

    print(f"  Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    def tokenize(examples):
        return tokenizer(
            examples["text"], padding="max_length",
            truncation=True, max_length=max_length,
        )

    def to_dataset(split_df):
        ds = Dataset.from_dict({
            "text": split_df["text"].tolist(),
            "label": split_df["label"].tolist(),
        })
        ds = ds.map(tokenize, batched=True)
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return ds

    dataset = DatasetDict({
        "train": to_dataset(train_df),
        "validation": to_dataset(val_df),
        "test": to_dataset(test_df),
    })
    return dataset, test_df
def train_encoder(model_name, train_df, val_df, test_df, model_key, output_dir, use_lora=True,
                  num_epochs=None, batch_size=None, lr=None, seed=SEED):
    """Fine-tune encoder with classification head (+ optional LoRA).
    Computes class weights for imbalanced data."""
    num_epochs = num_epochs or ENCODER_TRAIN_CONFIG["num_epochs"]
    batch_size = batch_size or ENCODER_TRAIN_CONFIG["batch_size"]
    lr = lr or ENCODER_TRAIN_CONFIG["learning_rate"]

    print(f"\nEncoder training: {model_key}")
    print(f"  LoRA: r={ENCODER_LORA_R}, alpha={ENCODER_LORA_ALPHA}, seed={seed}")

    model, tokenizer = setup_encoder(model_name, training=True)
    dataset, test_df = prepare_encoder_data(train_df, val_df, test_df, tokenizer)

    # Class weights for imbalanced data
    train_labels = np.array(dataset["train"]["label"])
    classes = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    hate_ratio = train_labels.mean()
    is_imbalanced = hate_ratio < 0.3 or hate_ratio > 0.7
    print(f"Class distribution: {hate_ratio * 100:.1f}% hate")
    print(f"Class weights: NOT={class_weights[0]:.3f}, HATE={class_weights[1]:.3f}"
          f"{'[IMBALANCED]' if is_imbalanced else ''}")

    if use_lora:
        # Auto-detect LoRA target modules for different BERT architectures
        model_module_names = [name for name, _ in model.named_modules()]
        if any("query" in n for n in model_module_names):
            target_modules = ["query", "key", "value", "dense"]
        elif any("q_proj" in n for n in model_module_names):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
        else:
            target_modules = ["query", "key", "value", "dense"]

        print(f"LoRA target modules: {target_modules}")
        lora_config = LoraConfig(
            r=ENCODER_LORA_R, lora_alpha=ENCODER_LORA_ALPHA,
            lora_dropout=ENCODER_LORA_DROPOUT,
            target_modules=target_modules,
            bias="none", task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  LoRA: {trainable:,}/{total:,} trainable ({100 * trainable / total:.2f}%)")

    save_dir = os.path.join(output_dir, model_key, "encoder_sft")
    os.makedirs(save_dir, exist_ok=True)

    def _trainer_compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        return {"f1": f1, "precision": p, "recall": r}

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=ENCODER_TRAIN_CONFIG["weight_decay"],
        warmup_ratio=ENCODER_TRAIN_CONFIG["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        logging_steps=20,
        report_to="none",
        seed=seed,
    )

    # Custom trainer with weighted loss for imbalanced data
    class WeightedTrainer(Trainer):
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            w = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=w)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model, args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=_trainer_compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=ENCODER_TRAIN_CONFIG["early_stopping_patience"]
        )],
    )
    trainer.train()

    best_path = os.path.join(save_dir, "best_model")
    model.save_pretrained(best_path)
    tokenizer.save_pretrained(best_path)

    del model, trainer
    _cleanup_gpu()

    print(f"  Saved: {best_path}")
    return best_path, test_df
def evaluate_encoder(model_name, test_df, adapter_path=None, batch_size=32):
    """Evaluate encoder on test set using classification head."""
    if adapter_path and os.path.exists(adapter_path):
        model = AutoModelForSequenceClassification.from_pretrained(adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        print(f"  Loaded fine-tuned encoder from {adapter_path}")
    else:
        model, tokenizer = setup_encoder(model_name, training=False)
        print(f"  Using untrained encoder (zero-shot baseline)")

    model = model.to(DEVICE)
    model.eval()

    texts = test_df["text"].tolist()
    labels = test_df["label"].values
    predictions = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoder eval"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            predictions.extend(preds.tolist())

    metrics = compute_metrics(labels, predictions)

    del model, tokenizer
    _cleanup_gpu()

    return metrics
