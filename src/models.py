"""
Model setup, inference, and response parsing
"""

import os
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import PROMPT_PATTERNS, DEVICE


def setup_model(model_name):
    """Setup model with proper GPU allocation."""
    print(f"\nLoading model: {model_name}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    try:
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": DEVICE,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if "gemma" in model_name.lower():
            load_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None, None


def parse_response(text, pattern_name):
    """Parse model response for binary classification."""
    text = text.strip().lower()

    # Strip echoed template fragments (Gemma echo issue)
    if "svar:" in text:
        text = text.split("svar:")[-1].strip()

    pattern = PROMPT_PATTERNS[pattern_name]

    if pattern_name == "choice_qa":
        if "a" in text[:10]:
            return 1
        return 0

    if pattern_name == "cloze":
        if "hadefuldt" in text:
            return 1
        return 0

    if text.startswith(pattern["positive"]):
        return 1
    elif text.startswith(pattern["negative"]):
        return 0
    elif pattern["positive"] in text[:50]:
        return 1

    return 0


def parse_verbal_confidence(text):
    """Parse confidence from verbal response."""
    import re
    text = text.lower()
    match = re.search(r"(\d+)\s*%", text)
    if match:
        conf = int(match.group(1)) / 100.0
        conf = max(0.0, min(1.0, conf))
        answer = 1 if "ja" in text[:20] else 0
        return answer, conf
    answer = 1 if "ja" in text[:20] else 0
    return answer, 0.5


def classify_with_pattern(
    model, tokenizer, texts, pattern_name, temperature=1.0, top_p=0.9, batch_size=8
):
    """Classify texts with specified prompt pattern using true batching."""
    results = []
    raw_outputs = []
    pattern = PROMPT_PATTERNS[pattern_name]

    if len(texts) > 1500:
        batch_size = 4

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Pattern: {pattern_name}"):
        batch_texts = texts[i : i + batch_size]
        batch_prompts = [pattern["template"].format(text=text) for text in batch_texts]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)

        input_lengths = inputs["attention_mask"].sum(dim=1)

        with torch.no_grad():
            if temperature > 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        for j in range(len(batch_prompts)):
            input_len = input_lengths[j].item()
            response = tokenizer.decode(
                outputs[j][input_len:],
                skip_special_tokens=True
            )

            raw_outputs.append(response)
            pred = parse_response(response, pattern_name)
            results.append(pred)

    return results, raw_outputs
