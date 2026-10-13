"""
Load the merged fine-tuned model and classify a few customer-support messages.

Every generation is passed through `labels.match_label()`, the same rule
`train.py` uses at evaluation time, so anything outside the 27-label vocabulary
is returned as "__other__" instead of an invented label.

Prerequisite: run `python train.py` first so `./merged_model/` exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from labels import OTHER, match_label

MODEL_DIR = Path("./merged_model")
MAX_NEW_TOKENS = 16

SYSTEM_PROMPT = (
    "You are an intent classifier for customer-support messages. "
    "Respond with exactly one intent label from the allowed set. "
    "Output only the label, nothing else."
)

SAMPLE_MESSAGES = [
    "I need to cancel my subscription right now",
    "Can you tell me where my package is at the moment?",
    "How do I change the shipping address on an order I already placed?",
    "What is your refund policy for returned items?",
    "I can't log into my account anymore, keeps saying wrong password",
    "Please give me the phone number for customer support",
    "I want to leave some feedback about my recent experience",
]


def classify(model, tokenizer, message: str) -> str:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": message},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[0][enc["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def classify_label(model, tokenizer, message: str) -> tuple[str, str]:
    """Return (label, raw_generation); label is a valid intent or '__other__'."""
    raw = classify(model, tokenizer, message)
    return match_label(raw), raw


def main():
    if not MODEL_DIR.exists():
        sys.exit(f"Merged model not found at {MODEL_DIR}. Run `python train.py` first.")

    print(f"Loading merged model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"  loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB\n")

    print("-" * 70)
    print(f"{'predicted intent':<32}  message")
    print("-" * 70)
    messages = SAMPLE_MESSAGES if len(sys.argv) <= 1 else sys.argv[1:]
    for msg in messages:
        label, raw = classify_label(model, tokenizer, msg)
        note = f"  (raw generation: {raw!r})" if label == OTHER else ""
        print(f"{label:<32}  {msg}{note}")
    print("-" * 70)


if __name__ == "__main__":
    main()
