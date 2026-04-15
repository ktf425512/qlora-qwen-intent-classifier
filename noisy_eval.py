"""
Hand-crafted noisy test set.

The Bitext benchmark used in `train.py` is procedurally generated and very
clean — which is why the fine-tuned model hits 100 % macro F1 on held-out
val. This script probes how that accuracy behaves on realistic noise:
typos, abbreviations, slang, frustrated tone, negation, and mixed intents.

Loads ./merged_model/ and evaluates 15 hand-written messages spanning six
noise categories. Writes the full case-by-case result to
outputs/noisy_eval.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path("./merged_model")
OUT_JSON  = Path("./outputs/noisy_eval.json")
MAX_NEW_TOKENS = 16

SYSTEM_PROMPT = (
    "You are an intent classifier for customer-support messages. "
    "Respond with exactly one intent label from the allowed set. "
    "Output only the label, nothing else."
)

# 15 hand-written noisy test cases. Gold labels are from the 27-intent
# Bitext taxonomy. Noise types are annotated so we can report per-category
# accuracy.
NOISY_CASES = [
    # message, gold_label, noise_category
    ("whers my pkg at?",                                                      "track_order",             "typo/abbrev"),
    ("can u chck my invc #8823",                                              "check_invoice",           "typo/abbrev"),
    ("my card got declined wen i tried to pay for smth",                      "payment_issue",           "typo/abbrev"),
    ("how do i sign up for a new acc",                                        "create_account",          "typo/abbrev"),
    ("yo how do i speak to a real person",                                    "contact_human_agent",     "slang"),
    ("when will my order arrive bro",                                         "delivery_period",         "slang"),
    ("what's ur return policy for used stuff",                                "check_refund_policy",     "slang"),
    ("this is ridiculous just give me my damn refund",                        "get_refund",              "frustrated"),
    ("need refund ASAP pls",                                                  "get_refund",              "frustrated"),
    ("I DON'T want to receive your newsletter anymore",                       "newsletter_subscription", "negation"),
    ("account deletion procedure?",                                           "delete_account",          "negation"),
    ("Hi there! I was wondering if perhaps you could help me figure out how to go about placing a new order - thanks!", "place_order", "long/polite"),
    ("i cant log in anymore, keeps saying wrong password",                    "recover_password",        "long/polite"),
    ("need to update my shipping address and check when my order arrives",    "change_shipping_address", "mixed-intent"),
    ("i wanna leave some feedback but also maybe cancel my subscription",     "review",                  "mixed-intent"),
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


def main():
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

    results = []
    for msg, gold, cat in NOISY_CASES:
        pred = classify(model, tokenizer, msg)
        ok = (pred == gold)
        results.append({
            "message": msg,
            "gold": gold,
            "pred": pred,
            "category": cat,
            "correct": ok,
        })

    # Per-category tally
    per_cat: dict[str, list[int]] = {}
    for r in results:
        per_cat.setdefault(r["category"], [0, 0])
        per_cat[r["category"]][1] += 1
        if r["correct"]:
            per_cat[r["category"]][0] += 1

    total_correct = sum(r["correct"] for r in results)
    total = len(results)

    # Print a readable table
    print(f"{'category':<14} {'match':<6} {'gold':<26} {'pred':<26} {'message'}")
    print("-" * 120)
    for r in results:
        mark = "OK  " if r["correct"] else "MISS"
        print(f"{r['category']:<14} {mark:<6} {r['gold']:<26} {r['pred'][:25]:<26} {r['message'][:60]}")

    print("\n" + "=" * 60)
    print(f"Overall: {total_correct}/{total}  ({100 * total_correct / total:.1f} %)")
    print("By category:")
    for cat in sorted(per_cat):
        c, t = per_cat[cat]
        print(f"  {cat:<14} {c}/{t}")
    print("=" * 60)

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "overall": {
            "correct": total_correct,
            "total": total,
            "accuracy": total_correct / total,
        },
        "by_category": {
            cat: {"correct": c, "total": t, "accuracy": c / t}
            for cat, (c, t) in per_cat.items()
        },
        "cases": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
