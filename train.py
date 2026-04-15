"""
QLoRA fine-tuning of Qwen2.5-7B-Instruct for customer-support intent classification.

Pipeline (6 sections):
  1. Config
  2. Data: load Bitext dataset, stratified 90/10 train/val split
  3. Format: ChatML (Qwen2.5-Instruct native format)
  4. Model: load base in 4-bit (NF4 + double-quant), inject LoRA adapters
  5. Train: 1 epoch, cosine schedule, per-epoch eval
  6. Evaluate: classification_report + confusion matrix + base vs fine-tuned
     demo, then merge adapter into bf16 base and save a deployable model.

Outputs
-------
  outputs/checkpoint-*/         HF trainer checkpoints (latest only)
  outputs/final_adapter/        PEFT adapter + tokenizer
  outputs/eval_report.json      Metrics + before/after sample comparison
  outputs/confusion_matrix.png  Val-set confusion matrix
  outputs/loss_curve.png        Training / eval loss curves
  merged_model/                 bf16 full model for deployment / inference
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

# ─── SECTION 1 · CONFIG ───────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME    = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

MAX_SEQ_LEN     = 512
LORA_RANK       = 16
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
EPOCHS          = 1
BATCH_SIZE      = 4
GRAD_ACCUM      = 4              # effective batch = 16
LR              = 2e-4
VAL_FRACTION    = 0.10
WARMUP_STEPS    = 50
SEED            = 42
EVAL_GEN_LIMIT  = 500            # val subsample used for generation-based eval
COMPARE_SAMPLES = 10             # # of examples in base vs fine-tuned table

OUTPUT_DIR = Path("./outputs")
MERGED_DIR = Path("./merged_model")
OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are an intent classifier for customer-support messages. "
    "Respond with exactly one intent label from the allowed set. "
    "Output only the label, nothing else."
)

# ─── Helpers ──────────────────────────────────────────────────────────────────
class LossLogger(TrainerCallback):
    """Print loss/lr/eval on their own line so tqdm doesn't eat them."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        parts = [f"[step {step:>5}]"]
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            parts.append(f"eval_loss={logs['eval_loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        print(" ".join(parts), flush=True)


def format_example(example, tokenizer):
    """ChatML-format one example as a supervised text string."""
    msgs = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": example["instruction"]},
        {"role": "assistant", "content": example["intent"]},
    ]
    example["text"] = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    return example


def build_inference_prompt(message, tokenizer):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": message},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def generate_batched(model, tokenizer, messages, batch_size=8, max_new_tokens=16):
    """Greedy batched generation. Returns list of decoded strings (generated portion only)."""
    orig_pad_side = tokenizer.padding_side
    orig_use_cache = model.config.use_cache
    tokenizer.padding_side = "left"        # correct padding side for left-decoded generation
    model.config.use_cache = True           # speed up decoding
    outputs = []
    try:
        for i in range(0, len(messages), batch_size):
            batch = messages[i : i + batch_size]
            prompts = [build_inference_prompt(m, tokenizer) for m in batch]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
            ).to(model.device)
            # autocast matches bf16 hidden states to the fp32 lm_head that
            # prepare_model_for_kbit_training upcast for training stability.
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            input_len = enc["input_ids"].shape[1]
            for seq in out:
                gen = seq[input_len:]
                outputs.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    finally:
        tokenizer.padding_side = orig_pad_side
        model.config.use_cache = orig_use_cache
    return outputs


def match_label(pred, valid):
    """Map a raw generation string to the closest valid label (or '__other__')."""
    if not pred:
        return "__other__"
    p = pred.strip().lower().split()[0] if pred.strip() else ""
    for label in valid:
        if p == label.lower():
            return label
    for label in valid:
        if label.lower() in pred.lower():
            return label
    return "__other__"


def plot_loss_curve(log_history, out_path):
    train_pts = [(h["step"], h["loss"]) for h in log_history if "loss" in h and "eval_loss" not in h]
    eval_pts  = [(h["step"], h["eval_loss"]) for h in log_history if "eval_loss" in h]
    if not train_pts:
        return
    ts, tl = zip(*train_pts)
    plt.figure(figsize=(9, 5))
    plt.plot(ts, tl, label="train loss")
    if eval_pts:
        es, el = zip(*eval_pts)
        plt.plot(es, el, "o-", label="eval loss", color="orange")
    plt.xlabel("step"); plt.ylabel("loss")
    plt.title("Training curves")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # ─── SECTION 2 · DATA ────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset ...")
    raw = load_dataset(DATASET_NAME, split="train")
    intents = sorted(set(raw["intent"]))
    print(f"  rows: {len(raw)}   intents: {len(intents)}")

    idx = np.arange(len(raw))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=VAL_FRACTION,
        stratify=raw["intent"],
        random_state=SEED,
    )
    train_ds_raw = raw.select(train_idx.tolist())
    val_ds_raw   = raw.select(val_idx.tolist())
    print(f"  train: {len(train_ds_raw)}   val: {len(val_ds_raw)}")

    # ─── SECTION 3 · TOKENIZER + ChatML ──────────────────────────────────────
    print("\n[2/6] Loading tokenizer and formatting ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = train_ds_raw.map(lambda e: format_example(e, tokenizer), desc="train fmt")
    val_ds   = val_ds_raw.map(lambda e: format_example(e, tokenizer),   desc="val fmt")
    print("  sample text (first train example):")
    print("  " + train_ds[0]["text"].replace("\n", "\n  "))

    # ─── SECTION 4 · MODEL + LoRA ────────────────────────────────────────────
    print("\n[3/6] Loading base model in 4-bit ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ─── SECTION 5 · TRAIN ───────────────────────────────────────────────────
    print("\n[4/6] Training ...")
    sft_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=False,
        dataset_num_proc=None,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LR,
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=SEED,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[LossLogger()],
        args=sft_args,
    )
    stats = trainer.train()
    print(f"  final train loss: {stats.training_loss:.4f}   steps: {stats.global_step}")

    # Save adapter + loss curve
    adapter_dir = OUTPUT_DIR / "final_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    plot_loss_curve(trainer.state.log_history, OUTPUT_DIR / "loss_curve.png")
    print(f"  adapter saved: {adapter_dir}")
    print(f"  loss curve:    {OUTPUT_DIR/'loss_curve.png'}")

    # ─── SECTION 6 · EVAL ────────────────────────────────────────────────────
    print("\n[5/6] Evaluating on val set (generation-based) ...")
    eval_ds = val_ds.shuffle(seed=SEED).select(range(min(EVAL_GEN_LIMIT, len(val_ds))))
    gold_labels = list(eval_ds["intent"])
    messages    = list(eval_ds["instruction"])

    model.eval()
    t_eval = time.time()
    preds_raw = generate_batched(model, tokenizer, messages, batch_size=8)
    preds_ft  = [match_label(p, intents) for p in preds_raw]
    print(f"  generation took {time.time()-t_eval:.1f}s on {len(messages)} samples")

    acc = float(np.mean([g == p for g, p in zip(gold_labels, preds_ft)]))
    macro_f1 = float(f1_score(gold_labels, preds_ft, average="macro", zero_division=0))
    labels_for_report = sorted(set(gold_labels) | set(preds_ft))
    cls_report = classification_report(
        gold_labels, preds_ft, labels=labels_for_report,
        zero_division=0, output_dict=True,
    )
    print(f"  accuracy : {acc:.4f}")
    print(f"  macro F1 : {macro_f1:.4f}")

    # Base vs fine-tuned on the same COMPARE_SAMPLES examples
    print(f"\n  Base vs fine-tuned on first {COMPARE_SAMPLES} val samples ...")
    cmp_msgs = messages[:COMPARE_SAMPLES]
    cmp_gold = gold_labels[:COMPARE_SAMPLES]
    cmp_ft_raw = preds_raw[:COMPARE_SAMPLES]
    cmp_ft = preds_ft[:COMPARE_SAMPLES]
    with model.disable_adapter():
        cmp_base_raw = generate_batched(model, tokenizer, cmp_msgs, batch_size=8)
    cmp_base = [match_label(p, intents) for p in cmp_base_raw]
    base_acc = float(np.mean([g == p for g, p in zip(cmp_gold, cmp_base)]))
    ft_acc   = float(np.mean([g == p for g, p in zip(cmp_gold, cmp_ft)]))
    print(f"  base acc on {COMPARE_SAMPLES} samples : {base_acc:.2f}")
    print(f"  FT   acc on same                      : {ft_acc:.2f}")

    examples = []
    for m, g, bp, bpr, fp, fpr in zip(cmp_msgs, cmp_gold, cmp_base, cmp_base_raw, cmp_ft, cmp_ft_raw):
        examples.append({
            "message": m,
            "gold": g,
            "base_pred": bp,
            "base_pred_raw": bpr,
            "finetuned_pred": fp,
            "finetuned_pred_raw": fpr,
        })

    # Save eval artifacts
    (OUTPUT_DIR / "eval_report.json").write_text(json.dumps({
        "n_eval": len(gold_labels),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": cls_report,
        "base_vs_finetuned": {
            "n": COMPARE_SAMPLES,
            "base_accuracy": base_acc,
            "finetuned_accuracy": ft_acc,
            "examples": examples,
        },
        "training": {
            "final_train_loss": float(stats.training_loss),
            "global_steps": int(stats.global_step),
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "model": MODEL_NAME,
            "lora_rank": LORA_RANK,
            "epochs": EPOCHS,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM,
            "learning_rate": LR,
        },
    }, indent=2, ensure_ascii=False))
    print(f"  eval_report.json saved")

    # Confusion matrix
    cm = confusion_matrix(gold_labels, preds_ft, labels=intents)
    plt.figure(figsize=(13, 11))
    sns.heatmap(cm, xticklabels=intents, yticklabels=intents, cmap="Blues",
                cbar_kws={"label": "count"})
    plt.xlabel("Predicted"); plt.ylabel("Gold")
    plt.title(f"Val confusion matrix  (n={len(gold_labels)}, macro F1={macro_f1:.3f})")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=120); plt.close()
    print("  confusion_matrix.png saved")

    # ─── SECTION 7 · MERGE + SAVE DEPLOYABLE MODEL ───────────────────────────
    print("\n[6/6] Merging adapter into bf16 base ...")
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()
    MERGED_DIR.mkdir(exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"  merged model saved: {MERGED_DIR}")

    total = time.time() - t0
    print("\n" + "=" * 64)
    print(f"DONE in {total/60:.1f} min")
    print(f"  val accuracy     : {acc:.4f}")
    print(f"  val macro F1     : {macro_f1:.4f}")
    print(f"  base -> FT delta : base {base_acc:.2f} -> FT {ft_acc:.2f} (on {COMPARE_SAMPLES} samples)")
    print(f"  adapter          : {adapter_dir}")
    print(f"  merged model     : {MERGED_DIR}")
    print(f"  artifacts        : {OUTPUT_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    main()
