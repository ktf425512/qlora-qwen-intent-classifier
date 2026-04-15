"""
Tiny end-to-end smoke test of train.py.

Monkey-patches constants so the full pipeline runs in ~3 minutes:
  - dataset subsampled to 300 rows
  - only a handful of optimizer steps
  - tiny eval subsets

Used only to validate the pipeline before the real training run.
"""
from __future__ import annotations

import train
from datasets import load_dataset as _orig_load


def small_load(name, **kwargs):
    ds = _orig_load(name, **kwargs)
    return ds.shuffle(seed=42).select(range(min(300, len(ds))))


train.load_dataset   = small_load      # dataset subsample
train.EVAL_GEN_LIMIT = 20              # tiny generation-based eval
train.COMPARE_SAMPLES = 3              # tiny base-vs-ft comparison
train.WARMUP_STEPS   = 2               # small warmup so a tiny run still steps LR

if __name__ == "__main__":
    train.main()
