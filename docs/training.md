# AXIOM Training Guide

## Overview
AXIOM v0.2.0 uses parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation). LoRA adds small, trainable adapter matrices to a frozen base model, which reduces compute cost while preserving the base weights.

## Data Preparation
Training data lives in `data/` and uses JSONL with `instruction` and `response` fields:
- `axiom_identity_v1.jsonl` (highest weight)
- `axiom_instruction_v1.jsonl` (primary)
- `axiom_refusal_v1.jsonl` (medium weight)

The LoRA pipeline merges these datasets with weighted sampling to preserve identity alignment during training.

## LoRA Fine-Tuning (AXIOM v0.2.0)
Configure hyperparameters in `axiom/training/config_lora.yaml`, then run:
```
python scripts/axiom_train.py \
  --base-model gpt2 \
  --dataset-dir data \
  --output-dir models/axiom-v0.2.0 \
  --device cpu
```

This creates LoRA adapter weights in the output directory. It does not modify the base model.

## Identity Consistency Evaluation
After training, evaluate identity alignment using the existing identity evaluation suite:
```
python scripts/axiom_eval.py --suite identity
```

Review outputs for correct AXIOM identification and ownership attribution to Cool Shot Systems.

## Best Practices
- Keep identity and refusal datasets included in every run to prevent drift.
- Track the dataset versions used for each training artifact.
- Validate identity compliance after each training cycle.
