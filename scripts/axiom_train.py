#!/usr/bin/env python
"""CLI entrypoint for AXIOM LoRA fine-tuning."""

from pathlib import Path

import yaml

from axiom.training.finetune_lora import LoRAFineTuneConfig, run_lora_finetune


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run AXIOM v0.2.0 LoRA fine-tuning")
    parser.add_argument("--base-model", required=True, help="Base model name or path")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Directory with AXIOM JSONL datasets")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for LoRA adapters")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "axiom" / "training" / "config_lora.yaml",
        help="Path to LoRA config YAML",
    )
    args = parser.parse_args()

    config_data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = LoRAFineTuneConfig(
        base_model_name=args.base_model,
        lora_r=int(config_data["lora_r"]),
        lora_alpha=int(config_data["lora_alpha"]),
        lora_dropout=float(config_data["lora_dropout"]),
        target_modules=list(config_data["target_modules"]),
        batch_size=int(config_data["batch_size"]),
        learning_rate=float(config_data["learning_rate"]),
        num_epochs=int(config_data["num_epochs"]),
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        device=args.device,
    )
    run_lora_finetune(config)


if __name__ == "__main__":
    main()
