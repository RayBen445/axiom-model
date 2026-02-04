"""LoRA fine-tuning pipeline for AXIOM."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from axiom.training.datasets import TextDataset, iter_training_text, load_jsonl


@dataclass
class LoRAConfig:
    model_path: Path
    output_path: Path
    dataset_path: Path
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 2e-4
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05


def run_lora_finetune(config: LoRAConfig) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    # LOAD TOKENIZER (LOCAL ONLY)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True
    )

    # FIX GPT-2 PADDING
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})


    # LOAD MODEL (LOCAL ONLY)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True
    )

     model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id



    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id


    # LoRA CONFIG
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # LOAD DATA
    examples = load_jsonl(config.dataset_path)
    training_text: List[str] = list(iter_training_text(examples))

  tokenized = tokenizer(
    training_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)


    training_args = TrainingArguments(
        output_dir=str(config.output_path),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=10,
        save_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TextDataset(tokenized),
    )

    trainer.train()
    model.save_pretrained(str(config.output_path))
