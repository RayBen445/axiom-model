"""LoRA fine-tuning pipeline for AXIOM v0.2.0."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class LoRAFineTuneConfig:
    base_model_name: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    batch_size: int
    learning_rate: float
    num_epochs: int
    output_dir: Path
    dataset_dir: Path
    device: str = "cpu"


def _load_jsonl(path: Path) -> List[Dict[str, str]]:
    import json

    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records.append(
                {
                    "instruction": record["instruction"],
                    "response": record["response"],
                }
            )
    return records


def _weighted_examples(dataset_dir: Path) -> List[Dict[str, str]]:
    identity = _load_jsonl(dataset_dir / "axiom_identity_v1.jsonl")
    instruction = _load_jsonl(dataset_dir / "axiom_instruction_v1.jsonl")
    refusal = _load_jsonl(dataset_dir / "axiom_refusal_v1.jsonl")

    weighted: List[Dict[str, str]] = []
    weighted.extend(identity * 4)
    weighted.extend(instruction * 3)
    weighted.extend(refusal * 2)
    return weighted


def _iter_training_text(records: Iterable[Dict[str, str]], assistant_tag: str) -> Iterable[str]:
    for record in records:
        yield f"User: {record['instruction']}\n{assistant_tag}: {record['response']}"


class LoRATextDataset:
    def __init__(self, tokenizer, texts: List[str]) -> None:
        self.tokenizer = tokenizer
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.tokenizer(self.texts[idx], truncation=True)


def run_lora_finetune(config: LoRAFineTuneConfig) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    records = _weighted_examples(config.dataset_dir)
    training_text = list(_iter_training_text(records, assistant_tag="AXIOM"))
    dataset = LoRATextDataset(tokenizer, training_text)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=10,
        save_steps=50,
        no_cuda=config.device.lower() == "cpu",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
