"""Dataset utilities for AXIOM."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class JsonlExample:
    prompt: str
    response: str
    metadata: Dict[str, str]


def load_jsonl(path: Path) -> List[JsonlExample]:
    examples: List[JsonlExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            examples.append(
                JsonlExample(
                    prompt=record["prompt"],
                    response=record["response"],
                    metadata=record.get("metadata", {}),
                )
            )
    return examples


def iter_training_text(
    examples: Iterable[JsonlExample],
    assistant_tag: str = "AXIOM",
) -> Iterable[str]:
    for example in examples:
        yield f"User: {example.prompt}\n{assistant_tag}: {example.response}"


class TextDataset:
    def __init__(self, tokenized: dict) -> None:
        self.tokenized = tokenized

    def __len__(self) -> int:
        return self.tokenized["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict:
        input_ids = self.tokenized["input_ids"][idx]
        attention_mask = self.tokenized.get("attention_mask")
        labels = input_ids.clone()
        if attention_mask is not None:
            attention = attention_mask[idx]
            labels = labels.masked_fill(attention == 0, -100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask[idx] if attention_mask is not None else None,
            "labels": labels,
        }
