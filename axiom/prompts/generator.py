"""System prompt generator for AXIOM."""

from pathlib import Path
from typing import Optional


def build_system_prompt(identity_statement: str, extra_guidance: Optional[str] = None) -> str:
    prompt_path = Path(__file__).resolve().parent / "system.txt"
    base_prompt = prompt_path.read_text(encoding="utf-8")
    if identity_statement not in base_prompt:
        base_prompt = f"{identity_statement}\n\n{base_prompt}"
    if extra_guidance:
        base_prompt = f"{base_prompt}\n\nAdditional Guidance:\n{extra_guidance}"
    return base_prompt
