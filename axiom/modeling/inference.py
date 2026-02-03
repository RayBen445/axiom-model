"""Inference pipeline for AXIOM."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from axiom.modeling.tokenizer import TokenizerAdapter
from axiom.utils.validators import IdentityValidator, SafetyValidator


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05


class AxiomInference:
    """High-level inference API with identity and safety enforcement."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        identity_validator: IdentityValidator,
        safety_validator: SafetyValidator,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = TokenizerAdapter(tokenizer)
        self.identity_validator = identity_validator
        self.safety_validator = safety_validator
        self.generation_config = generation_config or GenerationConfig()

    def generate(self, prompt: str, system_prompt: str) -> Dict[str, str]:
        if not self.safety_validator.is_prompt_allowed(prompt):
            return {
                "response": self.identity_validator.refusal_statement,
                "reason": "blocked_topic",
            }

        assistant_tag = self.identity_validator.model_name.strip() or "AXIOM"
        combined_prompt = f"{system_prompt}\n\nUser: {prompt}\n{assistant_tag}:"
        encoded = self.tokenizer.encode(combined_prompt)
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            encoded = encoded.__class__(
                input_ids=encoded.input_ids.to(model_device),
                attention_mask=(
                    encoded.attention_mask.to(model_device)
                    if encoded.attention_mask is not None
                    else None
                ),
            )
        output = self.model.generate(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            max_new_tokens=self.generation_config.max_new_tokens,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            repetition_penalty=self.generation_config.repetition_penalty,
        )
        input_length = encoded.input_ids.shape[-1]
        generated_tokens = output[0][input_length:]
        response = self.tokenizer.decode(generated_tokens).strip()

        if not self.identity_validator.is_identity_compliant(response):
            response = self.identity_validator.identity_statement

        return {"response": response, "reason": "ok"}
