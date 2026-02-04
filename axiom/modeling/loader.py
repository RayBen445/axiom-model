"""Model loading abstractions for AXIOM."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModelArtifacts:
    model: Any
    tokenizer: Any


@dataclass
class BaseModelConfig:
    model_path: Path
    tokenizer_path: Optional[Path] = None
    device: str = "cpu"
    dtype: Optional[str] = None
    trust_remote_code: bool = False


class BaseModelLoader:
    """Loads a base text model and tokenizer from local paths."""

    def __init__(self, config: BaseModelConfig) -> None:
        self.config = config

    def load(self) -> ModelArtifacts:
        tokenizer_path = self.config.tokenizer_path or self.config.model_path
        model, tokenizer = self._load_transformer_artifacts(
            model_path=self.config.model_path,
            tokenizer_path=tokenizer_path,
            device=self.config.device,
            dtype=self.config.dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        return ModelArtifacts(model=model, tokenizer=tokenizer)

    def _load_transformer_artifacts(
        self,
        model_path: Path,
        tokenizer_path: Path,
        device: str,
        dtype: Optional[str],
        trust_remote_code: bool,
    ) -> Tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype:
            import torch

            allowed_dtypes = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if dtype not in allowed_dtypes:
                raise ValueError(
                    f"Unsupported dtype '{dtype}'. Supported dtypes are: {', '.join(sorted(allowed_dtypes.keys()))}."
                )
            model_kwargs["torch_dtype"] = allowed_dtypes[dtype]

        # Check if model_path is a LoRA adapter directory
        adapter_config_path = Path(model_path) / "adapter_config.json"
        is_lora_adapter = adapter_config_path.exists()

        if is_lora_adapter:
            # Load base model (gpt2) and wrap with PeftModel
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained("gpt2", **model_kwargs)
            model = PeftModel.from_pretrained(base_model, model_path)
            # Always load tokenizer from base model (gpt2) for LoRA adapters
            tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=trust_remote_code)
        else:
            # Load model normally from model_path
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)

        model.to(device)
        model.eval()
        return model, tokenizer
