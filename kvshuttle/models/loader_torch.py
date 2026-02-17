"""Model loading via PyTorch / HuggingFace Transformers (FP16)."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvshuttle.models.loader import ModelInfo

logger = logging.getLogger(__name__)

# Full-precision HuggingFace model IDs (FP16 on GPU)
TORCH_MODEL_REGISTRY: dict[str, str] = {
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}


def load_model_torch(
    model_name: str,
    dtype: torch.dtype = torch.float16,
    device: str = "auto",
) -> tuple:
    """Load a model and tokenizer via HuggingFace Transformers.

    Args:
        model_name: Short name from TORCH_MODEL_REGISTRY or a full HuggingFace model ID.
        dtype: Torch dtype for model weights (default: float16).
        device: Device map string for accelerate (default: "auto").

    Returns:
        Tuple of (model, tokenizer, model_info).
    """
    hf_id = TORCH_MODEL_REGISTRY.get(model_name, model_name)
    logger.info("Loading model %s (%s) with dtype=%s...", model_name, hf_id, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    # Extract config from model.config (standard HF attributes)
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_attention_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_attention_heads)
    hidden_size = cfg.hidden_size
    head_dim = hidden_size // num_attention_heads

    info = ModelInfo(
        name=model_name,
        hf_id=hf_id,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
    )

    logger.info(
        "Loaded %s: %d layers, %d KV heads, head_dim=%d, arch=%s",
        info.name,
        info.num_layers,
        info.num_kv_heads,
        info.head_dim,
        info.architecture,
    )

    return model, tokenizer, info
