"""Model loading via MLX."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Supported models and their HuggingFace IDs
MODEL_REGISTRY: dict[str, str] = {
    "qwen2.5-3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "llama-3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "phi-3.5-mini": "mlx-community/Phi-3.5-mini-instruct-4bit",
    "qwen2.5-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "llama-3.1-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mistral-7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
}


@dataclass
class ModelInfo:
    """Metadata about a loaded model.

    Attributes:
        name: Short name (e.g., "qwen2.5-3b").
        hf_id: HuggingFace model ID.
        num_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads (queries).
        num_kv_heads: Number of key/value heads (may differ from attention heads for GQA).
        head_dim: Dimension per attention head.
        hidden_size: Model hidden dimension.
        architecture: "GQA" or "MHA".
    """

    name: str
    hf_id: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int

    @property
    def architecture(self) -> str:
        return "MHA" if self.num_kv_heads == self.num_attention_heads else "GQA"


def load_model(model_name: str) -> tuple:
    """Load a model and tokenizer via mlx-lm.

    Args:
        model_name: Short name from MODEL_REGISTRY or a full HuggingFace model ID.

    Returns:
        Tuple of (model, tokenizer, model_info).
    """
    from mlx_lm import load

    hf_id = MODEL_REGISTRY.get(model_name, model_name)
    logger.info("Loading model %s (%s)...", model_name, hf_id)

    model, tokenizer = load(hf_id)

    num_layers = len(model.model.layers)

    # Extract config from model.args (preferred, always accurate)
    # or fall back to attention module attributes
    args = getattr(model, "args", None)
    if args is not None:
        hidden_size = args.hidden_size
        num_attention_heads = args.num_attention_heads
        num_kv_heads = getattr(args, "num_key_value_heads", num_attention_heads)
        num_layers = getattr(args, "num_hidden_layers", num_layers)
    else:
        attn = model.model.layers[0].self_attn
        num_attention_heads = _get_attr(attn, "num_heads", "n_heads")
        num_kv_heads = _get_attr(attn, "num_kv_heads", "n_kv_heads", default=num_attention_heads)
        hidden_size = _get_attr(model.model, "hidden_size", default=4096)

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


def _get_attr(obj, *names: str, default=None):
    """Try multiple attribute names, return the first that exists."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if default is not None:
        return default
    raise AttributeError(f"Object {obj} has none of attributes: {names}")
