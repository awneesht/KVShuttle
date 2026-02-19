"""Smoke test for the PyTorch backend — verifies loader, extractor, injector on CPU."""

from __future__ import annotations

import sys
import traceback

import numpy as np


def main() -> int:
    errors = []

    # ── 1. Loader ──
    print("=" * 60)
    print("1. Testing loader_torch (gpt2 on CPU)...")
    try:
        from kvshuttle.models.loader_torch import load_model_torch

        model, tokenizer, info = load_model_torch(
            "gpt2",  # not in registry → used as raw HF ID, already cached
            device="cpu",
        )
        print(f"   Model loaded: {info}")
        print(f"   layers={info.num_layers}, kv_heads={info.num_kv_heads}, "
              f"head_dim={info.head_dim}, arch={info.architecture}")
        assert info.num_layers > 0
        assert info.head_dim > 0
        print("   PASS")
    except Exception as e:
        errors.append(("loader", e))
        traceback.print_exc()
        print("   FAIL — cannot continue without model")
        return 1

    # ── 2. Extractor ──
    print("=" * 60)
    print("2. Testing kv_extractor_torch...")
    try:
        from kvshuttle.models.kv_extractor_torch import extract_kv_cache_torch

        prompt = "The quick brown fox jumps over the lazy dog."
        kv = extract_kv_cache_torch(model, tokenizer, prompt)
        print(f"   Extracted: {kv}")
        print(f"   keys shape:  {kv.keys.shape}  dtype={kv.keys.dtype}")
        print(f"   values shape: {kv.values.shape}  dtype={kv.values.dtype}")
        assert kv.keys.shape == kv.values.shape
        assert kv.keys.ndim == 4  # [layers, heads, seq, dim]
        assert kv.keys.dtype == np.float16
        assert kv.num_layers == info.num_layers
        assert kv.num_heads == info.num_kv_heads
        assert kv.head_dim == info.head_dim
        assert kv.seq_len > 0
        print("   PASS")
    except Exception as e:
        errors.append(("extractor", e))
        traceback.print_exc()
        print("   FAIL")

    # ── 3. Injector (continuation forward pass) ──
    print("=" * 60)
    print("3. Testing kv_injector_torch (continuation forward pass)...")
    try:
        from kvshuttle.models.kv_injector_torch import (
            forward_continuation_with_kv_cache_torch,
        )

        # Split KV at 80% for prefix, use rest as continuation tokens
        total = kv.seq_len
        split = max(1, int(total * 0.8))
        keys_prefix = kv.keys[:, :, :split, :]
        values_prefix = kv.values[:, :, :split, :]

        # Tokenize to get continuation token IDs (same logic as extractor)
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            msgs = [{"role": "user", "content": prompt}]
            result = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True
            )
            if isinstance(result, str):
                token_ids = tokenizer.encode(result)
            elif hasattr(result, "keys") and "input_ids" in result:
                token_ids = list(result["input_ids"])
            else:
                token_ids = [int(x) for x in result]
        else:
            token_ids = tokenizer.encode(prompt)

        continuation_ids = token_ids[split:]
        if not continuation_ids:
            # Ensure at least 1 continuation token
            continuation_ids = token_ids[-1:]
            keys_prefix = kv.keys[:, :, : total - 1, :]
            values_prefix = kv.values[:, :, : total - 1, :]

        print(f"   prefix_len={keys_prefix.shape[2]}, continuation_len={len(continuation_ids)}")

        logits = forward_continuation_with_kv_cache_torch(
            model, keys_prefix, values_prefix, continuation_ids
        )
        print(f"   logits shape: {logits.shape}  dtype={logits.dtype}")
        assert logits.ndim == 2  # [cont_len, vocab_size]
        assert logits.shape[0] == len(continuation_ids)
        assert not np.isnan(logits).any(), "logits contain NaN!"
        assert not np.isinf(logits).any(), "logits contain Inf!"
        print("   PASS")
    except Exception as e:
        errors.append(("injector", e))
        traceback.print_exc()
        print("   FAIL")

    # ── 4. Compression round-trip with injector ──
    print("=" * 60)
    print("4. Testing compress → decompress → inject round-trip...")
    try:
        from kvshuttle.compression.registry import get_compressor

        compressor = get_compressor("uniform_int8")
        compressed = compressor.compress(kv.keys, kv.values)
        keys_recon, values_recon = compressor.decompress(compressed)

        keys_recon_prefix = keys_recon[:, :, :split, :]
        values_recon_prefix = values_recon[:, :, :split, :]

        logits_recon = forward_continuation_with_kv_cache_torch(
            model, keys_recon_prefix, values_recon_prefix, continuation_ids
        )
        print(f"   logits_recon shape: {logits_recon.shape}")

        # Token agreement between original and INT8-reconstructed
        orig_tokens = np.argmax(logits, axis=-1)
        recon_tokens = np.argmax(logits_recon, axis=-1)
        agreement = np.mean(orig_tokens == recon_tokens)
        print(f"   token agreement (identity vs INT8): {agreement:.3f}")

        # Cosine similarity of KV
        from kvshuttle.evaluation.attention_error import compute_attention_error

        error = compute_attention_error(kv.keys, kv.values, keys_recon, values_recon)
        print(f"   key cosine sim:   {error.mean_key_cosine_sim:.4f}")
        print(f"   value cosine sim: {error.mean_val_cosine_sim:.4f}")
        assert error.mean_key_cosine_sim > 0.99, "INT8 key cosine too low"
        print("   PASS")
    except Exception as e:
        errors.append(("round-trip", e))
        traceback.print_exc()
        print("   FAIL")

    # ── Summary ──
    print("=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} test(s)")
        for name, err in errors:
            print(f"  - {name}: {err}")
        return 1
    else:
        print("ALL TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
