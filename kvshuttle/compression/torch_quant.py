"""CUDA GPU-accelerated quantization compressors using PyTorch.

Implements 7 compressors using PyTorch tensor ops for NVIDIA CUDA GPUs.
Used to calibrate the CPU-to-GPU speedup factor for compression timing
in datacenter settings.

GPU-native compressors: uniform_int8, kivi_2bit, uniform_int4, fp8_e4m3,
cachegen, cascade_prune50_int4, palu_lr.

IMPORTANT: This module provides two interfaces:
  1. GPU-native functions (gpu_*) that operate entirely on torch.cuda tensors
     with no numpy copies — used for accurate GPU kernel timing.
  2. BaseCompressor wrappers that conform to the kvshuttle interface
     (numpy in/out) — used for integration with the benchmark pipeline.

Requires: pip install torch
"""

from __future__ import annotations

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.registry import register


def _get_device() -> "torch.device":
    """Get the best available torch device."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required. Install with: pip install torch")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===========================================================================
# GPU-NATIVE FUNCTIONS (zero-copy, for calibration timing)
# ===========================================================================
# These take and return torch tensors already on GPU.
# No numpy conversion happens inside these functions.
# Use torch.cuda.Event for precise GPU-only timing.
# ===========================================================================


def gpu_int8_compress(x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Per-layer symmetric INT8 quantization — GPU-native, zero-copy.

    Args:
        x: Float tensor on GPU, shape [num_layers, num_heads, seq_len, head_dim].

    Returns:
        (quantized_int8, per_layer_scales) — both on GPU.
    """
    qmax = 127
    num_layers = x.shape[0]
    flat = x.reshape(num_layers, -1)

    amax = flat.abs().amax(dim=1)
    amax = torch.where(amax == 0, torch.ones_like(amax), amax)
    scales = amax / qmax

    scales_exp = scales.view(num_layers, *([1] * (x.ndim - 1)))
    quantized = (x / scales_exp).round().clamp(-qmax, qmax).to(torch.int8)

    return quantized, scales


def gpu_int8_decompress(
    quantized: "torch.Tensor", scales: "torch.Tensor"
) -> "torch.Tensor":
    """Dequantize INT8 back to float — GPU-native, zero-copy.

    Args:
        quantized: INT8 tensor on GPU.
        scales: Per-layer scales on GPU.

    Returns:
        Float16 tensor on GPU.
    """
    num_layers = quantized.shape[0]
    s_exp = scales.view(num_layers, *([1] * (quantized.ndim - 1)))
    return (quantized.float() * s_exp).half()


def gpu_kivi_compress_keys(
    x: "torch.Tensor", qmax: int = 3
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Per-channel 2-bit quantization for keys — GPU-native, zero-copy.

    Quantizes along seq_len axis (dim=2). Scale/zero per [L, H, D].

    Args:
        x: Float tensor on GPU, shape [L, H, S, D].
        qmax: Maximum quantized value (3 for 2-bit).

    Returns:
        (quantized_uint8, scales, zeros) — all on GPU.
    """
    tmin = x.amin(dim=2)
    tmax = x.amax(dim=2)
    rng = tmax - tmin
    rng = torch.where(rng == 0, torch.ones_like(rng), rng)

    scales = rng / qmax
    zeros = tmin

    quantized = ((x - zeros.unsqueeze(2)) / scales.unsqueeze(2)).round().clamp(0, qmax).to(torch.uint8)
    return quantized, scales, zeros


def gpu_kivi_compress_values(
    x: "torch.Tensor", qmax: int = 3
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Per-token 2-bit quantization for values — GPU-native, zero-copy.

    Quantizes along head_dim axis (dim=3). Scale/zero per [L, H, S].

    Args:
        x: Float tensor on GPU, shape [L, H, S, D].
        qmax: Maximum quantized value (3 for 2-bit).

    Returns:
        (quantized_uint8, scales, zeros) — all on GPU.
    """
    tmin = x.amin(dim=3)
    tmax = x.amax(dim=3)
    rng = tmax - tmin
    rng = torch.where(rng == 0, torch.ones_like(rng), rng)

    scales = rng / qmax
    zeros = tmin

    quantized = ((x - zeros.unsqueeze(3)) / scales.unsqueeze(3)).round().clamp(0, qmax).to(torch.uint8)
    return quantized, scales, zeros


def gpu_kivi_decompress_keys(
    quantized: "torch.Tensor", scales: "torch.Tensor", zeros: "torch.Tensor"
) -> "torch.Tensor":
    """Dequantize per-channel keys — GPU-native, zero-copy."""
    return (quantized.float() * scales.unsqueeze(2) + zeros.unsqueeze(2)).half()


def gpu_kivi_decompress_values(
    quantized: "torch.Tensor", scales: "torch.Tensor", zeros: "torch.Tensor"
) -> "torch.Tensor":
    """Dequantize per-token values — GPU-native, zero-copy."""
    return (quantized.float() * scales.unsqueeze(3) + zeros.unsqueeze(3)).half()


# ===========================================================================
# GPU-NATIVE INT4 (per-group asymmetric, group_size=128)
# ===========================================================================


def gpu_int4_compress(
    x: "torch.Tensor", group_size: int = 128
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", list[int]]:
    """Per-group asymmetric INT4 quantization — GPU-native, zero-copy.

    Flattens, pads to group_size multiple, quantizes to [0,15], packs 2 per byte.

    Args:
        x: Float tensor on GPU, shape [L, H, S, D].
        group_size: Number of elements per quantization group.

    Returns:
        (packed_uint8, scales, zeros, original_shape) — all tensors on GPU.
    """
    original_shape = list(x.shape)
    flat = x.reshape(-1)
    n = flat.numel()

    # Pad to multiple of group_size
    padded_len = ((n + group_size - 1) // group_size) * group_size
    if padded_len > n:
        flat = torch.nn.functional.pad(flat, (0, padded_len - n))

    grouped = flat.reshape(-1, group_size)  # [n_groups, group_size]

    gmin = grouped.amin(dim=1)  # [n_groups]
    gmax = grouped.amax(dim=1)
    rng = gmax - gmin
    rng = torch.where(rng == 0, torch.ones_like(rng), rng)

    scales = rng / 15.0
    zeros = gmin

    # Quantize
    quantized = ((grouped - zeros.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    flat_q = quantized.reshape(-1)

    # Pack 2 values per byte on GPU
    if flat_q.numel() % 2 != 0:
        flat_q = torch.nn.functional.pad(flat_q, (0, 1))
    high = flat_q[0::2]
    low = flat_q[1::2]
    packed = (high << 4) | low

    return packed, scales, zeros, original_shape


def gpu_int4_decompress(
    packed: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    original_shape: list[int],
    group_size: int = 128,
) -> "torch.Tensor":
    """Dequantize packed INT4 back to float — GPU-native, zero-copy.

    Args:
        packed: Packed uint8 tensor on GPU.
        scales: Per-group scales on GPU.
        zeros: Per-group zeros on GPU.
        original_shape: Original tensor shape [L, H, S, D].
        group_size: Group size used during compression.

    Returns:
        Float16 tensor on GPU with original_shape.
    """
    # Unpack
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    flat_q = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    flat_q[0::2] = high
    flat_q[1::2] = low

    total_elements = 1
    for s in original_shape:
        total_elements *= s
    padded_len = scales.numel() * group_size
    grouped = flat_q[:padded_len].reshape(-1, group_size).float()

    # Dequantize
    result = grouped * scales.unsqueeze(1) + zeros.unsqueeze(1)
    return result.reshape(-1)[:total_elements].reshape(original_shape).half()


# ===========================================================================
# GPU-NATIVE FP8 (simulated E4M3 with per-layer scaling)
# ===========================================================================


def gpu_fp8_compress(
    x: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Simulated FP8 E4M3 quantization — GPU-native, zero-copy.

    Per-layer: scale = amax / 240, quantize to uint8 via offset+round.

    Args:
        x: Float tensor on GPU, shape [L, H, S, D].

    Returns:
        (quantized_uint8, per_layer_scales) — both on GPU.
    """
    num_layers = x.shape[0]
    flat = x.reshape(num_layers, -1)

    amax = flat.abs().amax(dim=1)  # [L]
    amax = torch.where(amax == 0, torch.ones_like(amax), amax)
    scales = amax / 240.0

    scales_exp = scales.view(num_layers, *([1] * (x.ndim - 1)))
    quantized = (x / scales_exp + 128.0).round().clamp(0, 255).to(torch.uint8)

    return quantized, scales


def gpu_fp8_decompress(
    quantized: "torch.Tensor", scales: "torch.Tensor"
) -> "torch.Tensor":
    """Dequantize simulated FP8 back to float — GPU-native, zero-copy.

    Args:
        quantized: Uint8 tensor on GPU.
        scales: Per-layer scales on GPU.

    Returns:
        Float16 tensor on GPU.
    """
    num_layers = quantized.shape[0]
    s_exp = scales.view(num_layers, *([1] * (quantized.ndim - 1)))
    return ((quantized.float() - 128.0) * s_exp).half()


# ===========================================================================
# GPU-NATIVE CACHEGEN (anchor INT8 + delta INT4, chunked)
# ===========================================================================


def gpu_cachegen_compress(
    x: "torch.Tensor", chunk_size: int = 10
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", list[int]]:
    """CacheGen-style anchor+delta compression — GPU-native, zero-copy.

    Per (L*H) slice: tokens split into chunks. Token 0 of each chunk is an
    anchor (INT8 quantized). Tokens 1..chunk_size-1 are deltas from anchor
    (INT4 quantized and packed).

    Args:
        x: Float tensor on GPU, shape [L, H, S, D].
        chunk_size: Tokens per chunk (default 10).

    Returns:
        (anchor_quant, anchor_scales, anchor_zeros,
         delta_packed, delta_scales, delta_zeros, original_shape)
    """
    original_shape = list(x.shape)
    L, H, S, D = x.shape
    N = L * H
    flat = x.reshape(N, S, D)  # [N, S, D]

    num_chunks = (S + chunk_size - 1) // chunk_size

    # Extract anchors: first token of each chunk
    anchor_indices = torch.arange(0, S, chunk_size, device=x.device)
    anchors = flat[:, anchor_indices, :]  # [N, num_chunks, D]

    # INT8 quantize anchors: per-chunk min/max
    a_flat = anchors.reshape(N * num_chunks, D)
    a_min = a_flat.amin(dim=1, keepdim=True)
    a_max = a_flat.amax(dim=1, keepdim=True)
    a_rng = a_max - a_min
    a_rng = torch.where(a_rng == 0, torch.ones_like(a_rng), a_rng)
    a_scales = (a_rng / 255.0).squeeze(1)  # [N*num_chunks]
    a_zeros = a_min.squeeze(1)  # [N*num_chunks]
    a_quant = ((a_flat - a_zeros.unsqueeze(1)) / a_scales.unsqueeze(1)).round().clamp(0, 255).to(torch.uint8)

    # Compute deltas: for each chunk, subtract anchor from non-anchor tokens
    # Build expanded anchors for subtraction
    anchors_expanded = torch.zeros_like(flat)  # [N, S, D]
    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, S)
        anchors_expanded[:, start:end, :] = anchors[:, c:c+1, :]

    # Mask out anchor positions (we only need deltas for non-anchor tokens)
    all_indices = torch.arange(S, device=x.device)
    is_anchor = (all_indices % chunk_size) == 0
    delta_mask = ~is_anchor  # [S]
    delta_indices = all_indices[delta_mask]

    deltas = flat[:, delta_indices, :] - anchors_expanded[:, delta_indices, :]  # [N, num_delta_tokens, D]

    # INT4 quantize deltas: per-(slice, chunk) groups
    # Reshape deltas to group by chunk for better quantization
    d_flat = deltas.reshape(-1, D)
    num_d_groups = d_flat.shape[0]
    d_min = d_flat.amin(dim=1, keepdim=True)
    d_max = d_flat.amax(dim=1, keepdim=True)
    d_rng = d_max - d_min
    d_rng = torch.where(d_rng == 0, torch.ones_like(d_rng), d_rng)
    d_scales = (d_rng / 15.0).squeeze(1)
    d_zeros = d_min.squeeze(1)
    d_quant = ((d_flat - d_zeros.unsqueeze(1)) / d_scales.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)

    # Pack deltas: 2 values per byte
    d_flat_q = d_quant.reshape(-1)
    if d_flat_q.numel() % 2 != 0:
        d_flat_q = torch.nn.functional.pad(d_flat_q, (0, 1))
    delta_packed = (d_flat_q[0::2] << 4) | d_flat_q[1::2]

    return a_quant, a_scales, a_zeros, delta_packed, d_scales, d_zeros, original_shape


def gpu_cachegen_decompress(
    a_quant: "torch.Tensor",
    a_scales: "torch.Tensor",
    a_zeros: "torch.Tensor",
    delta_packed: "torch.Tensor",
    d_scales: "torch.Tensor",
    d_zeros: "torch.Tensor",
    original_shape: list[int],
    chunk_size: int = 10,
) -> "torch.Tensor":
    """Decompress CacheGen anchor+delta — GPU-native, zero-copy.

    Args:
        a_quant: Anchor quantized uint8, [N*num_chunks, D].
        a_scales: Anchor scales, [N*num_chunks].
        a_zeros: Anchor zeros, [N*num_chunks].
        delta_packed: Packed delta uint8.
        d_scales: Delta scales per row.
        d_zeros: Delta zeros per row.
        original_shape: [L, H, S, D].
        chunk_size: Chunk size used during compression.

    Returns:
        Float16 tensor on GPU with original_shape.
    """
    L, H, S, D = original_shape
    N = L * H
    num_chunks = (S + chunk_size - 1) // chunk_size

    # Dequantize anchors
    anchors = (a_quant.float() * a_scales.unsqueeze(1) + a_zeros.unsqueeze(1))  # [N*num_chunks, D]
    anchors = anchors.reshape(N, num_chunks, D)

    # Unpack deltas
    high = (delta_packed >> 4) & 0x0F
    low = delta_packed & 0x0F
    d_unpacked = torch.empty(delta_packed.numel() * 2, dtype=torch.uint8, device=delta_packed.device)
    d_unpacked[0::2] = high
    d_unpacked[1::2] = low

    # Calculate expected delta elements
    all_indices = torch.arange(S, device=a_quant.device)
    num_delta_tokens = int((all_indices % chunk_size != 0).sum().item())
    total_delta_elements = N * num_delta_tokens * D
    d_flat = d_unpacked[:total_delta_elements].reshape(-1, D).float()

    # Dequantize deltas
    deltas = d_flat * d_scales[:d_flat.shape[0]].unsqueeze(1) + d_zeros[:d_flat.shape[0]].unsqueeze(1)
    deltas = deltas.reshape(N, num_delta_tokens, D)

    # Reconstruct
    result = torch.zeros(N, S, D, device=a_quant.device, dtype=torch.float32)

    # Place anchors
    anchor_positions = torch.arange(0, S, chunk_size, device=a_quant.device)
    result[:, anchor_positions, :] = anchors

    # Place deltas + anchor
    delta_positions = all_indices[all_indices % chunk_size != 0]
    # Compute which chunk each delta belongs to for anchor lookup
    chunk_ids = delta_positions // chunk_size  # [num_delta_tokens]
    result[:, delta_positions, :] = deltas + anchors[:, chunk_ids, :]

    return result.reshape(original_shape).half()


# ===========================================================================
# GPU-NATIVE CASCADE (prune 50% + INT4)
# ===========================================================================


def gpu_cascade_compress(
    keys: "torch.Tensor",
    values: "torch.Tensor",
    keep_ratio: float = 0.5,
    group_size: int = 128,
) -> tuple:
    """Cascade: prune 50% tokens by importance, then INT4 quantize — GPU-native.

    Args:
        keys: Float tensor on GPU, shape [L, H, S, D].
        values: Float tensor on GPU, shape [L, H, S, D].
        keep_ratio: Fraction of tokens to keep.
        group_size: INT4 quantization group size.

    Returns:
        (k_packed, k_scales, k_zeros, k_pruned_shape,
         v_packed, v_scales, v_zeros, v_pruned_shape,
         keep_indices, original_shape)
    """
    L, H, S, D = keys.shape
    keep_count = max(1, int(S * keep_ratio))
    original_shape = list(keys.shape)

    # Step 1: Compute importance via key L2 norm
    key_norms = torch.linalg.norm(keys, dim=3)  # [L, H, S]
    importance = key_norms.mean(dim=(0, 1))  # [S]

    # Protect first 2 and last 2 tokens
    protect = min(2, S)
    # Zero out protected tokens so they don't compete, then add back
    importance_copy = importance.clone()
    importance_copy[:protect] = -float('inf')
    if S > protect:
        importance_copy[max(protect, S - protect):] = -float('inf')

    remaining_budget = keep_count - min(2 * protect, S)
    if remaining_budget > 0:
        _, topk_indices = torch.topk(importance_copy, remaining_budget)
    else:
        topk_indices = torch.tensor([], dtype=torch.long, device=keys.device)

    # Combine protected + topk, sort
    protected_indices = list(range(protect)) + list(range(max(protect, S - protect), S))
    protected_tensor = torch.tensor(protected_indices, dtype=torch.long, device=keys.device)
    keep_indices = torch.cat([protected_tensor, topk_indices])
    keep_indices = torch.sort(keep_indices)[0][:keep_count]

    # Step 2: Prune
    pruned_keys = keys[:, :, keep_indices, :]  # [L, H, keep_count, D]
    pruned_values = values[:, :, keep_indices, :]

    # Step 3: INT4 quantize pruned tensors
    k_packed, k_scales, k_zeros, k_shape = gpu_int4_compress(pruned_keys, group_size)
    v_packed, v_scales, v_zeros, v_shape = gpu_int4_compress(pruned_values, group_size)

    return (k_packed, k_scales, k_zeros, k_shape,
            v_packed, v_scales, v_zeros, v_shape,
            keep_indices, original_shape)


def gpu_cascade_decompress(
    k_packed: "torch.Tensor",
    k_scales: "torch.Tensor",
    k_zeros: "torch.Tensor",
    k_pruned_shape: list[int],
    v_packed: "torch.Tensor",
    v_scales: "torch.Tensor",
    v_zeros: "torch.Tensor",
    v_pruned_shape: list[int],
    keep_indices: "torch.Tensor",
    original_shape: list[int],
    group_size: int = 128,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Decompress cascade: INT4 decompress, then scatter back — GPU-native.

    Returns:
        (keys, values) — Float16 tensors on GPU with original_shape.
    """
    # INT4 decompress
    pruned_keys = gpu_int4_decompress(k_packed, k_scales, k_zeros, k_pruned_shape, group_size)
    pruned_values = gpu_int4_decompress(v_packed, v_scales, v_zeros, v_pruned_shape, group_size)

    L, H, S, D = original_shape

    # Scatter back
    keys = torch.zeros(L, H, S, D, device=k_packed.device, dtype=torch.float16)
    values = torch.zeros(L, H, S, D, device=k_packed.device, dtype=torch.float16)
    keys[:, :, keep_indices, :] = pruned_keys
    values[:, :, keep_indices, :] = pruned_values

    return keys, values


# ===========================================================================
# GPU-NATIVE PALU (truncated SVD, rank_ratio=0.25)
# ===========================================================================


def gpu_palu_compress(
    x: "torch.Tensor", rank_ratio: float = 0.25
) -> tuple["torch.Tensor", "torch.Tensor", list[int], int]:
    """Truncated SVD low-rank compression — GPU-native, zero-copy.

    Per (L*H) slice: SVD → truncate to rank → store US and Vt.

    Args:
        x: Float tensor on GPU, shape [L, H, S, D].
        rank_ratio: Fraction of min(S, D) to keep.

    Returns:
        (US, Vt, original_shape, rank) — tensors on GPU.
        US shape: [L*H, S, rank], Vt shape: [L*H, rank, D].
    """
    original_shape = list(x.shape)
    L, H, S, D = x.shape
    rank = max(1, int(min(S, D) * rank_ratio))
    N = L * H

    # Batched SVD
    mats = x.reshape(N, S, D)  # [N, S, D]
    U, Sigma, Vt = torch.linalg.svd(mats, full_matrices=False)

    # Truncate
    US = U[:, :, :rank] * Sigma[:, None, :rank]  # [N, S, rank]
    Vt_r = Vt[:, :rank, :]  # [N, rank, D]

    return US.half(), Vt_r.half(), original_shape, rank


def gpu_palu_decompress(
    US: "torch.Tensor",
    Vt: "torch.Tensor",
    original_shape: list[int],
) -> "torch.Tensor":
    """Reconstruct from SVD factors — GPU-native, zero-copy.

    Args:
        US: [N, S, rank] float16 on GPU.
        Vt: [N, rank, D] float16 on GPU.
        original_shape: [L, H, S, D].

    Returns:
        Float16 tensor on GPU with original_shape.
    """
    result = torch.bmm(US.float(), Vt.float())  # [N, S, D]
    return result.reshape(original_shape).half()


# ===========================================================================
# Timing utility
# ===========================================================================


def gpu_time_ms(fn, *args, warmup: int = 3, repeats: int = 10) -> tuple[float, float]:
    """Time a GPU-native function using CUDA events (microsecond precision).

    Returns (median_ms, std_ms). All args must already be on GPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for gpu_time_ms")

    # Warmup
    for _ in range(warmup):
        fn(*args)
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn(*args)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    import statistics
    return statistics.median(times), statistics.stdev(times) if len(times) > 1 else 0.0


# ===========================================================================
# BaseCompressor wrappers (numpy in/out, for pipeline integration)
# ===========================================================================


@register("torch_uniform_int8")
class TorchUniformInt8Compressor(BaseCompressor):
    """Per-layer symmetric INT8 quantization using PyTorch (CUDA GPU).

    Uses GPU-native kernels. numpy conversion happens outside the hot path.
    """

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes
        device = _get_device()

        # Move to GPU once
        k_gpu = torch.from_numpy(keys.astype(np.float32)).to(device)
        v_gpu = torch.from_numpy(values.astype(np.float32)).to(device)

        # GPU-native compress
        k_quant, k_scales = gpu_int8_compress(k_gpu)
        v_quant, v_scales = gpu_int8_compress(v_gpu)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Move back for serialization
        k_q_np = k_quant.cpu().numpy()
        v_q_np = v_quant.cpu().numpy()
        data = k_q_np.tobytes() + v_q_np.tobytes()

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "k_scales": k_scales.cpu().tolist(),
                "v_scales": v_scales.cpu().tolist(),
                "key_bytes_len": len(k_q_np.tobytes()),
                "bits": 8,
            },
            original_size_bytes=original_size,
            compressed_size_bytes=len(data),
            num_layers=keys.shape[0],
            num_heads=keys.shape[1],
            seq_len=keys.shape[2],
            head_dim=keys.shape[3],
        )

    def decompress(self, compressed: CompressedKVCache) -> tuple[np.ndarray, np.ndarray]:
        meta = compressed.metadata
        key_len = meta["key_bytes_len"]
        device = _get_device()

        k_quant = torch.from_numpy(
            np.frombuffer(compressed.data[:key_len], dtype=np.int8).reshape(meta["key_shape"]).copy()
        ).to(device)
        v_quant = torch.from_numpy(
            np.frombuffer(compressed.data[key_len:], dtype=np.int8).reshape(meta["val_shape"]).copy()
        ).to(device)
        k_scales = torch.tensor(meta["k_scales"], dtype=torch.float32, device=device)
        v_scales = torch.tensor(meta["v_scales"], dtype=torch.float32, device=device)

        keys = gpu_int8_decompress(k_quant, k_scales)
        values = gpu_int8_decompress(v_quant, v_scales)

        if device.type == "cuda":
            torch.cuda.synchronize()

        return keys.cpu().numpy(), values.cpu().numpy()

    @property
    def name(self) -> str:
        return "torch_uniform_int8"


@register("torch_kivi_2bit")
class TorchKiviCompressor(BaseCompressor):
    """KIVI-style 2-bit asymmetric quantization using PyTorch (CUDA GPU)."""

    def __init__(self, bits: int = 2):
        self._bits = bits
        self._qmax = (1 << bits) - 1

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes
        num_layers, num_heads, seq_len, head_dim = keys.shape
        device = _get_device()

        k_gpu = torch.from_numpy(keys.astype(np.float32)).to(device)
        v_gpu = torch.from_numpy(values.astype(np.float32)).to(device)

        k_quant, k_scales, k_zeros = gpu_kivi_compress_keys(k_gpu, self._qmax)
        v_quant, v_scales, v_zeros = gpu_kivi_compress_values(v_gpu, self._qmax)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Pack 2-bit on CPU (trivial cost)
        k_packed = _pack_nbits(k_quant.cpu().numpy(), self._bits)
        v_packed = _pack_nbits(v_quant.cpu().numpy(), self._bits)

        kp_bytes = k_packed.tobytes()
        vp_bytes = v_packed.tobytes()
        ks_bytes = k_scales.cpu().numpy().tobytes()
        vs_bytes = v_scales.cpu().numpy().tobytes()
        kz_bytes = k_zeros.cpu().numpy().tobytes()
        vz_bytes = v_zeros.cpu().numpy().tobytes()

        data = kp_bytes + vp_bytes + ks_bytes + vs_bytes + kz_bytes + vz_bytes

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "bits": self._bits,
                "kp_len": len(kp_bytes),
                "vp_len": len(vp_bytes),
                "ks_len": len(ks_bytes),
                "vs_len": len(vs_bytes),
                "kz_len": len(kz_bytes),
                "k_scales_shape": list(k_scales.shape),
                "v_scales_shape": list(v_scales.shape),
            },
            original_size_bytes=original_size,
            compressed_size_bytes=len(data),
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
        )

    def decompress(self, compressed: CompressedKVCache) -> tuple[np.ndarray, np.ndarray]:
        meta = compressed.metadata
        buf = compressed.data
        bits = meta["bits"]
        device = _get_device()

        kp_end = meta["kp_len"]
        vp_end = kp_end + meta["vp_len"]
        ks_end = vp_end + meta["ks_len"]
        vs_end = ks_end + meta["vs_len"]
        kz_end = vs_end + meta["kz_len"]

        # Unpack on CPU then move to GPU
        k_packed = np.frombuffer(buf[:kp_end], dtype=np.uint8)
        v_packed = np.frombuffer(buf[kp_end:vp_end], dtype=np.uint8)
        key_shape = meta["key_shape"]
        val_shape = meta["val_shape"]
        total_k = 1
        total_v = 1
        for s in key_shape:
            total_k *= s
        for s in val_shape:
            total_v *= s

        k_quant = torch.from_numpy(
            _unpack_nbits(k_packed, bits, total_k).reshape(key_shape)
        ).to(device)
        v_quant = torch.from_numpy(
            _unpack_nbits(v_packed, bits, total_v).reshape(val_shape)
        ).to(device)

        k_scales = torch.from_numpy(
            np.frombuffer(buf[vp_end:ks_end], dtype=np.float32).reshape(meta["k_scales_shape"]).copy()
        ).to(device)
        v_scales = torch.from_numpy(
            np.frombuffer(buf[ks_end:vs_end], dtype=np.float32).reshape(meta["v_scales_shape"]).copy()
        ).to(device)
        k_zeros = torch.from_numpy(
            np.frombuffer(buf[vs_end:kz_end], dtype=np.float32).reshape(meta["k_scales_shape"]).copy()
        ).to(device)
        v_zeros = torch.from_numpy(
            np.frombuffer(buf[kz_end:], dtype=np.float32).reshape(meta["v_scales_shape"]).copy()
        ).to(device)

        keys = gpu_kivi_decompress_keys(k_quant, k_scales, k_zeros)
        values = gpu_kivi_decompress_values(v_quant, v_scales, v_zeros)

        if device.type == "cuda":
            torch.cuda.synchronize()

        return keys.cpu().numpy(), values.cpu().numpy()

    @property
    def name(self) -> str:
        return "torch_kivi_2bit"


# ---------------------------------------------------------------------------
# Bit packing helpers (CPU — trivial cost)
# ---------------------------------------------------------------------------


def _pack_nbits(arr: np.ndarray, bits: int) -> np.ndarray:
    """Pack n-bit unsigned integers into uint8 bytes."""
    flat = arr.reshape(-1).astype(np.uint8)
    vals_per_byte = 8 // bits

    pad_len = (vals_per_byte - len(flat) % vals_per_byte) % vals_per_byte
    if pad_len > 0:
        flat = np.append(flat, np.zeros(pad_len, dtype=np.uint8))

    packed = np.zeros(len(flat) // vals_per_byte, dtype=np.uint8)
    for i in range(vals_per_byte):
        packed |= flat[i::vals_per_byte] << (bits * (vals_per_byte - 1 - i))

    return packed


def _unpack_nbits(packed: np.ndarray, bits: int, total_elements: int) -> np.ndarray:
    """Unpack uint8 bytes back to n-bit values."""
    vals_per_byte = 8 // bits
    mask = (1 << bits) - 1

    flat = np.zeros(len(packed) * vals_per_byte, dtype=np.uint8)
    for i in range(vals_per_byte):
        flat[i::vals_per_byte] = (packed >> (bits * (vals_per_byte - 1 - i))) & mask

    return flat[:total_elements]
