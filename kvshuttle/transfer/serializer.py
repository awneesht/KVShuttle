"""Serialize CompressedKVCache to bytes and back for transfer."""

from __future__ import annotations

import json
import logging
import struct

from kvshuttle.compression.base import CompressedKVCache

logger = logging.getLogger(__name__)

# Wire format:
# [4 bytes: metadata JSON length (uint32)]
# [N bytes: metadata JSON]
# [remaining: compressed data payload]
_HEADER_FMT = "!I"  # network byte order, unsigned 32-bit int
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def serialize(compressed: CompressedKVCache) -> bytes:
    """Serialize a CompressedKVCache to a byte buffer for transfer.

    Args:
        compressed: The compressed KV cache to serialize.

    Returns:
        Byte buffer containing metadata header + compressed data.
    """
    full_meta = {
        **compressed.metadata,
        "_original_size_bytes": compressed.original_size_bytes,
        "_compressed_size_bytes": compressed.compressed_size_bytes,
        "_num_layers": compressed.num_layers,
        "_num_heads": compressed.num_heads,
        "_seq_len": compressed.seq_len,
        "_head_dim": compressed.head_dim,
    }
    meta_bytes = json.dumps(full_meta).encode("utf-8")
    header = struct.pack(_HEADER_FMT, len(meta_bytes))
    return header + meta_bytes + compressed.data


def deserialize(buffer: bytes) -> CompressedKVCache:
    """Deserialize a byte buffer back to a CompressedKVCache.

    Args:
        buffer: Byte buffer produced by serialize().

    Returns:
        Reconstructed CompressedKVCache.
    """
    meta_len = struct.unpack(_HEADER_FMT, buffer[:_HEADER_SIZE])[0]
    meta_end = _HEADER_SIZE + meta_len
    full_meta = json.loads(buffer[_HEADER_SIZE:meta_end].decode("utf-8"))

    # Extract internal fields
    original_size = full_meta.pop("_original_size_bytes")
    compressed_size = full_meta.pop("_compressed_size_bytes")
    num_layers = full_meta.pop("_num_layers")
    num_heads = full_meta.pop("_num_heads")
    seq_len = full_meta.pop("_seq_len")
    head_dim = full_meta.pop("_head_dim")

    data = buffer[meta_end:]

    return CompressedKVCache(
        data=data,
        metadata=full_meta,
        original_size_bytes=original_size,
        compressed_size_bytes=compressed_size,
        num_layers=num_layers,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
    )


def wire_size(compressed: CompressedKVCache) -> int:
    """Calculate the total wire size without actually serializing.

    Args:
        compressed: The compressed KV cache.

    Returns:
        Estimated total bytes on the wire.
    """
    meta_bytes = json.dumps(compressed.metadata).encode("utf-8")
    return _HEADER_SIZE + len(meta_bytes) + len(compressed.data)
