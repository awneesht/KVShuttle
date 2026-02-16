"""Peak memory tracking utilities."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot.

    Attributes:
        peak_bytes: Peak memory usage in bytes during the tracked operation.
        label: Optional label for the operation.
    """

    peak_bytes: int = 0
    label: str = ""

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)

    def __repr__(self) -> str:
        tag = f" ({self.label})" if self.label else ""
        return f"MemorySnapshot{tag}: {self.peak_mb:.1f} MB"


@contextmanager
def track_metal_memory(label: str = ""):
    """Track peak Metal (GPU) memory during a block.

    Requires MLX. Falls back gracefully if unavailable.

    Yields:
        MemorySnapshot populated on exit.
    """
    snapshot = MemorySnapshot(label=label)
    try:
        import mlx.core as mx

        mx.metal.reset_peak_memory()
        yield snapshot
        snapshot.peak_bytes = mx.metal.get_peak_memory()
        if label:
            logger.debug("%s: peak Metal memory = %.1f MB", label, snapshot.peak_mb)
    except (ImportError, AttributeError):
        logger.debug("MLX Metal memory tracking unavailable")
        yield snapshot
