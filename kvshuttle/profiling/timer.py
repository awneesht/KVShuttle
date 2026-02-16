"""High-resolution timing utilities with warmup support."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Result of a timed operation.

    Attributes:
        elapsed_ms: Elapsed time in milliseconds.
        name: Optional label for the operation.
    """

    elapsed_ms: float
    name: str = ""

    def __repr__(self) -> str:
        label = f" ({self.name})" if self.name else ""
        return f"TimingResult{label}: {self.elapsed_ms:.3f} ms"


@contextmanager
def timer(name: str = ""):
    """Context manager that yields a TimingResult populated on exit.

    Usage::

        with timer("compress") as t:
            result = compressor.compress(keys, values)
        print(t.elapsed_ms)

    Args:
        name: Optional label for logging.

    Yields:
        TimingResult that is populated with elapsed_ms upon context exit.
    """
    result = TimingResult(elapsed_ms=0.0, name=name)
    start = time.perf_counter_ns()
    try:
        yield result
    finally:
        elapsed_ns = time.perf_counter_ns() - start
        result.elapsed_ms = elapsed_ns / 1_000_000
        if name:
            logger.debug("%s: %.3f ms", name, result.elapsed_ms)


def benchmark(
    fn,
    *args,
    warmup: int = 3,
    repeats: int = 10,
    name: str = "",
    **kwargs,
) -> list[TimingResult]:
    """Run a function multiple times with warmup and return timing results.

    Args:
        fn: Callable to benchmark.
        *args: Positional arguments for fn.
        warmup: Number of warmup iterations (not timed).
        repeats: Number of timed iterations.
        name: Optional label.
        **kwargs: Keyword arguments for fn.

    Returns:
        List of TimingResult for each timed iteration.
    """
    for _ in range(warmup):
        fn(*args, **kwargs)

    results = []
    for i in range(repeats):
        with timer(f"{name}[{i}]" if name else "") as t:
            fn(*args, **kwargs)
        results.append(t)

    if name and results:
        times = [r.elapsed_ms for r in results]
        median = sorted(times)[len(times) // 2]
        logger.info(
            "%s: median=%.3f ms, min=%.3f ms, max=%.3f ms (%d runs)",
            name,
            median,
            min(times),
            max(times),
            repeats,
        )

    return results
