"""Registry for looking up compressors by name."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kvshuttle.compression.base import BaseCompressor

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseCompressor]] = {}


def register(name: str):
    """Decorator to register a compressor class under a given name.

    Args:
        name: The string identifier for this compressor.

    Usage::

        @register("my_compressor")
        class MyCompressor(BaseCompressor):
            ...
    """

    def decorator(cls: type[BaseCompressor]) -> type[BaseCompressor]:
        if name in _REGISTRY:
            logger.warning("Overwriting compressor %r in registry", name)
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_compressor(name: str, **kwargs) -> BaseCompressor:
    """Instantiate a compressor by its registered name.

    Args:
        name: Registered compressor name.
        **kwargs: Arguments forwarded to the compressor constructor.

    Returns:
        An instance of the requested compressor.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown compressor {name!r}. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_compressors() -> list[str]:
    """Return sorted list of all registered compressor names."""
    return sorted(_REGISTRY.keys())


def _ensure_builtins_loaded() -> None:
    """Import built-in compressor modules to trigger registration."""
    import kvshuttle.compression.identity  # noqa: F401

    try:
        import kvshuttle.compression.lossless  # noqa: F401
    except ImportError:
        pass
    try:
        import kvshuttle.compression.fp8  # noqa: F401
    except ImportError:
        pass
    try:
        import kvshuttle.compression.uniform_quant  # noqa: F401
    except ImportError:
        pass
    try:
        import kvshuttle.compression.kivi  # noqa: F401
    except ImportError:
        pass
    try:
        import kvshuttle.compression.cachegen  # noqa: F401
    except ImportError:
        pass
    try:
        import kvshuttle.compression.kvquant  # noqa: F401
    except ImportError:
        pass
    try:
        import kvshuttle.compression.low_rank  # noqa: F401
    except ImportError:
        pass
    try:
        import kvshuttle.compression.pruning  # noqa: F401
    except ImportError:
        pass
    try:
        import kvshuttle.compression.cascade  # noqa: F401
    except ImportError:
        pass
    try:
        import kvshuttle.compression.hybrid  # noqa: F401
    except ImportError:
        pass
    try:
        import kvshuttle.compression.mlx_quant  # noqa: F401
    except ImportError:
        pass
    try:
        import kvshuttle.compression.torch_quant  # noqa: F401
    except ImportError:
        pass


_ensure_builtins_loaded()
