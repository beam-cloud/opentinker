"""Tinker-compatible training backed by Beam and Beta9 GPUs."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import tinker as _tinker

from .adapter import BeamComputeAdapter, ProviderName

try:
    __version__ = version("opentinker")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0"


def __getattr__(name: str) -> Any:
    """Delegate Tinker's public API so ``import opentinker as tinker`` works."""

    return getattr(_tinker, name)


def __dir__() -> list[str]:
    return sorted({*globals(), *dir(_tinker)})


__all__ = ["BeamComputeAdapter", "ProviderName"]
