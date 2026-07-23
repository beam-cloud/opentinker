"""Beam/Beta9 SDK loading and profile activation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class ProviderRuntime:
    module: ModuleType
    context: Any | None


def load_provider(name: str, *, profile: str | None) -> ProviderRuntime:
    """Load a compatible provider module and activate its selected profile."""

    try:
        provider = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        raise ImportError(f"{name} is required; install `opentinker[{name}]`") from exc

    context: Any | None = None
    if profile is not None:
        from beta9.abstractions.base import set_channel
        from beta9.config import get_config_context

        context = get_config_context(profile)
        if context is None:
            raise ValueError(f"unknown Beta9 profile: {profile}")
        set_channel(context=context)

    missing = [symbol for symbol in ("Image", "Pod", "Volume") if not hasattr(provider, symbol)]
    if missing:
        raise ImportError(
            f"installed {name} package is incompatible (missing: {', '.join(missing)})"
        )
    return ProviderRuntime(module=provider, context=context)


__all__ = ["ProviderRuntime", "load_provider"]
