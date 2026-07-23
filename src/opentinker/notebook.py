"""Notebook-friendly lifecycle helpers.

Start one Beam-backed Tinker session in a setup cell::

    from opentinker.notebook import start

    adapter = start(base_model="Qwen/Qwen3-8B", profile="prod3")

The remaining tutorial cells can keep using ``tinker.ServiceClient()``. Re-running
the same setup cell returns the active adapter instead of creating another Pod.
Finish gracefully in a later cell with ``finish()``, or cancel with ``stop()``.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .adapter import BeamComputeAdapter

_lock = threading.RLock()
_active_adapter: BeamComputeAdapter | None = None
_active_signature: object | None = None


def _snapshot(value: Any) -> object:
    """Create a stable, non-rendered signature for notebook configuration."""

    if isinstance(value, Mapping):
        return (
            "mapping",
            frozenset((_snapshot(key), _snapshot(item)) for key, item in value.items()),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ("sequence", tuple(_snapshot(item) for item in value))
    if isinstance(value, set | frozenset):
        return ("set", frozenset(_snapshot(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return ("identity", id(value))
    return ("value", value)


def start(
    base_model: str,
    *,
    wait: bool = True,
    **options: Any,
) -> BeamComputeAdapter:
    """Start or reuse the process-wide notebook backend.

    ``options`` are passed to :class:`BeamComputeAdapter`. An exact rerun is
    idempotent. A different configuration is rejected while a session is active;
    call :func:`finish` or :func:`stop` before changing hardware or model options.
    """

    global _active_adapter, _active_signature

    configuration = {"base_model": base_model, **options}
    signature = _snapshot((configuration, wait))
    with _lock:
        active = _active_adapter
        if active is not None and active._notebook_is_running:
            if signature != _active_signature:
                raise RuntimeError(
                    "a different OpenTinker notebook session is already running; "
                    "call opentinker.notebook.finish() or stop() before changing "
                    "the model or hardware configuration"
                )
            return active.start_notebook(wait=wait)

        _active_adapter = None
        _active_signature = None
        adapter = cast(Any, BeamComputeAdapter)(**configuration)
        adapter.start_notebook(wait=wait)
        _active_adapter = adapter
        _active_signature = signature
        return adapter


def finish() -> bool:
    """Flush checkpoints and let the active notebook Pod exit successfully."""

    return _close(graceful=True)


def stop() -> bool:
    """Cancel the active notebook Pod after flushing completed checkpoints."""

    return _close(graceful=False)


def _close(*, graceful: bool) -> bool:
    global _active_adapter, _active_signature

    with _lock:
        adapter = _active_adapter
        if adapter is None:
            return True
        try:
            return adapter.finish() if graceful else adapter.stop()
        finally:
            if not adapter._notebook_is_running:
                _active_adapter = None
                _active_signature = None


__all__ = ["finish", "start", "stop"]
