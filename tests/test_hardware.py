from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from opentinker._hardware import HardwareManager


def manager(run: Callable[..., Any]) -> HardwareManager:
    return HardwareManager(
        provider="beam",
        profile="prod3",
        gpu="H100",
        gpu_count=4,
        pool="training",
        on_demand=False,
        machine_ttl="1h",
        machine_name=None,
        release_machine=True,
        run=run,
        find_executable=lambda _name: "/usr/local/bin/beam",
    )


def test_pool_capacity_uses_one_machine_not_pool_total() -> None:
    def run(_command: list[str], **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "machines": [
                        {"pool_name": "training", "gpu": "H100,H100", "gpu_count": 2},
                        {"pool_name": "training", "gpu": "H100,H100", "gpu_count": 2},
                    ]
                }
            ),
            stderr="",
        )

    hardware = manager(run)

    with pytest.raises(RuntimeError, match="at most 2 GPUs on one machine"):
        hardware.select()


def test_failed_release_keeps_ownership_for_retry() -> None:
    results = iter(
        (
            SimpleNamespace(returncode=1, stdout="", stderr="busy"),
            SimpleNamespace(returncode=0, stdout="released", stderr=""),
        )
    )
    hardware = manager(lambda *_args, **_kwargs: next(results))
    hardware.owned_pool = "training"

    assert hardware.release() is False
    assert hardware.owned_pool == "training"
    assert hardware.release() is True
    assert hardware.owned_pool is None


def test_operator_command_includes_selected_profile() -> None:
    hardware = manager(lambda *_args, **_kwargs: None)

    assert (
        hardware.operator_command("container", "attach", "pod-123")
        == "beam --context prod3 container attach pod-123"
    )
