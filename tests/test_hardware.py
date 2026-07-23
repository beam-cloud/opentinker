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


def test_headless_multi_gpu_reservation_selects_a_fitting_offer() -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        if "offers" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "offers": [
                            {
                                "id": "L40S",
                                "provider": "shadeform",
                                "region": "cheap",
                                "gpu_count": 1,
                                "available": 2,
                                "hourly_cost_micros": 968000,
                            },
                            {
                                "id": "L40Sx4",
                                "provider": "shadeform",
                                "region": "four-gpu",
                                "gpu_count": 4,
                                "available": 1,
                                "hourly_cost_micros": 3872000,
                            },
                            {
                                "id": "L40Sx8",
                                "provider": "shadeform",
                                "region": "eight-gpu",
                                "gpu_count": 8,
                                "available": 1,
                                "hourly_cost_micros": 7744000,
                            },
                        ]
                    }
                ),
                stderr="",
            )
        if "list" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "machines": [
                            {
                                "pool_name": "ddp-test",
                                "gpu": "L40S,L40S,L40S,L40S",
                                "gpu_count": 4,
                            }
                        ]
                    }
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    hardware = HardwareManager(
        provider="beam",
        profile="prod3",
        gpu="L40S",
        gpu_count=4,
        pool=None,
        on_demand=True,
        machine_ttl="2h",
        machine_name="ddp-test",
        release_machine=True,
        run=run,
        find_executable=lambda _name: "/usr/local/bin/beam",
        interactive=False,
        sleep=lambda _seconds: None,
    )

    selection = hardware.select()

    assert selection.gpu == "L40S"
    assert selection.pool == "ddp-test"
    scale = commands[1]
    assert scale[:6] == [
        "/usr/local/bin/beam",
        "--context",
        "prod3",
        "pool",
        "scale",
        "ddp-test",
    ]
    assert scale[scale.index("--offer-id") + 1] == "L40Sx4"
    assert scale[scale.index("--max-spend") + 1] == "15.49"
    assert scale[scale.index("--region") + 1] == "four-gpu"
