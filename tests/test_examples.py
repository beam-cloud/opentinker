from __future__ import annotations

import argparse

import pytest

from opentinker import BeamComputeAdapter
from opentinker._examples import add_compute_arguments, compute_options_from_args


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--max-length", type=int, default=512)
    add_compute_arguments(value, machine_ttl="2h")
    return value


def test_example_compute_options_build_a_consistent_adapter() -> None:
    args = parser().parse_args(
        [
            "--provider",
            "beta9",
            "--profile",
            "prod3",
            "--gpu",
            "A16",
            "--gpu-count",
            "4",
            "--interconnect",
            "nvlink",
            "--pool",
            "training",
            "--volume-name",
            "checkpoints",
        ]
    )

    adapter = BeamComputeAdapter(
        base_model="Qwen/Qwen3-0.6B",
        **compute_options_from_args(args),
    )

    assert adapter.provider == "beta9"
    assert adapter.profile == "prod3"
    assert adapter.gpu == "A16"
    assert adapter.gpu_count == 4
    assert adapter.interconnect == "nvlink"
    assert adapter.pool == "training"
    assert adapter.machine_ttl == "2h"
    assert adapter.volume_name == "checkpoints"
    assert adapter.max_length == 512
    assert adapter.sampling_gpu is False


def test_on_demand_example_defers_gpu_selection_to_beam() -> None:
    args = parser().parse_args(["--on-demand"])

    adapter = BeamComputeAdapter(
        base_model="Qwen/Qwen3-0.6B",
        **compute_options_from_args(args),
    )

    assert adapter.on_demand is True
    assert adapter.gpu is None


def test_private_pool_example_defers_gpu_discovery_to_beam() -> None:
    args = parser().parse_args(["--pool", "my-hardware"])

    adapter = BeamComputeAdapter(
        base_model="Qwen/Qwen3-0.6B",
        **compute_options_from_args(args),
    )

    assert adapter.pool == "my-hardware"
    assert adapter.gpu is None


def test_pool_and_on_demand_are_distinct_cli_modes() -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(["--pool", "my-hardware", "--on-demand"])
