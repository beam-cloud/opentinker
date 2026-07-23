"""Shared command-line plumbing for the runnable examples.

These helpers cover infrastructure options only. Each example keeps its Tinker
training loop inline.
"""

from __future__ import annotations

import argparse
from typing import Any

from tinker_cookbook.supervised.common import compute_mean_nll


def add_compute_arguments(
    parser: argparse.ArgumentParser,
    *,
    machine_ttl: str = "1h",
) -> None:
    """Add the consistent Beam/Beta9 options used by every example."""

    group = parser.add_argument_group("Beam compute")
    group.add_argument("--provider", choices=("beam", "beta9"), default="beam")
    group.add_argument("--profile", help="Beam/Beta9 profile from ~/.beta9/config.ini")
    group.add_argument(
        "--gpu",
        help="GPU type (default: A10G; auto-detected for --pool or an unfiltered picker)",
    )
    group.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="GPUs on one machine; values above 1 use PyTorch DDP and NCCL",
    )
    group.add_argument(
        "--interconnect",
        choices=("auto", "nvlink"),
        default="auto",
        help="auto-detect links, or require all allocated GPUs to use NVLink/NVSwitch",
    )
    group.add_argument("--cpu", type=float, default=4, help="Pod CPU cores")
    group.add_argument("--memory", default="16Gi", help="Pod memory, for example 64Gi")
    hardware = group.add_mutually_exclusive_group()
    hardware.add_argument(
        "--pool",
        help="existing reserved or private hardware pool (GPU auto-detected)",
    )
    hardware.add_argument(
        "--on-demand",
        action="store_true",
        help="open Beam's machine picker and release the reservation when finished",
    )
    group.add_argument("--machine-ttl", default=machine_ttl, help="on-demand reservation TTL")
    group.add_argument("--volume-name", default="tinker-checkpoints")


def compute_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return adapter options registered by :func:`add_compute_arguments`."""

    return {
        "provider": args.provider,
        "profile": args.profile,
        "gpu": args.gpu,
        "gpu_count": args.gpu_count,
        "interconnect": args.interconnect,
        "cpu": args.cpu,
        "memory": args.memory,
        "pool": args.pool,
        "on_demand": args.on_demand,
        "machine_ttl": args.machine_ttl,
        "volume_name": args.volume_name,
        "sampling_gpu": False,
        "max_length": args.max_length,
    }


def mean_nll(result: Any, batch: list[Any]) -> float:
    """Compute the cookbook's token-weighted mean negative log likelihood."""

    return compute_mean_nll(
        [item["logprobs"] for item in result.loss_fn_outputs],
        [datum.loss_fn_inputs["weights"] for datum in batch],
    )
