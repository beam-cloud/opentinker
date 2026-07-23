from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from opentinker._distributed import (
    inspect_cuda_topology,
    parse_nvidia_smi_topology,
    validate_cuda_topology,
)

NVSWITCH_TOPOLOGY = """\
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV18    NV18    NV18    0-47            0
GPU1    NV18     X      NV18    NV18    0-47            0
GPU2    NV18    NV18     X      NV18    48-95           1
GPU3    NV18    NV18    NV18     X      48-95           1
"""


def fake_torch(count: int, *, peer_access: bool = True) -> Any:
    cuda = SimpleNamespace(
        is_available=lambda: count > 0,
        device_count=lambda: count,
        get_device_name=lambda index: f"Test GPU {index}",
        can_device_access_peer=lambda _left, _right: peer_access,
    )
    return SimpleNamespace(cuda=cuda)


def test_parses_all_nvlink_pairs_from_nvidia_smi() -> None:
    assert parse_nvidia_smi_topology(NVSWITCH_TOPOLOGY, device_count=4) == {
        (0, 1): "NV18",
        (0, 2): "NV18",
        (0, 3): "NV18",
        (1, 2): "NV18",
        (1, 3): "NV18",
        (2, 3): "NV18",
    }


def test_reports_fully_connected_nvlink_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "opentinker._distributed._nvidia_smi_links",
        lambda _count: parse_nvidia_smi_topology(NVSWITCH_TOPOLOGY, device_count=4),
    )

    topology = inspect_cuda_topology(fake_torch(4))

    assert topology["interconnect"] == "nvlink"
    assert topology["nvlink_pairs"] == 6
    assert topology["peer_access_pairs"] == 6


def test_nvlink_policy_rejects_pcie_only_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "opentinker._distributed._nvidia_smi_links",
        lambda _count: {(0, 1): "SYS"},
    )

    with pytest.raises(RuntimeError, match="detected 0/1 pairs"):
        validate_cuda_topology(
            fake_torch(2),
            expected_gpu_count=2,
            interconnect="nvlink",
        )


def test_visible_gpu_count_must_match_beta9_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("opentinker._distributed._nvidia_smi_links", lambda _count: {})

    with pytest.raises(RuntimeError, match="allocated 8 GPUs, but PyTorch sees 4"):
        validate_cuda_topology(
            fake_torch(4),
            expected_gpu_count=8,
            interconnect="auto",
        )
