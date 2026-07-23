# pyright: reportMissingImports=false

"""Remote OpenTinker server composition root.

Protocol routing and GPU execution live in separate modules for independent
tests. Run this module with ``python -m opentinker._server``.
"""

from __future__ import annotations

import os
from datetime import timedelta

from ._api import ComputeEngine, FutureStore, create_app
from ._distributed import (
    DistributedEngine,
    run_worker,
    validate_cuda_topology,
)
from ._engine import TransformersEngine


def main() -> None:
    """Run the server from the environment configured by the adapter."""

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - remote image contract
        raise ImportError("The Beam compute server requires uvicorn") from exc
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - remote image contract
        raise ImportError("The Beam compute server requires PyTorch") from exc
    base_model = os.environ["OPENTINKER_BASE_MODEL"]
    port = int(os.environ.get("OPENTINKER_PORT", "8000"))
    expected_gpu_count = int(os.environ.get("OPENTINKER_GPU_COUNT", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if expected_gpu_count > 1 and world_size != expected_gpu_count:
        raise RuntimeError(
            f"OpenTinker requested {expected_gpu_count} GPUs but torchrun started "
            f"{world_size} processes"
        )
    control_group = None
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        # Stateful API calls may be minutes or hours apart. Keep command traffic
        # off NCCL so idle worker ranks do not trip its collective watchdog.
        control_group = torch.distributed.new_group(
            backend="gloo",
            timeout=timedelta(days=7),
        )
    engine = TransformersEngine(
        base_model=base_model,
        checkpoint_root=os.environ.get("OPENTINKER_CHECKPOINT_ROOT", "/tinker-data"),
        volume_name=os.environ.get("OPENTINKER_VOLUME_NAME", "tinker-checkpoints"),
        max_length=int(os.environ.get("OPENTINKER_MAX_LENGTH", "8192")),
        trust_remote_code=os.environ.get("OPENTINKER_TRUST_REMOTE_CODE") == "1",
        sampling_gpu=os.environ.get("OPENTINKER_SAMPLING_GPU", "1") == "1",
        device=f"cuda:{local_rank}",
        distributed_rank=rank,
        distributed_world_size=world_size,
    )
    topology = validate_cuda_topology(
        torch,
        expected_gpu_count=expected_gpu_count,
        interconnect=os.environ.get("OPENTINKER_INTERCONNECT", "auto"),
    )
    engine.configure_runtime(topology)
    if rank != 0:
        run_worker(engine, torch, control_group=control_group)
        return
    compute_engine: ComputeEngine = (
        DistributedEngine(engine, torch, topology, control_group=control_group)
        if world_size > 1
        else engine
    )
    server_ref: list[uvicorn.Server] = []

    def request_shutdown() -> None:
        server_ref[0].should_exit = True

    app = create_app(compute_engine, request_shutdown=request_shutdown)
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port))
    server_ref.append(server)
    server.run()


__all__ = ["ComputeEngine", "FutureStore", "TransformersEngine", "create_app", "main"]


if __name__ == "__main__":
    main()
