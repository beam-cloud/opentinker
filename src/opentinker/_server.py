# pyright: reportMissingImports=false

"""Remote OpenTinker server composition root.

Protocol routing and GPU execution live in separate modules for independent
tests. Run this module with ``python -m opentinker._server``.
"""

from __future__ import annotations

import os

from ._api import ComputeEngine, FutureStore, create_app
from ._engine import TransformersEngine


def main() -> None:
    """Run the server from the environment configured by the adapter."""

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - remote image contract
        raise ImportError("The Beam compute server requires uvicorn") from exc
    base_model = os.environ["OPENTINKER_BASE_MODEL"]
    port = int(os.environ.get("OPENTINKER_PORT", "8000"))
    engine = TransformersEngine(
        base_model=base_model,
        checkpoint_root=os.environ.get("OPENTINKER_CHECKPOINT_ROOT", "/tinker-data"),
        volume_name=os.environ.get("OPENTINKER_VOLUME_NAME", "tinker-checkpoints"),
        volume_sync_seconds=float(os.environ.get("OPENTINKER_VOLUME_SYNC_SECONDS", "60")),
        max_length=int(os.environ.get("OPENTINKER_MAX_LENGTH", "8192")),
        trust_remote_code=os.environ.get("OPENTINKER_TRUST_REMOTE_CODE") == "1",
        sampling_gpu=os.environ.get("OPENTINKER_SAMPLING_GPU", "1") == "1",
    )
    uvicorn.run(create_app(engine), host="0.0.0.0", port=port)


__all__ = ["ComputeEngine", "FutureStore", "TransformersEngine", "create_app", "main"]


if __name__ == "__main__":
    main()
