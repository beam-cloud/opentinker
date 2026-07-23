# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

"""Single-node distributed execution for the OpenTinker compute engine."""

from __future__ import annotations

import random
import re
import subprocess
import threading
import traceback
import uuid
from typing import Any


def inspect_cuda_topology(torch: Any) -> dict[str, Any]:
    """Describe the CUDA devices and links visible inside the container."""

    count = int(torch.cuda.device_count())
    names = [str(torch.cuda.get_device_name(index)) for index in range(count)]
    links = _nvidia_smi_links(count)
    pairs = count * (count - 1) // 2
    nvlink_pairs = sum(1 for link in links.values() if link.startswith("NV"))
    if count <= 1:
        interconnect = "single"
    elif len(links) != pairs:
        interconnect = "unknown"
    elif nvlink_pairs == pairs:
        interconnect = "nvlink"
    elif nvlink_pairs:
        interconnect = "mixed"
    else:
        interconnect = "pcie"
    peer_pairs = 0
    for left in range(count):
        for right in range(left + 1, count):
            if torch.cuda.can_device_access_peer(left, right):
                peer_pairs += 1
    return {
        "gpu_count": count,
        "gpu_names": names,
        "interconnect": interconnect,
        "nvlink_pairs": nvlink_pairs,
        "peer_access_pairs": peer_pairs,
        "pair_count": pairs,
        "links": {f"{left}-{right}": link for (left, right), link in sorted(links.items())},
    }


def validate_cuda_topology(
    torch: Any,
    *,
    expected_gpu_count: int,
    interconnect: str,
) -> dict[str, Any]:
    """Validate allocation size and an optional all-pairs NVLink requirement."""

    if not torch.cuda.is_available():
        raise RuntimeError("OpenTinker requires a CUDA GPU")
    topology = inspect_cuda_topology(torch)
    visible = int(topology["gpu_count"])
    if visible != expected_gpu_count:
        raise RuntimeError(
            f"Beta9 allocated {expected_gpu_count} GPUs, but PyTorch sees {visible}; "
            "check the machine's GPU inventory and container runtime configuration"
        )
    if interconnect == "nvlink" and expected_gpu_count > 1:
        pair_count = int(topology["pair_count"])
        nvlink_pairs = int(topology["nvlink_pairs"])
        if topology["interconnect"] == "unknown":
            raise RuntimeError(
                "interconnect='nvlink' requires nvidia-smi topology data, but the "
                "container could not inspect every GPU link"
            )
        if nvlink_pairs != pair_count:
            raise RuntimeError(
                "interconnect='nvlink' requires direct NVLink/NVSwitch connectivity "
                f"between every allocated GPU pair; detected {nvlink_pairs}/{pair_count} pairs"
            )
    return topology


def _nvidia_smi_links(device_count: int) -> dict[tuple[int, int], str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    return parse_nvidia_smi_topology(result.stdout, device_count=device_count)


def parse_nvidia_smi_topology(
    output: str,
    *,
    device_count: int,
) -> dict[tuple[int, int], str]:
    """Parse the GPU-to-GPU portion of ``nvidia-smi topo -m``."""

    header: list[int] = []
    links: dict[tuple[int, int], str] = {}
    for raw_line in output.splitlines():
        columns = raw_line.split()
        if not columns:
            continue
        if not header:
            gpu_columns = []
            for value in columns:
                if re.fullmatch(r"GPU\d+", value) is None:
                    break
                gpu_columns.append(value)
            if gpu_columns:
                header = [int(value.removeprefix("GPU")) for value in gpu_columns]
                continue
        match = re.fullmatch(r"GPU(\d+)", columns[0])
        if match is None or not header:
            continue
        row = int(match.group(1))
        values = columns[1 : 1 + len(header)]
        for column, link in zip(header, values, strict=False):
            if row >= device_count or column >= device_count or row >= column:
                continue
            links[(row, column)] = link
    return links


class DistributedEngine:
    """Rank-zero facade that executes stateful engine operations on every rank."""

    def __init__(
        self,
        local_engine: Any,
        torch: Any,
        topology: dict[str, Any],
        *,
        control_group: Any = None,
    ) -> None:
        self._local = local_engine
        self._torch = torch
        self._dist = torch.distributed
        self._control_group = control_group
        self._world_size = int(self._dist.get_world_size())
        self._lock = threading.RLock()
        self._closed = False
        self.base_model = local_engine.base_model
        self.max_length = local_engine.max_length
        self.runtime_info = {
            **topology,
            "distributed": True,
            "strategy": "ddp",
            "collective_backend": str(self._dist.get_backend()),
        }

    def create_model(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._first(self._dispatch("create_model", request))

    def get_info(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._local.get_info(request)

    def weights_info(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._local.weights_info(request)

    def forward_backward(self, request: dict[str, Any], *, backward: bool) -> dict[str, Any]:
        results = self._dispatch("forward_backward", request, backward=backward)
        payload = request.get("forward_backward_input") or request.get("forward_input") or {}
        data = payload.get("data") or []
        outputs: list[dict[str, Any] | None] = [None] * len(data)
        for result in results:
            indices = result.pop("_distributed_indices", [])
            for index, output in zip(indices, result["loss_fn_outputs"], strict=True):
                outputs[int(index)] = output
        if any(output is None for output in outputs):
            raise RuntimeError("distributed forward pass did not return every datum")
        merged = dict(results[0])
        merged["loss_fn_outputs"] = outputs
        return merged

    def optim_step(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._first(self._dispatch("optim_step", request))

    def save_weights(self, request: dict[str, Any], *, for_sampler: bool) -> dict[str, Any]:
        result = self._local.save_weights(request, for_sampler=for_sampler)
        session_id = result.get("sampling_session_id")
        if session_id is not None:
            state = self._local.sampling_session(str(session_id))
            self._dispatch(
                "register_sampling_session",
                {"sampling_session_id": session_id, "state": state},
                skip_rank_zero=True,
            )
        return result

    def load_weights(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._first(self._dispatch("load_weights", request))

    def unload_model(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._first(self._dispatch("unload_model", request))

    def create_sampling_session(self, request: dict[str, Any]) -> dict[str, Any]:
        request = {**request, "_sampling_session_id": str(uuid.uuid4())}
        return self._first(self._dispatch("create_sampling_session", request))

    def get_sampler(self, sampling_session_id: str) -> dict[str, Any]:
        return self._local.get_sampler(sampling_session_id)

    def sample(self, request: dict[str, Any]) -> dict[str, Any]:
        request = {
            **request,
            "_sampling_seed": int(
                request.get("_sampling_seed", random.SystemRandom().randrange(2**31))
            ),
        }
        results = self._dispatch("sample", request)
        count = int(request.get("num_samples", 1))
        sequences: list[dict[str, Any] | None] = [None] * count
        for result in results:
            indices = result.pop("_distributed_indices", [])
            for index, sequence in zip(indices, result["sequences"], strict=True):
                sequences[int(index)] = sequence
        if any(sequence is None for sequence in sequences):
            raise RuntimeError("distributed sampler did not return every sequence")
        merged = dict(results[0])
        merged["sequences"] = sequences
        return merged

    def prepare_shutdown(self) -> dict[str, Any]:
        return self._local.prepare_shutdown()

    def runtime_status(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        del request
        ranks = self._dispatch("runtime_status", {})
        return {
            **self.runtime_info,
            "ranks": sorted(ranks, key=lambda item: int(item["rank"])),
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._dispatch("__close__", {})
            finally:
                self._closed = True
                if self._dist.is_initialized():
                    if self._control_group is not None:
                        self._dist.destroy_process_group(self._control_group)
                    self._dist.destroy_process_group()

    def _dispatch(
        self,
        operation: str,
        request: dict[str, Any],
        *,
        skip_rank_zero: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if self._closed:
                raise RuntimeError("distributed engine is closed")
            command = {
                "operation": operation,
                "request": request,
                "kwargs": kwargs,
                "skip_rank_zero": skip_rank_zero,
            }
            objects: list[Any] = [command]
            self._dist.broadcast_object_list(objects, src=0, group=self._control_group)
            local = (
                {"ok": True, "value": None}
                if skip_rank_zero
                else _execute_command(self._local, command)
            )
            gathered: list[Any] = [None] * self._world_size
            self._dist.gather_object(local, gathered, dst=0, group=self._control_group)
            failures = [item for item in gathered if not item.get("ok")]
            if failures:
                failure = failures[0]
                raise RuntimeError(
                    f"distributed rank {failure['rank']} failed: {failure['error']}\n"
                    f"{failure['traceback']}"
                )
            return [item["value"] for item in gathered if item["value"] is not None]

    @staticmethod
    def _first(results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            raise RuntimeError("distributed operation returned no result")
        return results[0]


def run_worker(engine: Any, torch: Any, *, control_group: Any = None) -> None:
    """Run the command loop used by non-zero ``torchrun`` ranks."""

    dist = torch.distributed
    try:
        while True:
            objects: list[Any] = [None]
            dist.broadcast_object_list(objects, src=0, group=control_group)
            command = objects[0]
            if not isinstance(command, dict):
                result = _failure(RuntimeError("received an invalid distributed command"), dist)
            elif command.get("skip_rank_zero") and int(dist.get_rank()) == 0:
                result = {"ok": True, "value": None}
            else:
                result = _execute_command(engine, command)
            dist.gather_object(result, None, dst=0, group=control_group)
            if isinstance(command, dict) and command.get("operation") == "__close__":
                break
    finally:
        if dist.is_initialized():
            if control_group is not None:
                dist.destroy_process_group(control_group)
            dist.destroy_process_group()


def _execute_command(engine: Any, command: dict[str, Any]) -> dict[str, Any]:
    operation = str(command["operation"])
    request = command.get("request") or {}
    kwargs = command.get("kwargs") or {}
    try:
        if operation == "__close__":
            close = getattr(engine, "close", None)
            if close is not None:
                close()
            value = {"type": "close"}
        else:
            value = getattr(engine, operation)(request, **kwargs)
        return {"ok": True, "value": value}
    except BaseException as exc:
        return _failure(exc, engine._imports()[0].distributed)


def _failure(exc: BaseException, dist: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "rank": int(dist.get_rank()) if dist.is_initialized() else -1,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


__all__ = [
    "DistributedEngine",
    "inspect_cuda_topology",
    "parse_nvidia_smi_topology",
    "run_worker",
    "validate_cuda_topology",
]
