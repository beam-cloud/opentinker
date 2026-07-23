"""Beam/Beta9 machine discovery and reservation lifecycle."""

from __future__ import annotations

import json
import logging
import shlex
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

GpuRequest = str | Sequence[str] | None
CommandRunner = Callable[..., Any]
ExecutableFinder = Callable[[str], str | None]


@dataclass(frozen=True)
class HardwareSelection:
    """Resources to pass to a Beam/Beta9 Pod."""

    gpu: str | Sequence[str]
    pool: str | None


class HardwareManager:
    """Resolve serverless hardware or own one on-demand machine reservation."""

    def __init__(
        self,
        *,
        provider: str,
        profile: str | None,
        gpu: GpuRequest,
        gpu_count: int,
        pool: str | None,
        on_demand: bool,
        machine_ttl: str,
        machine_name: str | None,
        release_machine: bool,
        run: CommandRunner,
        find_executable: ExecutableFinder,
    ) -> None:
        self.provider = provider
        self.profile = profile
        self.gpu = gpu
        self.gpu_count = gpu_count
        self.pool = pool
        self.on_demand = on_demand
        self.machine_ttl = machine_ttl
        self.machine_name = machine_name
        self.release_machine = release_machine
        self._run = run
        self._find_executable = find_executable
        self.owned_pool: str | None = None

    def select(self) -> HardwareSelection:
        """Resolve the GPU and optional pool for one Pod."""

        if self.on_demand:
            return self._reserve()
        if self.pool is not None:
            pool_gpu, available = self.inspect_pool(self.pool)
            if pool_gpu is None:
                raise RuntimeError(
                    f"pool {self.pool!r} has no connected machines; attach hardware with "
                    f"`{self.provider} pool join {self.pool}` or choose another pool"
                )
            if self.gpu is None:
                self.gpu = pool_gpu
            elif isinstance(self.gpu, str) and self.gpu != pool_gpu:
                raise RuntimeError(
                    f"pool {self.pool!r} provides {pool_gpu}, but OpenTinker requested {self.gpu}"
                )
            self._require_capacity(self.pool, available)
        if self.gpu is None:
            raise RuntimeError("no GPU was selected for the Beam/Beta9 Pod")
        return HardwareSelection(gpu=self.gpu, pool=self.pool)

    def operator_command(self, *arguments: str) -> str:
        """Render a copyable provider command without requiring CLI discovery."""

        command = [self.provider, *self._context_arguments(), *arguments]
        return shlex.join(command)

    def inspect_pool(self, pool: str) -> tuple[str | None, int]:
        """Return a pool's GPU type and largest single-machine GPU count."""

        command = self._command(
            "machine",
            "list",
            "--pool",
            pool,
            "--no-offers",
            "--format",
            "json",
        )
        result = self._run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"could not inspect Beam/Beta9 pool {pool!r}: {detail}")
        try:
            payload = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Beam/Beta9 returned invalid JSON while inspecting the pool"
            ) from exc
        machines = payload.get("machines")
        if not isinstance(machines, list):
            raise RuntimeError("Beam's machine list response did not contain a machine list")
        matched = [
            machine
            for machine in machines
            if isinstance(machine, dict) and machine.get("pool_name") == pool
        ]
        if not matched:
            return None, 0
        gpu_types = {
            gpu.strip()
            for machine in matched
            for gpu in str(machine.get("gpu") or "").split(",")
            if gpu.strip()
        }
        if not gpu_types:
            raise RuntimeError("OpenTinker requires a GPU, but the selected machine is CPU-only")
        if len(gpu_types) != 1:
            raise RuntimeError(
                f"Beam/Beta9 pool {pool!r} contains multiple GPU types; pass gpu= explicitly"
            )
        gpu_counts = [
            int(machine.get("gpu_count") or 1) for machine in matched if machine.get("gpu")
        ]
        return gpu_types.pop(), max(gpu_counts, default=0)

    def release(self) -> bool:
        """Release the reservation owned by this manager, if configured to do so."""

        pool = self.owned_pool
        if pool is None or not self.release_machine:
            return True
        result = self._run(
            self._command("machine", "release", "--pool", pool, "--yes"),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            logger.error("Failed to release on-demand Beam pool %s: %s", pool, detail)
            return False
        self.owned_pool = None
        logger.info("Released on-demand Beam pool %s", pool)
        return True

    def _reserve(self) -> HardwareSelection:
        if self.gpu is not None and not isinstance(self.gpu, str):
            raise ValueError("on-demand reservations accept one GPU type or None")
        if self.owned_pool is not None:
            raise RuntimeError("this adapter already owns an on-demand machine")

        requested_gpu = self.gpu
        original_pool = self.pool
        gpu_label = requested_gpu.lower() if requested_gpu is not None else "ondemand"
        pool = self.pool or self.machine_name or f"opentinker-{gpu_label}-{uuid.uuid4().hex[:8]}"
        arguments = [
            "machine",
            "reserve",
            "--nodes",
            "1",
            "--ttl",
            self.machine_ttl,
            "--name",
            pool,
        ]
        if requested_gpu is not None:
            arguments.extend(["--gpu", requested_gpu])

        self.pool = pool
        self.owned_pool = pool
        try:
            # The provider CLI opens its interactive offer picker on a TTY.
            result = self._run(self._command(*arguments), check=False)
        except BaseException:
            self.release()
            raise
        if result.returncode != 0:
            self.release()
            raise RuntimeError("failed to reserve an on-demand machine; see Beam's output above")

        try:
            selected_gpu, available = self.inspect_pool(pool)
            if selected_gpu is None:
                # Declining the interactive picker is a successful CLI exit with no pool.
                self.owned_pool = None
                self.pool = original_pool
                raise RuntimeError("on-demand reservation was cancelled; no machine was created")
            if requested_gpu is not None and selected_gpu != requested_gpu:
                raise RuntimeError(
                    f"Beam reserved {selected_gpu}, but OpenTinker requested {requested_gpu}"
                )
            self._require_capacity(pool, available)
        except BaseException:
            self.release()
            raise

        self.gpu = selected_gpu
        logger.info(
            "Reserved on-demand Beam pool %s with %sx %s",
            pool,
            available,
            selected_gpu,
        )
        return HardwareSelection(gpu=selected_gpu, pool=pool)

    def _require_capacity(self, pool: str, available: int) -> None:
        if available < self.gpu_count:
            noun = "GPU" if available == 1 else "GPUs"
            raise RuntimeError(
                f"pool {pool!r} has at most {available} {noun} on one machine, "
                f"but OpenTinker requested {self.gpu_count}; a multi-GPU Pod cannot "
                "span machines"
            )

    def _command(self, *arguments: str) -> list[str]:
        executable = self._find_executable(self.provider)
        if executable is None:
            raise RuntimeError(f"could not find the {self.provider!r} CLI")
        return [executable, *self._context_arguments(), *arguments]

    def _context_arguments(self) -> list[str]:
        return ["--context", self.profile] if self.profile is not None else []


__all__ = ["HardwareManager", "HardwareSelection"]
