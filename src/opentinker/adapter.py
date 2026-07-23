# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnnecessaryIsInstance=false

"""Run ordinary Tinker workflows with model compute on Beam/Beta9 GPUs."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType, TracebackType
from typing import Any, Literal, Protocol, cast

import httpx
import tinker as tinker_module
from tinker.lib.public_interfaces.service_client import ServiceClient

from ._hardware import HardwareManager
from ._image import BackendImageSpec, build_backend_image
from ._provider import load_provider

ProviderName = Literal["beam", "beta9"]
InterconnectPolicy = Literal["auto", "nvlink"]
PoolFallback = Literal["wait", "fail"]

_DEFAULT_PORT = 8000
_DEFAULT_VOLUME_MOUNT = "/tinker-data"
_DEFAULT_BASE_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
_CLIENT_API_KEY = "tml-beam-compute"
_BEAM_CONTAINERS_DASHBOARD = "https://platform.beam.cloud/containers"
logger = logging.getLogger(__name__)


class _PodInstance(Protocol):
    container_id: str
    url: str
    management_url: str
    ok: bool
    error_msg: str

    def terminate(self) -> bool: ...


def _validate_string_sequence(name: str, values: object) -> None:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} entries must be non-empty strings")


def _normalize_url(url: str) -> str:
    value = url.strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")
    return value


@dataclass(eq=False)
class BeamComputeAdapter:
    """Provide a normal Tinker client whose model compute runs on Beam.

    The context manager returns :class:`tinker.ServiceClient`. It also routes
    any ``ServiceClient()`` created inside the context to the same backend, so
    existing Tinker Cookbook entrypoints can run without recipe changes.

    With no hardware configuration, serverless mode defaults to A10G.
    On-demand mode opens Beam's hardware picker, while an existing ``pool``
    discovers its GPU type automatically. A concrete ``gpu`` remains useful
    for filtering offers or validating a heterogeneous environment.

    The backend is single-node. ``gpu_count`` GPUs train one PEFT LoRA adapter
    with PyTorch DDP and NCCL. Sampling requests with multiple completions are
    spread across the same GPUs.
    """

    base_model: str
    provider: ProviderName = "beam"
    profile: str | None = None

    gpu: str | Sequence[str] | None = None
    gpu_count: int | None = None
    interconnect: InterconnectPolicy = "auto"
    sampling_gpu: bool = False
    cpu: int | float | str = 4
    memory: int | str = "16Gi"
    max_length: int = 8192
    trust_remote_code: bool = False

    app: str = "tinker-training"
    name: str | None = None
    pool: str | None = None
    pool_fallback: PoolFallback = "wait"
    allow_marketplace: bool = False
    keep_warm_seconds: int = -1

    on_demand: bool = False
    machine_ttl: str = "1h"
    machine_name: str | None = None
    release_machine: bool = True
    show_dashboard_link: bool = True

    volume_name: str = "tinker-checkpoints"
    volume_mount_path: str = _DEFAULT_VOLUME_MOUNT
    volume_verify_timeout: float = 1800
    volume: Any | None = field(default=None, repr=False)
    volumes: Sequence[Any] = ()
    secrets: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict)

    port: int = _DEFAULT_PORT
    authorized: bool = True
    access_token: str | None = field(default=None, repr=False)
    wait_timeout: float = 1800
    poll_interval: float = 5

    image: Any | None = field(default=None, repr=False)
    base_image: str = _DEFAULT_BASE_IMAGE
    tinker_requirement: str | None = None
    python_packages: Sequence[str] = ()
    commands: Sequence[str] = ()

    _instance: _PodInstance | None = field(default=None, init=False, repr=False)
    _client: ServiceClient | None = field(default=None, init=False, repr=False)
    _provider_context: Any | None = field(default=None, init=False, repr=False)
    _resolved_token: str | None = field(default=None, init=False, repr=False)
    _hardware: HardwareManager | None = field(default=None, init=False, repr=False)
    _reserved_pool: str | None = field(default=None, init=False, repr=False)
    _last_container_id: str | None = field(default=None, init=False, repr=False)
    _last_dashboard_url: str | None = field(default=None, init=False, repr=False)
    _runtime_info: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _checkpoint_verification: list[dict[str, Any]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _original_service_client: Any | None = field(default=None, init=False, repr=False)
    _service_client_proxy: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.provider not in ("beam", "beta9"):
            raise ValueError("provider must be either 'beam' or 'beta9'")
        if not self.base_model.strip():
            raise ValueError("base_model must be a non-empty string")
        if self.profile is not None and not self.profile.strip():
            raise ValueError("profile must be non-empty or None")
        if self.gpu is None:
            if not self.on_demand and self.pool is None:
                self.gpu = "A10G"
        elif isinstance(self.gpu, str):
            if not self.gpu.strip():
                raise ValueError("gpu must be a non-empty resource value")
        else:
            _validate_string_sequence("gpu", self.gpu)
        if self.gpu_count is None:
            # Compatibility with the original two-device training/sampling switch.
            self.gpu_count = 2 if self.sampling_gpu else 1
        if isinstance(self.gpu_count, bool) or self.gpu_count <= 0:
            raise ValueError("gpu_count must be greater than zero")
        if self.sampling_gpu and self.gpu_count < 2:
            raise ValueError("sampling_gpu=True requires gpu_count of at least 2")
        if self.interconnect not in ("auto", "nvlink"):
            raise ValueError("interconnect must be either 'auto' or 'nvlink'")
        if isinstance(self.cpu, (int, float)) and (isinstance(self.cpu, bool) or self.cpu <= 0):
            raise ValueError("cpu must be greater than zero")
        if isinstance(self.cpu, str) and not self.cpu.strip():
            raise ValueError("cpu must be a non-empty resource value")
        if isinstance(self.memory, int) and (isinstance(self.memory, bool) or self.memory <= 0):
            raise ValueError("memory must be greater than zero")
        if isinstance(self.memory, str) and not self.memory.strip():
            raise ValueError("memory must be a non-empty resource value")
        if self.max_length <= 0:
            raise ValueError("max_length must be greater than zero")
        if not self.app.strip():
            raise ValueError("app must be a non-empty string")
        if self.name is not None and not self.name.strip():
            raise ValueError("name must be non-empty or None")
        if self.pool is not None and not self.pool.strip():
            raise ValueError("pool must be non-empty or None")
        if self.pool_fallback not in ("wait", "fail"):
            raise ValueError("pool_fallback must be either 'wait' or 'fail'")
        if self.on_demand and self.pool is not None:
            raise ValueError("on_demand and pool are mutually exclusive hardware modes")
        if not self.machine_ttl.strip():
            raise ValueError("machine_ttl must be non-empty")
        if self.machine_name is not None and not self.machine_name.strip():
            raise ValueError("machine_name must be non-empty or None")
        if self.on_demand and self.gpu is not None and not isinstance(self.gpu, str):
            raise ValueError("on_demand accepts one GPU type or None for Beam's hardware picker")
        if not self.volume_name.strip():
            raise ValueError("volume_name must be a non-empty string")
        if not self.volume_mount_path.startswith("/"):
            raise ValueError("volume_mount_path must be an absolute path")
        if not self.volume_mount_path.strip("/"):
            raise ValueError("volume_mount_path must name a directory below root")
        if not 1 <= self.volume_verify_timeout <= 7200:
            raise ValueError("volume_verify_timeout must be between 1 and 7200")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.keep_warm_seconds != -1 and self.keep_warm_seconds <= 0:
            raise ValueError("keep_warm_seconds must be -1 or greater than zero")
        if self.wait_timeout <= 0 or self.poll_interval <= 0:
            raise ValueError("wait_timeout and poll_interval must be greater than zero")
        if not self.base_image.strip():
            raise ValueError("base_image must be a non-empty string")
        if self.tinker_requirement is not None and not self.tinker_requirement.strip():
            raise ValueError("tinker_requirement must be non-empty or None")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.env.items()
        ):
            raise TypeError("env keys and values must be strings")
        _validate_string_sequence("secrets", self.secrets)
        _validate_string_sequence("python_packages", self.python_packages)
        _validate_string_sequence("commands", self.commands)

    @property
    def endpoint_url(self) -> str | None:
        """The active backend URL, if the adapter has been started."""

        return _normalize_url(self._instance.url) if self._instance is not None else None

    @property
    def container_id(self) -> str | None:
        """The current or most recently created Beam/Beta9 container ID."""

        if self._instance is not None and self._instance.container_id:
            return self._instance.container_id
        return self._last_container_id

    @property
    def dashboard_url(self) -> str | None:
        """The current or most recent Pod's provider dashboard URL."""

        if self._instance is not None:
            value = getattr(self._instance, "management_url", "").strip()
            if value:
                return value
        if self._last_dashboard_url is not None:
            return self._last_dashboard_url
        if self.provider == "beam" and self.container_id is not None:
            return _BEAM_CONTAINERS_DASHBOARD
        return None

    @property
    def runtime_info(self) -> Mapping[str, Any] | None:
        """Validated GPU and interconnect details reported by the active backend."""

        return dict(self._runtime_info) if self._runtime_info is not None else None

    @property
    def checkpoint_verification(self) -> Sequence[Mapping[str, Any]]:
        """Checkpoints durably verified inside the most recent Pod."""

        return tuple(dict(item) for item in self._checkpoint_verification)

    def refresh_runtime_info(self) -> Mapping[str, Any]:
        """Fetch per-rank CUDA work counters from the running backend."""

        if self._instance is None:
            raise RuntimeError("the Beam compute backend is not running")
        headers = (
            {"Authorization": f"Bearer {self._resolved_token}"} if self._resolved_token else {}
        )
        url = f"{_normalize_url(self._instance.url)}/opentinker/runtime"
        response = httpx.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Beam compute backend returned invalid runtime information")
        self._runtime_info = payload
        return dict(payload)

    def start(self, *, wait: bool = True) -> ServiceClient:
        """Start the backend and return an ordinary Tinker ``ServiceClient``."""

        if self._instance is not None:
            raise RuntimeError("this adapter already has a running backend")
        self._checkpoint_verification.clear()
        provider = self._load_provider()
        self._hardware = self._new_hardware_manager()
        try:
            image = self.image if self.image is not None else self._build_image(provider)
            if self.on_demand:
                # Build before reserving so paid hardware is held only for the workload.
                if not getattr(image, "override_python_version", False):
                    image.ignore_python = True
                build_result = image.build()
                if not build_result.success:
                    raise RuntimeError("failed to build Beam/Beta9 image")
            selection = self._hardware.select()
            self.gpu = selection.gpu
            self.pool = selection.pool
            self._reserved_pool = self._hardware.owned_pool
            primary_volume = self.volume
            if primary_volume is None:
                primary_volume = provider.Volume(
                    name=self.volume_name,
                    mount_path=self.volume_mount_path,
                )
            pod_options = self._pod_options(image, primary_volume)
            pod = provider.Pod(**pod_options)
            self._configure_pool_fallback(pod)
            instance = cast(_PodInstance, pod.create())
            self._instance = instance
            if not instance.ok:
                raise RuntimeError(f"failed to create Beam/Beta9 Pod: {instance.error_msg}")
            self._remember_instance(instance)
            self._announce_pod()
            if not instance.url:
                raise RuntimeError("Beam/Beta9 created the Pod but returned no endpoint URL")
            token = self._resolve_access_token(pod)
            if self.authorized and not token:
                raise RuntimeError(
                    "authorized=True requires a Beam/Beta9 token; configure the selected profile "
                    "or pass access_token"
                )
            self._resolved_token = token
            if wait:
                self._wait_until_ready(token)
            self._client = ServiceClient(**self._client_options(token))
            return self._client
        except BaseException as exc:
            if self._instance is not None:
                self._announce_exception(exc, persist_checkpoints=False)
            clean = self._cleanup(persist_checkpoints=False)
            if not clean:
                self._announce_cleanup_failure()
            raise

    def _remember_instance(self, instance: _PodInstance) -> None:
        """Retain monitoring details after cleanup for error reports and callers."""

        self._last_container_id = instance.container_id or None
        management_url = getattr(instance, "management_url", "").strip()
        self._last_dashboard_url = management_url or None

    def _operator_command(self, *arguments: str) -> str:
        manager = self._hardware or self._new_hardware_manager()
        return manager.operator_command(*arguments)

    def _new_hardware_manager(self) -> HardwareManager:
        return HardwareManager(
            provider=self.provider,
            profile=self.profile,
            gpu=self.gpu,
            gpu_count=int(self.gpu_count or 1),
            pool=self.pool,
            on_demand=self.on_demand,
            machine_ttl=self.machine_ttl,
            machine_name=self.machine_name,
            release_machine=self.release_machine,
            run=subprocess.run,
            find_executable=shutil.which,
            interactive=sys.stdin.isatty() and sys.stderr.isatty(),
            reservation_timeout=self.wait_timeout,
            poll_interval=self.poll_interval,
        )

    def _announce_pod(self) -> None:
        """Print the live monitoring handoff before waiting for model readiness."""

        if not self.show_dashboard_link:
            return
        container_id = self.container_id
        lines = [f"OpenTinker Pod created: {container_id or 'unknown'}"]
        if self.dashboard_url is not None:
            lines.append(f"  dashboard: {self.dashboard_url}")
        elif container_id is not None:
            lines.append(
                "  dashboard: provider did not return a direct link; find the Pod with "
                f"`{self._operator_command('container', 'list')}`"
            )
        if container_id is not None:
            lines.append(
                f"  attach:    {self._operator_command('container', 'attach', container_id)}"
            )
        lines.extend(
            (
                f"  app:       {self.app}",
                f"  hardware:  {self.gpu_count}x {self.gpu} (interconnect={self.interconnect})",
                "  Ctrl+C stops the training loop, flushes completed checkpoints, and "
                "terminates this Pod.",
            )
        )
        print("\n".join(lines), file=sys.stderr, flush=True)

    def _announce_exception(self, exc: BaseException, *, persist_checkpoints: bool) -> None:
        if not self.show_dashboard_link:
            return
        reason = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        lines = [f"OpenTinker training {reason}; cleaning up Pod {self.container_id or 'unknown'}."]
        if self.dashboard_url is not None:
            lines.append(f"  dashboard: {self.dashboard_url}")
        if persist_checkpoints:
            lines.append("  Completed checkpoints will be flushed before the Pod is terminated.")
        else:
            lines.append("  The partially started Pod will be terminated.")
        print("\n".join(lines), file=sys.stderr, flush=True)

    def _announce_runtime(self) -> None:
        if not self.show_dashboard_link or self._runtime_info is None:
            return
        names = self._runtime_info.get("gpu_names") or []
        gpu_name = names[0] if names else self.gpu
        count = self._runtime_info.get("gpu_count", self.gpu_count)
        strategy = self._runtime_info.get("strategy", "unknown")
        interconnect = self._runtime_info.get("interconnect", "unknown")
        print(
            f"OpenTinker backend ready: {count}x {gpu_name} "
            f"({strategy}, interconnect={interconnect})",
            file=sys.stderr,
            flush=True,
        )

    def _announce_cleanup_failure(self) -> None:
        if not self.show_dashboard_link:
            return
        lines = ["OpenTinker cleanup was incomplete; inspect the remote resources manually."]
        if self.dashboard_url is not None:
            lines.append(f"  dashboard: {self.dashboard_url}")
        if self.container_id is not None:
            lines.append(
                f"  stop Pod:  {self._operator_command('container', 'stop', self.container_id)}"
            )
        if self._reserved_pool is not None:
            lines.append(
                "  release:   "
                f"{self._operator_command('machine', 'release', '--pool', self._reserved_pool, '--yes')}"
            )
        print("\n".join(lines), file=sys.stderr, flush=True)

    def _pod_options(self, image: Any, primary_volume: Any) -> dict[str, Any]:
        options: dict[str, Any] = {
            "app": self.app,
            "name": self.name,
            "entrypoint": self._entrypoint(),
            "ports": [self.port],
            "cpu": self.cpu,
            "memory": self.memory,
            "gpu": self.gpu if isinstance(self.gpu, str) else list(self.gpu or ()),
            "gpu_count": self.gpu_count,
            "image": image,
            "volumes": [primary_volume, *self.volumes],
            "secrets": list(self.secrets),
            "env": self._server_environment(),
            "keep_warm_seconds": self.keep_warm_seconds,
            "authorized": self.authorized,
            "allow_marketplace": self.allow_marketplace,
        }
        if self.pool is not None:
            options["pool"] = self.pool
        return options

    def _server_environment(self) -> dict[str, str]:
        return {
            **dict(self.env),
            "OPENTINKER_BASE_MODEL": self.base_model,
            # Beta9 exposes SDK Volumes under /volumes/<mount-name>.
            "OPENTINKER_CHECKPOINT_ROOT": (
                f"/volumes/{self.volume_mount_path.strip('/')}/checkpoints"
            ),
            "OPENTINKER_VOLUME_NAME": self.volume_name,
            "OPENTINKER_MAX_LENGTH": str(self.max_length),
            "OPENTINKER_PORT": str(self.port),
            "OPENTINKER_GPU_COUNT": str(self.gpu_count),
            "OPENTINKER_INTERCONNECT": self.interconnect,
            "OPENTINKER_SAMPLING_GPU": "1" if self.sampling_gpu else "0",
            "OPENTINKER_TRUST_REMOTE_CODE": "1" if self.trust_remote_code else "0",
        }

    def _configure_pool_fallback(self, pod: Any) -> None:
        """Keep private-pool jobs on their selected hardware."""

        if self.pool is None:
            return
        pool_config = getattr(pod, "pool_config", None)
        if pool_config is None:
            raise RuntimeError("installed Beam/Beta9 SDK cannot configure private-pool fallback")
        pool_config.fallback = self.pool_fallback

    def stop(self) -> bool:
        """Close the Tinker client and terminate the adapter-created Pod."""

        return self._cleanup(persist_checkpoints=True)

    def _cleanup(self, *, persist_checkpoints: bool) -> bool:
        """Release every owned resource, including after a partial startup."""

        clean = True
        try:
            self._restore_tinker_service_client()
        except Exception:
            logger.exception("Could not restore Tinker's ServiceClient")
            clean = False

        client, self._client = self._client, None
        if client is not None:
            try:
                self._close_client(client)
            except Exception:
                logger.exception("Could not close the Tinker client")
                clean = False

        instance, self._instance = self._instance, None
        if instance is not None:
            if persist_checkpoints:
                clean = self._prepare_volume_shutdown(instance) and clean
            try:
                clean = instance.terminate() and clean
            except Exception:
                logger.exception("Could not terminate the Beam/Beta9 Pod")
                clean = False

        self._resolved_token = None
        try:
            clean = self._release_reserved_machine() and clean
        except Exception:
            logger.exception("Could not release the on-demand Beam/Beta9 machine")
            clean = False
        return clean

    def _prepare_volume_shutdown(self, instance: _PodInstance) -> bool:
        """Confirm completed checkpoints were durably published inside the Pod."""

        headers = (
            {"Authorization": f"Bearer {self._resolved_token}"} if self._resolved_token else {}
        )
        url = f"{_normalize_url(instance.url)}/opentinker/prepare-shutdown"
        try:
            response = httpx.post(
                url,
                headers=headers,
                timeout=self.volume_verify_timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            logger.warning("Could not verify Beam Volume checkpoints before shutdown: %s", exc)
            return False
        if not isinstance(payload, dict):
            logger.error("Backend returned invalid checkpoint verification response")
            return False
        checkpoints = payload.get("checkpoints", [])
        if not isinstance(checkpoints, list):
            logger.error("Backend returned invalid checkpoint verification metadata")
            return False
        verified: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict) or checkpoint.get("verified") is not True:
                logger.error("Backend reported an unverified Beam Volume checkpoint")
                return False
            if not all(
                isinstance(checkpoint.get(key), str) and checkpoint[key]
                for key in ("uri", "volume_path", "manifest_sha256")
            ):
                logger.error("Backend returned incomplete checkpoint verification metadata")
                return False
            verified.append(dict(checkpoint))
            print(
                "Verified inside Pod with fsync + SHA-256 metadata: "
                f"beam://{checkpoint['volume_path']} "
                f"({checkpoint.get('file_count', '?')} files, zero downloaded)",
                file=sys.stderr,
                flush=True,
            )
        self._checkpoint_verification = verified
        return True

    @staticmethod
    def _close_client(client: Any) -> None:
        """Close both lazy holders on current Tinker and the legacy holder API."""

        holders = [
            holder
            for holder in (
                getattr(client, "_session_holder", None),
                getattr(client, "_rest_holder", None),
            )
            if holder is not None
        ]
        if not holders and not hasattr(client, "_session_holder"):
            holders.append(client.holder)
        for holder in dict.fromkeys(holders):
            holder.close()

    @classmethod
    def connect(
        cls,
        url: str,
        *,
        access_token: str | None = None,
        api_key: str = _CLIENT_API_KEY,
    ) -> ServiceClient:
        """Return a normal Tinker client for an already-running adapter endpoint."""

        headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
        return ServiceClient(api_key=api_key, base_url=_normalize_url(url), default_headers=headers)

    def _wait_until_ready(self, token: str | None) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        deadline = time.monotonic() + self.wait_timeout
        url = f"{_normalize_url(cast(_PodInstance, self._instance).url)}/api/v1/healthz"
        last_error = "endpoint did not become reachable"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(url, headers=headers, timeout=10)
                payload = response.json() if response.status_code == 200 else {}
                if payload.get("status") == "ok":
                    runtime = payload.get("runtime")
                    if isinstance(runtime, dict):
                        self._runtime_info = runtime
                        self._announce_runtime()
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except (httpx.HTTPError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(self.poll_interval, max(deadline - time.monotonic(), 0)))
        raise TimeoutError(
            f"Beam Tinker backend was not ready after {self.wait_timeout:g}s ({last_error})"
        )

    def _client_options(self, token: str | None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return {
            "api_key": _CLIENT_API_KEY,
            "base_url": _normalize_url(cast(_PodInstance, self._instance).url),
            "default_headers": headers,
        }

    def _resolve_access_token(self, pod: Any) -> str | None:
        if self.access_token is not None:
            return self.access_token
        context = self._provider_context or getattr(pod, "config_context", None)
        token = getattr(context, "token", None)
        if token:
            return cast(str, token)
        environment_name = "BEAM_TOKEN" if self.provider == "beam" else "BETA9_TOKEN"
        return os.environ.get(environment_name)

    def _entrypoint(self) -> list[str]:
        script = (
            'export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"; '
            'if [ "${OPENTINKER_GPU_COUNT:-1}" -gt 1 ]; then '
            "exec torchrun --nnodes=1 --node-rank=0 "
            "--master-addr=127.0.0.1 --master-port=29500 "
            '--nproc-per-node="${OPENTINKER_GPU_COUNT}" '
            "-m opentinker._server; "
            "else exec python -m opentinker._server; fi"
        )
        return ["/bin/bash", "-lc", script]

    def _build_image(self, provider: ModuleType) -> Any:
        return build_backend_image(
            provider,
            BackendImageSpec(
                base_image=self.base_image,
                tinker_requirement=self.tinker_requirement,
                python_packages=self.python_packages,
                commands=self.commands,
            ),
        )

    def _load_provider(self) -> ModuleType:
        runtime = load_provider(self.provider, profile=self.profile)
        self._provider_context = runtime.context
        return runtime.module

    def _release_reserved_machine(self) -> bool:
        if self._hardware is None:
            return True
        clean = self._hardware.release()
        self.gpu = self._hardware.gpu
        self.pool = self._hardware.pool
        self._reserved_pool = self._hardware.owned_pool
        return clean

    def _patch_tinker_service_client(self) -> None:
        if self._original_service_client is not None:
            raise RuntimeError("Tinker ServiceClient is already scoped by this adapter")
        original = tinker_module.ServiceClient
        defaults = self._client_options(self._resolved_token)

        def service_client(*args: Any, **kwargs: Any) -> ServiceClient:
            if kwargs.get("base_url") in (None, ""):
                kwargs["base_url"] = defaults["base_url"]
            if kwargs.get("api_key") in (None, ""):
                kwargs["api_key"] = defaults["api_key"]
            caller_headers = kwargs.pop("default_headers", {}) or {}
            kwargs["default_headers"] = {
                **defaults["default_headers"],
                **caller_headers,
            }
            return original(*args, **kwargs)

        self._original_service_client = original
        self._service_client_proxy = service_client
        tinker_module.ServiceClient = service_client  # type: ignore[misc]

    def _restore_tinker_service_client(self) -> None:
        if self._original_service_client is None:
            return
        if tinker_module.ServiceClient is self._service_client_proxy:
            tinker_module.ServiceClient = self._original_service_client
        self._original_service_client = None
        self._service_client_proxy = None

    def __enter__(self) -> ServiceClient:
        client = self.start()
        self._patch_tinker_service_client()
        return client

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_value is not None:
            self._announce_exception(exc_value, persist_checkpoints=True)
        clean = self.stop()
        if not clean:
            self._announce_cleanup_failure()
        if not clean and exc_type is None:
            raise RuntimeError(
                "Beam adapter cleanup did not complete; inspect the logs for checkpoint, "
                "Pod, or machine-release errors"
            )


__all__ = ["BeamComputeAdapter", "ProviderName"]
