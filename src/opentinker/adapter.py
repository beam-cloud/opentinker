# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnnecessaryIsInstance=false

"""Run ordinary Tinker workflows with model compute on Beam/Beta9 GPUs."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Literal, Protocol, cast

import httpx
import tinker as tinker_module
from tinker.lib.public_interfaces.service_client import ServiceClient

ProviderName = Literal["beam", "beta9"]

_DEFAULT_PORT = 8000
_DEFAULT_VOLUME_MOUNT = "/tinker-data"
_DEFAULT_BASE_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
_CLIENT_API_KEY = "tml-beam-compute"
_REPRODUCIBLE_MTIME = 315532800  # 1980-01-01; valid for ZIP-based build contexts.

logger = logging.getLogger(__name__)


class _PodInstance(Protocol):
    container_id: str
    url: str
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

    The first implementation is intentionally single-node. One GPU trains a
    PEFT LoRA adapter; sampling can share it or use a second GPU.
    """

    base_model: str
    provider: ProviderName = "beam"
    profile: str | None = None

    gpu: str | Sequence[str] = "A10G"
    sampling_gpu: bool = False
    cpu: int | float | str = 4
    memory: int | str = "16Gi"
    max_length: int = 8192
    trust_remote_code: bool = False

    app: str = "tinker-training"
    name: str | None = None
    pool: str | None = None
    allow_marketplace: bool = False
    keep_warm_seconds: int = -1

    on_demand: bool = False
    machine_ttl: str = "1h"
    machine_name: str | None = None
    release_machine: bool = True

    volume_name: str = "tinker-checkpoints"
    volume_mount_path: str = _DEFAULT_VOLUME_MOUNT
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
    _reserved_pool: str | None = field(default=None, init=False, repr=False)
    _original_service_client: Any | None = field(default=None, init=False, repr=False)
    _service_client_proxy: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.provider not in ("beam", "beta9"):
            raise ValueError("provider must be either 'beam' or 'beta9'")
        if not self.base_model.strip():
            raise ValueError("base_model must be a non-empty string")
        if self.profile is not None and not self.profile.strip():
            raise ValueError("profile must be non-empty or None")
        if isinstance(self.gpu, str):
            if not self.gpu.strip():
                raise ValueError("gpu must be a non-empty resource value")
        else:
            _validate_string_sequence("gpu", self.gpu)
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
        if not self.machine_ttl.strip():
            raise ValueError("machine_ttl must be non-empty")
        if self.machine_name is not None and not self.machine_name.strip():
            raise ValueError("machine_name must be non-empty or None")
        if self.on_demand and not isinstance(self.gpu, str):
            raise ValueError("on_demand requires one concrete GPU type")
        if self.on_demand and self.sampling_gpu:
            raise ValueError("on_demand currently requires sampling_gpu=False")
        if not self.volume_name.strip():
            raise ValueError("volume_name must be a non-empty string")
        if not self.volume_mount_path.startswith("/"):
            raise ValueError("volume_mount_path must be an absolute path")
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
    def gpu_count(self) -> int:
        """Number of GPUs allocated to the Pod."""

        return 2 if self.sampling_gpu else 1

    @property
    def endpoint_url(self) -> str | None:
        """The active backend URL, if the adapter has been started."""

        return _normalize_url(self._instance.url) if self._instance is not None else None

    def start(self, *, wait: bool = True) -> ServiceClient:
        """Start the backend and return an ordinary Tinker ``ServiceClient``."""

        if self._instance is not None:
            raise RuntimeError("this adapter already has a running backend")
        provider = self._load_provider()
        image = self.image if self.image is not None else self._build_image(provider)
        if self.on_demand:
            # Image builds run on Beam's builder, not the reserved training node.
            # Build first so the paid machine is only held for Pod startup/training.
            if not getattr(image, "override_python_version", False):
                image.ignore_python = True
            build_result = image.build()
            if not build_result.success:
                raise RuntimeError("failed to build Beam/Beta9 image")
            self._reserve_machine()
        primary_volume = self.volume
        if primary_volume is None:
            try:
                primary_volume = provider.Volume(
                    name=self.volume_name,
                    mount_path=self.volume_mount_path,
                )
            except BaseException:
                self._release_reserved_machine()
                raise
        pod_options: dict[str, Any] = {
            "app": self.app,
            "name": self.name,
            "entrypoint": self._entrypoint(),
            "ports": [self.port],
            "cpu": self.cpu,
            "memory": self.memory,
            "gpu": self.gpu if isinstance(self.gpu, str) else list(self.gpu),
            "gpu_count": self.gpu_count,
            "image": image,
            "volumes": [primary_volume, *self.volumes],
            "secrets": list(self.secrets),
            "env": {
                "OPENTINKER_BASE_MODEL": self.base_model,
                "OPENTINKER_CHECKPOINT_ROOT": f"{self.volume_mount_path}/checkpoints",
                "OPENTINKER_VOLUME_NAME": self.volume_name,
                "OPENTINKER_MAX_LENGTH": str(self.max_length),
                "OPENTINKER_PORT": str(self.port),
                "OPENTINKER_SAMPLING_GPU": "1" if self.sampling_gpu else "0",
                "OPENTINKER_TRUST_REMOTE_CODE": "1" if self.trust_remote_code else "0",
                **dict(self.env),
            },
            "keep_warm_seconds": self.keep_warm_seconds,
            "authorized": self.authorized,
            "allow_marketplace": self.allow_marketplace,
        }
        if self.pool is not None:
            pod_options["pool"] = self.pool
        try:
            pod = provider.Pod(**pod_options)
        except BaseException:
            self._release_reserved_machine()
            raise
        try:
            instance = cast(_PodInstance, pod.create())
        except BaseException:
            self._release_reserved_machine()
            raise
        if not instance.ok:
            self._release_reserved_machine()
            raise RuntimeError(f"failed to create Beam/Beta9 Pod: {instance.error_msg}")
        if not instance.url:
            instance.terminate()
            self._release_reserved_machine()
            raise RuntimeError("Beam/Beta9 created the Pod but returned no endpoint URL")
        self._instance = instance
        token = self._resolve_access_token(pod)
        if self.authorized and not token:
            self.stop()
            raise RuntimeError(
                "authorized=True requires a Beam/Beta9 token; configure the selected profile "
                "or pass access_token"
            )
        self._resolved_token = token
        if wait:
            try:
                self._wait_until_ready(token)
            except BaseException:
                self.stop()
                raise
        options = self._client_options(token)
        try:
            self._client = ServiceClient(**options)
        except BaseException:
            self.stop()
            raise
        return self._client

    def stop(self) -> bool:
        """Close the Tinker client and terminate the adapter-created Pod."""

        self._restore_tinker_service_client()
        if self._client is not None:
            self._close_client(self._client)
            self._client = None
        terminated = True
        try:
            if self._instance is not None:
                instance = self._instance
                self._instance = None
                terminated = instance.terminate()
        finally:
            self._resolved_token = None
            released = self._release_reserved_machine()
        return terminated and released

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
                if response.status_code == 200 and response.json().get("status") == "ok":
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
            "exec python -m opentinker._server"
        )
        return ["/bin/bash", "-lc", script]

    def _build_image(self, provider: ModuleType) -> Any:
        tinker_version = importlib.metadata.version("tinker")
        packages = [
            self.tinker_requirement or f"tinker=={tinker_version}",
            "peft>=0.14,<1",
            "transformers>=4.57.6,<5",
            "fastapi>=0.115,<1",
            "uvicorn[standard]>=0.34,<1",
            *self.python_packages,
        ]
        if any(character in self.base_image for character in "\r\n"):
            raise ValueError("base_image must not contain newlines")

        # Beam/Beta9 deliberately skips source sync for custom images. Embed the
        # small server package into the image so an installed OpenTinker wheel is
        # sufficient to launch a Pod; no repository checkout is needed remotely.
        package_source = Path(__file__).parent
        with tempfile.TemporaryDirectory(prefix="opentinker-image-") as temp_directory:
            context = Path(temp_directory)
            shutil.copytree(
                package_source,
                context / "opentinker",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            install = " ".join(shlex.quote(package) for package in packages)
            dockerfile_lines = [
                f"FROM {self.base_image}",
                f"RUN python -m pip install --no-cache-dir {install}",
                *(f"RUN /bin/bash -lc {shlex.quote(command)}" for command in self.commands),
                "COPY opentinker /opt/opentinker/opentinker",
                "ENV PYTHONPATH=/opt/opentinker",
            ]
            dockerfile = context / "Dockerfile"
            dockerfile.write_text("\n".join(dockerfile_lines) + "\n")

            # Beam hashes the archived build context. Temporary directories and
            # the generated Dockerfile otherwise get a fresh timestamp on every
            # invocation, defeating image reuse even when the inputs are equal.
            for path in sorted(context.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                os.utime(path, (_REPRODUCIBLE_MTIME, _REPRODUCIBLE_MTIME))
            os.utime(context, (_REPRODUCIBLE_MTIME, _REPRODUCIBLE_MTIME))
            return provider.Image.from_dockerfile(str(dockerfile), context_dir=str(context))

    def _load_provider(self) -> ModuleType:
        try:
            provider = importlib.import_module(self.provider)
        except ModuleNotFoundError as exc:
            if exc.name != self.provider:
                raise
            raise ImportError(
                f"{self.provider} is required; install `opentinker[{self.provider}]`"
            ) from exc
        if self.profile is not None:
            from beta9.abstractions.base import set_channel
            from beta9.config import get_config_context

            context = get_config_context(self.profile)
            if context is None:
                raise ValueError(f"unknown Beta9 profile: {self.profile}")
            set_channel(context=context)
            self._provider_context = context
        missing = [name for name in ("Image", "Pod", "Volume") if not hasattr(provider, name)]
        if missing:
            raise ImportError(
                f"installed {self.provider} package is incompatible (missing: {', '.join(missing)})"
            )
        return provider

    def _machine_command(self, *arguments: str) -> list[str]:
        executable = shutil.which(self.provider)
        if executable is None:
            raise RuntimeError(f"could not find the {self.provider!r} CLI")
        command = [executable]
        if self.profile is not None:
            command.extend(["--context", self.profile])
        command.extend(arguments)
        return command

    def _reserve_machine(self) -> None:
        assert isinstance(self.gpu, str)
        if self._reserved_pool is not None:
            raise RuntimeError("this adapter already owns an on-demand machine")
        pool = self.pool or self.machine_name or f"opentinker-{self.gpu.lower()}"
        command = self._machine_command(
            "machine",
            "reserve",
            "--gpu",
            self.gpu,
            "--nodes",
            "1",
            "--ttl",
            self.machine_ttl,
            "--name",
            pool,
            "--yes",
        )
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        except BaseException:
            # The reservation RPC may have succeeded before the local wait was
            # interrupted. Record ownership first so cleanup always attempts a release.
            self.pool = pool
            self._reserved_pool = pool
            self._release_reserved_machine()
            raise
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"failed to reserve on-demand machine: {detail}")
        self.pool = pool
        self._reserved_pool = pool
        logger.info("Reserved on-demand Beam pool %s", pool)

    def _release_reserved_machine(self) -> bool:
        pool = self._reserved_pool
        if pool is None or not self.release_machine:
            return True
        command = self._machine_command("machine", "release", "--pool", pool, "--yes")
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            logger.error("Failed to release on-demand Beam pool %s: %s", pool, detail)
            return False
        self._reserved_pool = None
        logger.info("Released on-demand Beam pool %s", pool)
        return True

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
        self.stop()


__all__ = ["BeamComputeAdapter", "ProviderName"]
