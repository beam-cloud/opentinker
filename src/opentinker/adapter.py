# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnnecessaryIsInstance=false

"""Run ordinary Tinker workflows with model compute on Beam/Beta9 GPUs."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
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
_BEAM_CONTAINERS_DASHBOARD = "https://platform.beam.cloud/containers"
_REPRODUCIBLE_MTIME = 315619200  # 1980-01-02 UTC; ZIP-safe in western timezones.

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

    The backend is single-node. One GPU trains a PEFT LoRA adapter; sampling
    can share it or use a second GPU.
    """

    base_model: str
    provider: ProviderName = "beam"
    profile: str | None = None

    gpu: str | Sequence[str] | None = None
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
    show_dashboard_link: bool = True

    volume_name: str = "tinker-checkpoints"
    volume_mount_path: str = _DEFAULT_VOLUME_MOUNT
    volume_sync_seconds: float = 60
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
    _last_container_id: str | None = field(default=None, init=False, repr=False)
    _last_dashboard_url: str | None = field(default=None, init=False, repr=False)
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
        if self.on_demand and self.pool is not None:
            raise ValueError("on_demand and pool are mutually exclusive hardware modes")
        if not self.machine_ttl.strip():
            raise ValueError("machine_ttl must be non-empty")
        if self.machine_name is not None and not self.machine_name.strip():
            raise ValueError("machine_name must be non-empty or None")
        if self.on_demand and self.gpu is not None and not isinstance(self.gpu, str):
            raise ValueError("on_demand accepts one GPU type or None for Beam's hardware picker")
        if self.on_demand and self.sampling_gpu:
            raise ValueError("on_demand currently requires sampling_gpu=False")
        if not self.volume_name.strip():
            raise ValueError("volume_name must be a non-empty string")
        if not self.volume_mount_path.startswith("/"):
            raise ValueError("volume_mount_path must be an absolute path")
        if not self.volume_mount_path.strip("/"):
            raise ValueError("volume_mount_path must name a directory below root")
        if not 0 <= self.volume_sync_seconds <= 600:
            raise ValueError("volume_sync_seconds must be between zero and 600")
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

    def start(self, *, wait: bool = True) -> ServiceClient:
        """Start the backend and return an ordinary Tinker ``ServiceClient``."""

        if self._instance is not None:
            raise RuntimeError("this adapter already has a running backend")
        provider = self._load_provider()
        try:
            image = self.image if self.image is not None else self._build_image(provider)
            if self.on_demand:
                # Build before reserving so paid hardware is held only for the workload.
                if not getattr(image, "override_python_version", False):
                    image.ignore_python = True
                build_result = image.build()
                if not build_result.success:
                    raise RuntimeError("failed to build Beam/Beta9 image")
                self._reserve_machine()
            elif self.pool is not None and self.gpu is None:
                self.gpu = self._pool_gpu(self.pool)
                if self.gpu is None:
                    raise RuntimeError(
                        f"pool {self.pool!r} has no connected machines; attach hardware with "
                        f"`{self.provider} pool join {self.pool}` or choose another pool"
                    )
            if self.gpu is None:
                raise RuntimeError("no GPU was selected for the Beam/Beta9 Pod")
            primary_volume = self.volume
            if primary_volume is None:
                primary_volume = provider.Volume(
                    name=self.volume_name,
                    mount_path=self.volume_mount_path,
                )
            pod_options = self._pod_options(image, primary_volume)
            pod = provider.Pod(**pod_options)
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
        command = [self.provider]
        if self.profile is not None:
            command.extend(["--context", self.profile])
        command.extend(arguments)
        return shlex.join(command)

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
            "OPENTINKER_BASE_MODEL": self.base_model,
            # Beta9 exposes SDK Volumes under /volumes/<mount-name>.
            "OPENTINKER_CHECKPOINT_ROOT": (
                f"/volumes/{self.volume_mount_path.strip('/')}/checkpoints"
            ),
            "OPENTINKER_VOLUME_NAME": self.volume_name,
            "OPENTINKER_VOLUME_SYNC_SECONDS": str(self.volume_sync_seconds),
            "OPENTINKER_MAX_LENGTH": str(self.max_length),
            "OPENTINKER_PORT": str(self.port),
            "OPENTINKER_SAMPLING_GPU": "1" if self.sampling_gpu else "0",
            "OPENTINKER_TRUST_REMOTE_CODE": "1" if self.trust_remote_code else "0",
            **dict(self.env),
        }

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
                checkpoints = self._get_checkpoint_export_plan(instance)
                if checkpoints:
                    clean = self._persist_volume_checkpoints(instance, checkpoints) and clean
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

    def _get_checkpoint_export_plan(self, instance: _PodInstance) -> list[dict[str, str]]:
        """Get the live service's checkpoint export plan before shutdown."""

        if self.volume_sync_seconds == 0:
            return []
        headers = (
            {"Authorization": f"Bearer {self._resolved_token}"} if self._resolved_token else {}
        )
        url = f"{_normalize_url(instance.url)}/opentinker/prepare-shutdown"
        try:
            response = httpx.post(url, headers=headers, timeout=10)
            response.raise_for_status()
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            logger.warning("Could not prepare Beam Volume for shutdown: %s", exc)
            return []
        payload = response.json()
        checkpoints = payload.get("checkpoints", [])
        if not isinstance(checkpoints, list):
            return []
        return [
            {"uri": str(item["uri"]), "volume_path": str(item["volume_path"])}
            for item in checkpoints
            if isinstance(item, dict)
            and isinstance(item.get("uri"), str)
            and item.get("uri")
            and isinstance(item.get("volume_path"), str)
            and item.get("volume_path")
        ]

    def _persist_volume_checkpoints(
        self,
        instance: _PodInstance,
        checkpoints: Sequence[Mapping[str, str]],
    ) -> bool:
        """Upload live checkpoints through Beam when a remote worker mount is local-only."""

        headers = (
            {"Authorization": f"Bearer {self._resolved_token}"} if self._resolved_token else {}
        )
        export_url = f"{_normalize_url(instance.url)}/opentinker/export-checkpoint"
        volume_paths: list[str] = []
        try:
            with tempfile.TemporaryDirectory(prefix="opentinker-export-") as temporary:
                staging_root = Path(temporary)
                for index, checkpoint in enumerate(checkpoints):
                    uri = checkpoint["uri"]
                    volume_path = checkpoint["volume_path"]
                    volume_paths.append(volume_path)
                    archive_path = staging_root / f"checkpoint-{index}.tar.gz"
                    with httpx.stream(
                        "POST",
                        export_url,
                        headers=headers,
                        json={"path": uri},
                        timeout=httpx.Timeout(600, connect=10),
                    ) as response:
                        response.raise_for_status()
                        with archive_path.open("wb") as archive_file:
                            for chunk in response.iter_bytes():
                                archive_file.write(chunk)
                    extracted = staging_root / f"extracted-{index}"
                    with tarfile.open(archive_path, "r:gz") as archive:
                        archive.extractall(extracted, filter="data")
                    source = extracted / "checkpoint"
                    if not source.is_dir():
                        raise RuntimeError(
                            f"checkpoint export for {uri} had no checkpoint directory"
                        )
                    command = self._machine_command(
                        "cp",
                        str(source),
                        f"beam://{volume_path}",
                    )
                    interactive = sys.stderr.isatty()
                    if interactive:
                        print(
                            f"Persisting {uri} to beam://{volume_path}",
                            file=sys.stderr,
                            flush=True,
                        )
                    result = subprocess.run(
                        command,
                        check=False,
                        capture_output=not interactive,
                        text=True,
                        cwd=temporary,
                    )
                    if result.returncode != 0:
                        detail = (result.stderr or result.stdout).strip()
                        raise RuntimeError(f"could not upload {uri} to Beam Volume: {detail}")
            return self._wait_for_published_volume_paths(volume_paths)
        except Exception as exc:
            logger.error("Could not persist Beam Volume checkpoints: %s", exc)
            return False

    def _wait_for_published_volume_paths(self, volume_paths: Sequence[str]) -> bool:
        """Wait until checkpoint files are visible through Beam's Volume API."""

        try:
            from beta9.channel import ServiceClient as ProviderServiceClient
            from beta9.clients.volume import ListPathRequest
            from beta9.config import get_config_context

            configured_context = (
                get_config_context(self.profile)
                if self.profile is not None
                else get_config_context()
            )
            context = self._provider_context or configured_context
            deadline = time.monotonic() + self.volume_sync_seconds
            with ProviderServiceClient(config=context) as provider_client:
                while True:
                    pending = []
                    for path in volume_paths:
                        result = provider_client.volume.list_path(ListPathRequest(path=path))
                        if not result.ok or not result.path_infos:
                            pending.append(path)
                    if not pending:
                        return True
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.error(
                            "Beam Volume paths were not published before machine release: %s",
                            ", ".join(pending),
                        )
                        return False
                    time.sleep(min(self.poll_interval, remaining))
        except Exception as exc:
            logger.warning("Could not verify Beam Volume publication: %s", exc)
            return False

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

        # Custom Beam/Beta9 images skip source sync. Embed the server package so
        # an installed OpenTinker wheel can launch a Pod without a remote checkout.
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
        assert self.gpu is None or isinstance(self.gpu, str)
        if self._reserved_pool is not None:
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
        command = self._machine_command(*arguments)
        self.pool = pool
        self._reserved_pool = pool
        try:
            # Inherit the terminal streams. Beam opens its native offer picker
            # when interactive and selects the cheapest offer when headless.
            result = subprocess.run(command, check=False)
        except BaseException:
            # The reservation RPC may have succeeded before the local wait was
            # interrupted. Record ownership first so cleanup always attempts a release.
            self._release_reserved_machine()
            raise
        if result.returncode != 0:
            self._release_reserved_machine()
            raise RuntimeError("failed to reserve an on-demand machine; see Beam's output above")

        try:
            selected_gpu = self._pool_gpu(pool)
        except BaseException:
            self._release_reserved_machine()
            raise
        if selected_gpu is None:
            # A declined interactive confirmation exits successfully without
            # creating a pool. There is nothing to release in that case.
            self._reserved_pool = None
            self.pool = original_pool
            raise RuntimeError("on-demand reservation was cancelled; no machine was created")
        if requested_gpu is not None and selected_gpu != requested_gpu:
            self._release_reserved_machine()
            raise RuntimeError(
                f"Beam reserved {selected_gpu}, but OpenTinker requested {requested_gpu}"
            )
        self.gpu = selected_gpu
        logger.info("Reserved on-demand Beam pool %s with %s", pool, selected_gpu)

    def _pool_gpu(self, pool: str) -> str | None:
        """Return the single GPU type advertised by a connected machine pool."""

        command = self._machine_command(
            "machine",
            "list",
            "--pool",
            pool,
            "--no-offers",
            "--format",
            "json",
        )
        result = subprocess.run(command, check=False, capture_output=True, text=True)
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
        matched_machines = [
            machine
            for machine in machines
            if isinstance(machine, dict) and machine.get("pool_name") == pool
        ]
        if not matched_machines:
            return None
        gpus = {str(machine.get("gpu")) for machine in matched_machines if machine.get("gpu")}
        if not gpus:
            raise RuntimeError("OpenTinker requires a GPU, but the selected machine is CPU-only")
        if len(gpus) != 1:
            raise RuntimeError(
                f"Beam/Beta9 pool {pool!r} contains multiple GPU types; pass gpu= explicitly"
            )
        return gpus.pop()

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
