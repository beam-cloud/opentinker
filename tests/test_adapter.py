from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import httpx
import pytest
import tinker

from opentinker import _image as image_module
from opentinker import adapter as beam_module
from opentinker._hardware import HardwareManager
from opentinker.adapter import BeamComputeAdapter


class FakeImage:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.override_python_version = False
        self.ignore_python = False
        self.build_calls = 0

    def build(self) -> SimpleNamespace:
        self.build_calls += 1
        return SimpleNamespace(success=True)

    @classmethod
    def from_dockerfile(cls, path: str, context_dir: str | None = None) -> FakeImage:
        context = Path(context_dir or Path(path).parent)
        return cls(
            dockerfile=Path(path).read_text(),
            context_files=sorted(
                item.relative_to(context).as_posix()
                for item in context.rglob("*")
                if item.is_file()
            ),
            context_mtimes={int(item.stat().st_mtime) for item in context.rglob("*")},
        )


class FakeVolume:
    def __init__(self, *, name: str, mount_path: str) -> None:
        self.name = name
        self.mount_path = mount_path


class FakePodInstance:
    def __init__(
        self,
        *,
        url: str = "https://pod.example.test",
        management_url: str = "https://dashboard.example.test/pods/pod-123",
        ok: bool = True,
        error_msg: str = "",
    ) -> None:
        self.container_id = "pod-123"
        self.task_id = "task-123"
        self.url = url
        self.management_url = management_url
        self.ok = ok
        self.error_msg = error_msg
        self.terminate_calls = 0

    def terminate(self) -> bool:
        self.terminate_calls += 1
        return True


class FakeHolder:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class FakeServiceClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.holder = FakeHolder()


def fake_provider(
    instance: FakePodInstance | None = None,
    *,
    context_token: str | None = "provider-token",
) -> tuple[ModuleType, list[dict[str, Any]], FakePodInstance]:
    provider = ModuleType("fake_provider")
    pod_calls: list[dict[str, Any]] = []
    pod_instance = instance or FakePodInstance()

    class FakePod:
        def __init__(self, **kwargs: Any) -> None:
            pod_calls.append(kwargs)
            self.config_context = SimpleNamespace(token=context_token)
            self.pool_config = SimpleNamespace(fallback="")

        def create(self) -> FakePodInstance:
            pod_calls[-1]["configured_pool_fallback"] = self.pool_config.fallback
            return pod_instance

    provider.__dict__["Image"] = FakeImage
    provider.__dict__["Volume"] = FakeVolume
    provider.__dict__["Pod"] = FakePod
    return provider, pod_calls, pod_instance


def patch_adapter_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    provider: ModuleType,
) -> list[FakeServiceClient]:
    clients: list[FakeServiceClient] = []

    def service_client(**kwargs: Any) -> FakeServiceClient:
        client = FakeServiceClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(beam_module, "ServiceClient", service_client)
    monkeypatch.setattr(BeamComputeAdapter, "_load_provider", lambda _self: provider)
    monkeypatch.setattr(BeamComputeAdapter, "_prepare_volume_shutdown", lambda *_args: True)
    monkeypatch.setattr(BeamComputeAdapter, "_finish_remote", lambda *_args: True)
    monkeypatch.setattr(
        BeamComputeAdapter, "_wait_for_task_completion", lambda *_args: "COMPLETE"
    )
    return clients


def test_starts_backend_and_returns_normal_client(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider, pod_calls, instance = fake_provider()
    clients = patch_adapter_dependencies(monkeypatch, provider)

    adapter = BeamComputeAdapter(
        base_model="Qwen/Qwen3-8B",
        gpu=("H100", "A100-80"),
        sampling_gpu=True,
        pool="training",
        allow_marketplace=True,
        secrets=("HF_TOKEN",),
        env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    )
    monkeypatch.setattr(HardwareManager, "inspect_pool", lambda _self, _pool: ("H100", 8))
    client = adapter.start(wait=False)

    monitoring = capsys.readouterr().err
    assert "OpenTinker Pod created: pod-123" in monitoring
    assert "dashboard: https://dashboard.example.test/pods/pod-123" in monitoring
    assert "beam container attach pod-123" in monitoring
    assert "app:       tinker-training" in monitoring
    assert adapter.container_id == "pod-123"
    assert adapter.dashboard_url == "https://dashboard.example.test/pods/pod-123"

    assert client is clients[0]
    assert clients[0].kwargs == {
        "api_key": "tml-beam-compute",
        "base_url": "https://pod.example.test",
        "default_headers": {"Authorization": "Bearer provider-token"},
    }
    options = pod_calls[0]
    assert options["gpu"] == ["H100", "A100-80"]
    assert options["gpu_count"] == 2
    assert options["cpu"] == 4
    assert options["memory"] == "16Gi"
    assert options["pool"] == "training"
    assert options["configured_pool_fallback"] == "wait"
    assert options["allow_marketplace"] is True
    assert options["secrets"] == ["HF_TOKEN"]
    assert options["env"] == {
        "OPENTINKER_BASE_MODEL": "Qwen/Qwen3-8B",
        "OPENTINKER_CHECKPOINT_ROOT": "/volumes/tinker-data/checkpoints",
        "OPENTINKER_VOLUME_NAME": "tinker-checkpoints",
        "OPENTINKER_MAX_LENGTH": "8192",
        "OPENTINKER_PORT": "8000",
        "OPENTINKER_GPU_COUNT": "2",
        "OPENTINKER_INTERCONNECT": "auto",
        "OPENTINKER_SAMPLING_GPU": "1",
        "OPENTINKER_TRUST_REMOTE_CODE": "0",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    assert "torchrun --nnodes=1 --node-rank=0" in options["entrypoint"][-1]
    assert "--master-addr=127.0.0.1 --master-port=29500" in options["entrypoint"][-1]
    assert options["volumes"][0].mount_path == "/tinker-data"

    assert adapter.stop() is True
    assert clients[0].holder.close_calls == 1
    assert instance.terminate_calls == 1
    assert adapter.container_id == "pod-123"
    assert adapter.dashboard_url == "https://dashboard.example.test/pods/pod-123"
    assert adapter.stop() is True
    assert instance.terminate_calls == 1


def test_monitoring_handoff_precedes_readiness_and_interrupt_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider, _, instance = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)
    adapter = BeamComputeAdapter(base_model="Qwen/Qwen3-0.6B", profile="prod3")

    def interrupt(_token: str | None) -> None:
        assert "dashboard: https://dashboard.example.test/pods/pod-123" in capsys.readouterr().err
        raise KeyboardInterrupt

    monkeypatch.setattr(adapter, "_wait_until_ready", interrupt)

    with pytest.raises(KeyboardInterrupt):
        adapter.start()

    output = capsys.readouterr().err
    assert "OpenTinker training interrupted; cleaning up Pod pod-123." in output
    assert "dashboard: https://dashboard.example.test/pods/pod-123" in output
    assert "partially started Pod will be terminated" in output
    assert instance.terminate_calls == 1


def test_beam_dashboard_falls_back_to_live_container_page(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider, _, _ = fake_provider(FakePodInstance(management_url=""))
    patch_adapter_dependencies(monkeypatch, provider)
    adapter = BeamComputeAdapter(base_model="Qwen/Qwen3-0.6B", profile="prod3")

    adapter.start(wait=False)

    output = capsys.readouterr().err
    assert "dashboard: https://platform.beam.cloud/containers" in output
    assert "beam --context prod3 container attach pod-123" in output
    assert adapter.dashboard_url == "https://platform.beam.cloud/containers"
    assert adapter.stop() is True


def test_self_hosted_beta9_without_management_url_prints_cli_fallback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider, _, _ = fake_provider(FakePodInstance(management_url=""))
    patch_adapter_dependencies(monkeypatch, provider)
    adapter = BeamComputeAdapter(
        base_model="Qwen/Qwen3-0.6B",
        provider="beta9",
        profile="private-cluster",
    )

    adapter.start(wait=False)

    output = capsys.readouterr().err
    assert "provider did not return a direct link" in output
    assert "beta9 --context private-cluster container list" in output
    assert adapter.dashboard_url is None
    assert adapter.stop() is True


def test_context_interrupt_prints_dashboard_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider, _, instance = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)
    monkeypatch.setattr(BeamComputeAdapter, "_wait_until_ready", lambda *_args: None)
    adapter = BeamComputeAdapter(base_model="Qwen/Qwen3-0.6B")

    with pytest.raises(KeyboardInterrupt), adapter:
        _ = capsys.readouterr()
        raise KeyboardInterrupt

    output = capsys.readouterr().err
    assert "OpenTinker training interrupted; cleaning up Pod pod-123." in output
    assert "dashboard: https://dashboard.example.test/pods/pod-123" in output
    assert "Completed checkpoints will be flushed" in output
    assert instance.terminate_calls == 1


def test_prepare_shutdown_uses_in_pod_volume_verification(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter = BeamComputeAdapter(base_model="Qwen/Qwen3-0.6B")
    adapter._resolved_token = "provider-token"

    def post(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        assert url == "https://pod.example.test/opentinker/prepare-shutdown"
        assert headers == {"Authorization": "Bearer provider-token"}
        assert timeout == 1800
        return httpx.Response(
            200,
            json={
                "checkpoint_saved": True,
                "volume_paths": ["tinker-checkpoints/checkpoints/model/weights/final"],
                "checkpoints": [
                    {
                        "uri": "tinker://model/weights/final",
                        "volume_path": "tinker-checkpoints/checkpoints/model/weights/final",
                        "manifest_sha256": "abc123",
                        "verified": True,
                        "verification": "geesefs-fsync-sha256",
                        "file_count": 4,
                    }
                ],
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(beam_module.httpx, "post", post)
    monkeypatch.setattr(
        beam_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint verification must not download Volume objects")
        ),
    )

    assert adapter._prepare_volume_shutdown(FakePodInstance()) is True
    assert adapter.checkpoint_verification[0]["verification"] == "geesefs-fsync-sha256"
    assert "zero downloaded" in capsys.readouterr().err


def test_context_scopes_cookbook_created_service_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _, instance = fake_provider()
    clients = patch_adapter_dependencies(monkeypatch, provider)
    monkeypatch.setattr(BeamComputeAdapter, "_wait_until_ready", lambda *_: None)
    holder_calls: list[dict[str, Any]] = []

    class CookbookHolder:
        _session_id = "session"

        def __init__(self, **kwargs: Any) -> None:
            holder_calls.append(kwargs)

    monkeypatch.setattr(
        "tinker.lib.public_interfaces.service_client.InternalClientHolder",
        CookbookHolder,
    )

    with BeamComputeAdapter(base_model="Qwen/Qwen3-8B") as returned:
        _ = tinker.ServiceClient(base_url=None, user_metadata={"recipe": "grpo"}).holder

    assert returned is clients[0]
    assert holder_calls[0]["base_url"] == "https://pod.example.test"
    assert holder_calls[0]["api_key"] == "tml-beam-compute"
    assert holder_calls[0]["default_headers"]["Authorization"] == "Bearer provider-token"
    assert holder_calls[0]["user_metadata"] == {"recipe": "grpo"}
    assert instance.terminate_calls == 0


def test_builds_owned_backend_image(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, pod_calls, _ = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)

    BeamComputeAdapter(
        base_model="Qwen/Qwen3-8B",
        tinker_requirement="tinker==9.9.9",
        python_packages=("flash-attn",),
        commands=("apt-get update",),
    ).start(wait=False)

    image = pod_calls[0]["image"]
    dockerfile = image.kwargs["dockerfile"]
    assert dockerfile.startswith(
        "FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime\n"
        "RUN python -m pip install --no-cache-dir tinker==9.9.9"
    )
    assert "'peft>=0.18,<0.19'" in dockerfile
    assert "'transformers>=4.57.6,<5'" in dockerfile
    assert "'fastapi>=0.115,<1'" in dockerfile
    assert "'uvicorn[standard]>=0.34,<1'" in dockerfile
    assert " flash-attn\n" in dockerfile
    assert "RUN /bin/bash -lc 'apt-get update'\n" in dockerfile
    assert dockerfile.endswith(
        "COPY opentinker /opt/opentinker/opentinker\nENV PYTHONPATH=/opt/opentinker\n"
    )
    assert image.kwargs["context_files"] == [
        "Dockerfile",
        "opentinker/__init__.py",
        "opentinker/_api.py",
        "opentinker/_checkpoint.py",
        "opentinker/_distillation.py",
        "opentinker/_distributed.py",
        "opentinker/_engine.py",
        "opentinker/_examples.py",
        "opentinker/_hardware.py",
        "opentinker/_image.py",
        "opentinker/_provider.py",
        "opentinker/_server.py",
        "opentinker/adapter.py",
        "opentinker/data.py",
        "opentinker/py.typed",
    ]
    assert image.kwargs["context_mtimes"] == {image_module._REPRODUCIBLE_MTIME}


def test_default_image_uses_running_tinker_version(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, pod_calls, _ = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)

    BeamComputeAdapter(base_model="Qwen/Qwen3-8B").start(wait=False)

    dockerfile = pod_calls[0]["image"].kwargs["dockerfile"]
    assert f"tinker=={version('tinker')}" in dockerfile
    assert "twinkle" not in dockerfile


def test_waits_for_health_before_creating_client(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _, _ = fake_provider()
    clients = patch_adapter_dependencies(monkeypatch, provider)
    responses = [
        httpx.Response(503, text="loading"),
        httpx.Response(
            200,
            json={
                "status": "ok",
                "runtime": {
                    "gpu_count": 1,
                    "gpu_names": ["NVIDIA A10G"],
                    "strategy": "single",
                    "interconnect": "single",
                },
            },
        ),
    ]

    def get(url: str, *, headers: dict[str, str], timeout: int) -> httpx.Response:
        assert url == "https://pod.example.test/api/v1/healthz"
        assert headers == {"Authorization": "Bearer provider-token"}
        assert timeout == 10
        assert not clients
        return responses.pop(0)

    monkeypatch.setattr(beam_module.httpx, "get", get)
    monkeypatch.setattr(beam_module.time, "sleep", lambda _: None)

    adapter = BeamComputeAdapter(base_model="Qwen/Qwen3-8B")
    adapter.start()

    assert len(clients) == 1
    assert adapter.runtime_info == {
        "gpu_count": 1,
        "gpu_names": ["NVIDIA A10G"],
        "strategy": "single",
        "interconnect": "single",
    }


def test_wait_failure_terminates_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _, instance = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)
    monkeypatch.setattr(
        BeamComputeAdapter,
        "_wait_until_ready",
        lambda *_: (_ for _ in ()).throw(TimeoutError("still loading")),
    )

    with pytest.raises(TimeoutError, match="still loading"):
        BeamComputeAdapter(base_model="Qwen/Qwen3-8B").start()
    assert instance.terminate_calls == 1


def test_failed_start_skips_checkpoint_export(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _, instance = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)
    export_calls = 0

    def export_plan(*_args: Any) -> bool:
        nonlocal export_calls
        export_calls += 1
        return True

    monkeypatch.setattr(BeamComputeAdapter, "_prepare_volume_shutdown", export_plan)
    monkeypatch.setattr(
        BeamComputeAdapter,
        "_wait_until_ready",
        lambda *_: (_ for _ in ()).throw(TimeoutError("still loading")),
    )

    with pytest.raises(TimeoutError, match="still loading"):
        BeamComputeAdapter(base_model="Qwen/Qwen3-8B").start()

    assert export_calls == 0
    assert instance.terminate_calls == 1


def test_cleanup_attempts_every_resource_after_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = BeamComputeAdapter(base_model="Qwen/Qwen3-8B")
    instance = FakePodInstance()
    adapter._client = cast(Any, FakeServiceClient())
    adapter._instance = instance
    released = False

    def release() -> bool:
        nonlocal released
        released = True
        return True

    monkeypatch.setattr(
        adapter,
        "_close_client",
        lambda _client: (_ for _ in ()).throw(RuntimeError("close failed")),
    )
    monkeypatch.setattr(adapter, "_prepare_volume_shutdown", lambda _instance: True)
    monkeypatch.setattr(adapter, "_release_reserved_machine", release)

    assert adapter.stop() is False
    assert adapter._client is None
    assert adapter._instance is None
    assert instance.terminate_calls == 1
    assert released is True


def test_graceful_finish_does_not_cancel_when_status_observation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = BeamComputeAdapter(base_model="Qwen/Qwen3-8B")
    instance = FakePodInstance()
    adapter._instance = instance
    released = False

    def release() -> bool:
        nonlocal released
        released = True
        return True

    monkeypatch.setattr(adapter, "_finish_remote", lambda _instance: True)
    monkeypatch.setattr(adapter, "_wait_for_task_completion", lambda _instance: None)
    monkeypatch.setattr(adapter, "_release_reserved_machine", release)

    assert adapter.finish() is False
    assert instance.terminate_calls == 0
    assert released is False


def test_wait_for_task_completion_uses_profiled_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = BeamComputeAdapter(base_model="Qwen/Qwen3-8B", poll_interval=0.01)
    instance = FakePodInstance()
    requests: list[Any] = []
    statuses = ["RUNNING", "COMPLETE"]

    class Gateway:
        def list_tasks(self, request: Any) -> SimpleNamespace:
            requests.append(request)
            return SimpleNamespace(
                ok=True,
                err_msg="",
                tasks=[SimpleNamespace(status=statuses.pop(0))],
            )

    instance.gateway_stub = Gateway()
    monkeypatch.setattr(beam_module.time, "sleep", lambda _delay: None)

    assert adapter._wait_for_task_completion(cast(Any, instance)) == "COMPLETE"
    assert requests[0].filters["id"].values == ["task-123"]


def test_graceful_finish_releases_hardware_after_terminal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = BeamComputeAdapter(base_model="Qwen/Qwen3-8B")
    instance = FakePodInstance()
    adapter._instance = instance
    released = False

    def release() -> bool:
        nonlocal released
        released = True
        return True

    monkeypatch.setattr(adapter, "_finish_remote", lambda _instance: True)
    monkeypatch.setattr(adapter, "_wait_for_task_completion", lambda _instance: "ERROR")
    monkeypatch.setattr(adapter, "_release_reserved_machine", release)

    assert adapter.finish() is False
    assert instance.terminate_calls == 0
    assert released is True


def test_authorized_backend_requires_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _, instance = fake_provider(context_token=None)
    patch_adapter_dependencies(monkeypatch, provider)
    monkeypatch.delenv("BEAM_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="requires a Beam/Beta9 token"):
        BeamComputeAdapter(base_model="Qwen/Qwen3-8B").start(wait=False)
    assert instance.terminate_calls == 1


def test_named_profile_token_takes_precedence_over_pod_placeholder() -> None:
    adapter = BeamComputeAdapter(
        base_model="Qwen/Qwen3-0.6B",
        provider="beta9",
        profile="prod3",
    )
    adapter._provider_context = SimpleNamespace(token="profile-token")
    pod = SimpleNamespace(config_context=SimpleNamespace(token="placeholder-token"))

    assert adapter._resolve_access_token(pod) == "profile-token"


def test_on_demand_machine_is_reserved_and_released(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, pod_calls, instance = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        if "list" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "machines": [
                            {
                                "pool_name": "opentinker-test",
                                "gpu": "A6000",
                                "gpu_count": 1,
                            }
                        ]
                    }
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(beam_module.shutil, "which", lambda _name: "/usr/local/bin/beam")
    monkeypatch.setattr(beam_module.subprocess, "run", run)

    adapter = BeamComputeAdapter(
        base_model="Qwen/Qwen3-0.6B",
        profile="prod3",
        gpu="A6000",
        on_demand=True,
        machine_ttl="45m",
        machine_name="opentinker-test",
    )
    adapter.start(wait=False)

    assert commands[0] == [
        "/usr/local/bin/beam",
        "--context",
        "prod3",
        "machine",
        "reserve",
        "--nodes",
        "1",
        "--ttl",
        "45m",
        "--name",
        "opentinker-test",
        "--gpu",
        "A6000",
    ]
    assert commands[1][-7:] == [
        "machine",
        "list",
        "--pool",
        "opentinker-test",
        "--no-offers",
        "--format",
        "json",
    ]
    assert pod_calls[0]["pool"] == "opentinker-test"
    assert pod_calls[0]["image"].build_calls == 1
    assert pod_calls[0]["image"].ignore_python is True

    assert adapter.stop() is True
    assert instance.terminate_calls == 1
    assert commands[2] == [
        "/usr/local/bin/beam",
        "--context",
        "prod3",
        "machine",
        "release",
        "--pool",
        "opentinker-test",
        "--yes",
    ]


def test_existing_pool_discovers_its_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, pod_calls, instance = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"machines": [{"pool_name": "my-hardware", "gpu": "RTX4090"}]}),
            stderr="",
        )

    monkeypatch.setattr(beam_module.shutil, "which", lambda _name: "/usr/local/bin/beam")
    monkeypatch.setattr(beam_module.subprocess, "run", run)

    adapter = BeamComputeAdapter(
        base_model="Qwen/Qwen3-0.6B",
        profile="prod3",
        pool="my-hardware",
    )
    assert adapter.gpu is None

    adapter.start(wait=False)

    assert commands == [
        [
            "/usr/local/bin/beam",
            "--context",
            "prod3",
            "machine",
            "list",
            "--pool",
            "my-hardware",
            "--no-offers",
            "--format",
            "json",
        ]
    ]
    assert adapter.gpu == "RTX4090"
    assert pod_calls[0]["gpu"] == "RTX4090"
    assert pod_calls[0]["pool"] == "my-hardware"
    assert adapter.stop() is True
    assert instance.terminate_calls == 1


def test_multi_gpu_pool_requires_one_machine_with_enough_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, pod_calls, _ = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)
    monkeypatch.setattr(beam_module.shutil, "which", lambda _name: "/usr/local/bin/beam")
    monkeypatch.setattr(
        beam_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "machines": [
                        {
                            "pool_name": "h100s",
                            "gpu": "H100,H100,H100,H100",
                            "gpu_count": 4,
                        },
                        {
                            "pool_name": "h100s",
                            "gpu": "H100,H100,H100,H100",
                            "gpu_count": 4,
                        },
                    ]
                }
            ),
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="at most 4 GPUs on one machine"):
        BeamComputeAdapter(
            base_model="Qwen/Qwen3-0.6B",
            pool="h100s",
            gpu_count=8,
        ).start(wait=False)

    assert pod_calls == []


def test_multi_gpu_options_launch_torchrun_and_require_nvlink() -> None:
    adapter = BeamComputeAdapter(
        base_model="Qwen/Qwen3-0.6B",
        gpu="H100",
        gpu_count=8,
        interconnect="nvlink",
    )

    entrypoint = adapter._entrypoint()[-1]
    assert "torchrun --nnodes=1 --node-rank=0" in entrypoint
    assert "--master-addr=127.0.0.1 --master-port=29500" in entrypoint
    assert '--nproc-per-node="${OPENTINKER_GPU_COUNT}"' in entrypoint
    assert adapter._server_environment()["OPENTINKER_GPU_COUNT"] == "8"
    assert adapter._server_environment()["OPENTINKER_INTERCONNECT"] == "nvlink"


def test_existing_pool_without_connected_hardware_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, pod_calls, _ = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)
    monkeypatch.setattr(beam_module.shutil, "which", lambda _name: "/usr/local/bin/beam")
    monkeypatch.setattr(
        beam_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"machines": []}),
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="has no connected machines"):
        BeamComputeAdapter(
            base_model="Qwen/Qwen3-0.6B",
            pool="empty-pool",
        ).start(wait=False)

    assert pod_calls == []


def test_on_demand_machine_releases_when_pod_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _, _ = fake_provider(FakePodInstance(ok=False, error_msg="no capacity"))
    patch_adapter_dependencies(monkeypatch, provider)
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        if "list" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"machines": [{"pool_name": "opentinker-test", "gpu": "A6000"}]}),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(beam_module.shutil, "which", lambda _name: "/usr/local/bin/beam")
    monkeypatch.setattr(beam_module.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="no capacity"):
        BeamComputeAdapter(
            base_model="Qwen/Qwen3-0.6B",
            gpu="A6000",
            on_demand=True,
            machine_name="opentinker-test",
        ).start(wait=False)

    assert "reserve" in commands[0]
    assert "list" in commands[1]
    assert commands[2][-5:] == [
        "machine",
        "release",
        "--pool",
        "opentinker-test",
        "--yes",
    ]


def test_on_demand_image_build_failure_does_not_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _, _ = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)
    commands: list[list[str]] = []

    monkeypatch.setattr(FakeImage, "build", lambda _self: SimpleNamespace(success=False))
    monkeypatch.setattr(beam_module.shutil, "which", lambda _name: "/usr/local/bin/beam")
    monkeypatch.setattr(
        beam_module.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command),
    )

    with pytest.raises(RuntimeError, match="failed to build"):
        BeamComputeAdapter(
            base_model="Qwen/Qwen3-0.6B",
            gpu="A16",
            on_demand=True,
        ).start(wait=False)

    assert commands == []


def test_interrupted_reservation_attempts_release(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _, _ = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        if "reserve" in command:
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=0, stdout="released", stderr="")

    monkeypatch.setattr(beam_module.shutil, "which", lambda _name: "/usr/local/bin/beam")
    monkeypatch.setattr(beam_module.subprocess, "run", run)

    with pytest.raises(KeyboardInterrupt):
        BeamComputeAdapter(
            base_model="Qwen/Qwen3-0.6B",
            profile="prod3",
            gpu="A16",
            on_demand=True,
            machine_name="opentinker-test",
        ).start(wait=False)

    assert "reserve" in commands[0]
    assert commands[1][-5:] == [
        "machine",
        "release",
        "--pool",
        "opentinker-test",
        "--yes",
    ]


def test_on_demand_without_gpu_uses_beam_picker_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, pod_calls, instance = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        if "list" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"machines": [{"pool_name": "opentinker-picked", "gpu": "H100"}]}
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(beam_module.shutil, "which", lambda _name: "/usr/local/bin/beam")
    monkeypatch.setattr(beam_module.subprocess, "run", run)

    adapter = BeamComputeAdapter(
        base_model="Qwen/Qwen3-0.6B",
        gpu=None,
        on_demand=True,
        machine_name="opentinker-picked",
    )
    adapter.start(wait=False)

    assert "--gpu" not in commands[0]
    assert "--yes" not in commands[0]
    assert adapter.gpu == "H100"
    assert pod_calls[0]["gpu"] == "H100"
    assert pod_calls[0]["pool"] == "opentinker-picked"
    assert adapter.stop() is True
    assert instance.terminate_calls == 1


def test_cancelling_on_demand_picker_does_not_create_or_release_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _, _ = fake_provider()
    patch_adapter_dependencies(monkeypatch, provider)
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        if "list" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"machines": []}),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(beam_module.shutil, "which", lambda _name: "/usr/local/bin/beam")
    monkeypatch.setattr(beam_module.subprocess, "run", run)

    adapter = BeamComputeAdapter(
        base_model="Qwen/Qwen3-0.6B",
        on_demand=True,
        machine_name="opentinker-cancelled",
    )
    with pytest.raises(RuntimeError, match="reservation was cancelled"):
        adapter.start(wait=False)

    assert len(commands) == 2
    assert "reserve" in commands[0]
    assert "list" in commands[1]
    assert adapter.pool is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_model": ""}, "base_model"),
        ({"provider": "other"}, "provider"),
        ({"profile": ""}, "profile"),
        ({"cpu": 0}, "cpu"),
        ({"memory": 0}, "memory"),
        ({"gpu": ""}, "gpu"),
        ({"gpu_count": 0}, "gpu_count"),
        ({"interconnect": "infiniband"}, "interconnect"),
        ({"pool_fallback": "internal"}, "pool_fallback"),
        ({"max_length": 0}, "max_length"),
        ({"volume_mount_path": "relative"}, "absolute"),
        ({"volume_verify_timeout": 0}, "volume_verify_timeout"),
        ({"port": 0}, "port"),
        ({"wait_timeout": 0}, "wait_timeout"),
        ({"poll_interval": 0}, "poll_interval"),
        ({"env": {"TOKEN": 1}}, "env"),
        ({"tinker_requirement": ""}, "tinker_requirement"),
        ({"on_demand": True, "pool": "existing"}, "mutually exclusive"),
    ],
)
def test_validates_configuration(kwargs: dict[str, Any], message: str) -> None:
    kwargs.setdefault("base_model", "Qwen/Qwen3-8B")
    with pytest.raises((TypeError, ValueError), match=message):
        BeamComputeAdapter(**kwargs)
