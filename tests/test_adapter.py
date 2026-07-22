from __future__ import annotations

import time
from importlib.metadata import version
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest
import tinker

from opentinker import adapter as beam_module
from opentinker._server import FutureStore, TransformersEngine, create_app
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
        ok: bool = True,
        error_msg: str = "",
    ) -> None:
        self.container_id = "pod-123"
        self.url = url
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

        def create(self) -> FakePodInstance:
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
    return clients


def test_starts_backend_and_returns_normal_client(monkeypatch: pytest.MonkeyPatch) -> None:
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
    client = adapter.start(wait=False)

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
    assert options["allow_marketplace"] is True
    assert options["secrets"] == ["HF_TOKEN"]
    assert options["env"] == {
        "OPENTINKER_BASE_MODEL": "Qwen/Qwen3-8B",
        "OPENTINKER_CHECKPOINT_ROOT": "/tinker-data/checkpoints",
        "OPENTINKER_VOLUME_NAME": "tinker-checkpoints",
        "OPENTINKER_MAX_LENGTH": "8192",
        "OPENTINKER_PORT": "8000",
        "OPENTINKER_SAMPLING_GPU": "1",
        "OPENTINKER_TRUST_REMOTE_CODE": "0",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    assert options["entrypoint"][-1].endswith("python -m opentinker._server")
    assert options["volumes"][0].mount_path == "/tinker-data"

    assert adapter.stop() is True
    assert clients[0].holder.close_calls == 1
    assert instance.terminate_calls == 1
    assert adapter.stop() is True
    assert instance.terminate_calls == 1


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
    assert instance.terminate_calls == 1


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
    assert "'peft>=0.14,<1'" in dockerfile
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
        "opentinker/_server.py",
        "opentinker/adapter.py",
        "opentinker/py.typed",
    ]
    assert image.kwargs["context_mtimes"] == {beam_module._REPRODUCIBLE_MTIME}


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
    responses = [httpx.Response(503, text="loading"), httpx.Response(200, json={"status": "ok"})]

    def get(url: str, *, headers: dict[str, str], timeout: int) -> httpx.Response:
        assert url == "https://pod.example.test/api/v1/healthz"
        assert headers == {"Authorization": "Bearer provider-token"}
        assert timeout == 10
        assert not clients
        return responses.pop(0)

    monkeypatch.setattr(beam_module.httpx, "get", get)
    monkeypatch.setattr(beam_module.time, "sleep", lambda _: None)

    BeamComputeAdapter(base_model="Qwen/Qwen3-8B").start()

    assert len(clients) == 1


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
        "--gpu",
        "A6000",
        "--nodes",
        "1",
        "--ttl",
        "45m",
        "--name",
        "opentinker-test",
        "--yes",
    ]
    assert pod_calls[0]["pool"] == "opentinker-test"
    assert pod_calls[0]["image"].build_calls == 1
    assert pod_calls[0]["image"].ignore_python is True

    assert adapter.stop() is True
    assert instance.terminate_calls == 1
    assert commands[1] == [
        "/usr/local/bin/beam",
        "--context",
        "prod3",
        "machine",
        "release",
        "--pool",
        "opentinker-test",
        "--yes",
    ]


def test_on_demand_machine_releases_when_pod_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _, _ = fake_provider(FakePodInstance(ok=False, error_msg="no capacity"))
    patch_adapter_dependencies(monkeypatch, provider)
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(beam_module.shutil, "which", lambda _name: "/usr/local/bin/beam")
    monkeypatch.setattr(beam_module.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="no capacity"):
        BeamComputeAdapter(
            base_model="Qwen/Qwen3-0.6B",
            gpu="A6000",
            on_demand=True,
        ).start(wait=False)

    assert [command[3:5] for command in commands] == [
        ["--gpu", "A6000"],
        ["--pool", "opentinker-a6000"],
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
        ).start(wait=False)

    assert "reserve" in commands[0]
    assert commands[1][-5:] == [
        "machine",
        "release",
        "--pool",
        "opentinker-a16",
        "--yes",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_model": ""}, "base_model"),
        ({"provider": "other"}, "provider"),
        ({"profile": ""}, "profile"),
        ({"cpu": 0}, "cpu"),
        ({"memory": 0}, "memory"),
        ({"gpu": ""}, "gpu"),
        ({"max_length": 0}, "max_length"),
        ({"volume_mount_path": "relative"}, "absolute"),
        ({"port": 0}, "port"),
        ({"wait_timeout": 0}, "wait_timeout"),
        ({"poll_interval": 0}, "poll_interval"),
        ({"env": {"TOKEN": 1}}, "env"),
        ({"tinker_requirement": ""}, "tinker_requirement"),
    ],
)
def test_validates_configuration(kwargs: dict[str, Any], message: str) -> None:
    kwargs.setdefault("base_model", "Qwen/Qwen3-8B")
    with pytest.raises((TypeError, ValueError), match=message):
        BeamComputeAdapter(**kwargs)


class ContractEngine:
    base_model = "Qwen/Qwen3-0.6B"
    max_length = 4096

    def create_model(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"type": "create_model", "model_id": request["_model_id"]}

    def get_info(self, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_id": request["model_id"],
            "is_lora": True,
            "lora_rank": 8,
            "model_data": {"model_name": self.base_model, "tokenizer_id": self.base_model},
        }

    def weights_info(self, request: dict[str, Any]) -> dict[str, Any]:
        assert request["tinker_path"]
        return {
            "base_model": self.base_model,
            "is_lora": True,
            "lora_rank": 8,
            "train_mlp": True,
            "train_attn": True,
            "train_unembed": True,
        }

    def forward_backward(self, request: dict[str, Any], *, backward: bool) -> dict[str, Any]:
        key = "forward_backward_input" if backward else "forward_input"
        assert request[key]["loss_fn"] == "cross_entropy"
        return {
            "loss_fn_output_type": "CrossEntropyLossReturn",
            "loss_fn_outputs": [{"logprobs": {"data": [-1.0], "dtype": "float32", "shape": [1]}}],
            "metrics": {"loss:mean": 1.0},
        }

    def optim_step(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"metrics": {"learning_rate": request["adam_params"]["learning_rate"]}}

    def save_weights(self, request: dict[str, Any], *, for_sampler: bool) -> dict[str, Any]:
        kind = "sampler_weights" if for_sampler else "weights"
        return {"path": f"tinker://{request['model_id']}/{kind}/test"}

    def load_weights(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"type": "load_weights", "path": request["path"]}

    def unload_model(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"type": "unload_model", "model_id": request["model_id"]}

    def create_sampling_session(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"type": "create_sampling_session", "sampling_session_id": "sampler-1"}

    def sample(self, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "sample",
            "sequences": [{"stop_reason": "length", "tokens": [42], "logprobs": [-0.1]}],
            "prompt_cache_hit_tokens": 0,
        }


async def retrieve(client: httpx.AsyncClient, future: dict[str, Any]) -> dict[str, Any]:
    for _ in range(100):
        response = (
            await client.post(
                "/api/v1/retrieve_future",
                json={"request_id": future["request_id"]},
            )
        ).json()
        if response.get("type") != "try_again":
            return response
        time.sleep(0.001)
    raise AssertionError("future did not resolve")


async def test_owned_server_implements_tinker_contract() -> None:
    transport = httpx.ASGITransport(app=create_app(ContractEngine()))
    client = httpx.AsyncClient(transport=transport, base_url="http://test")

    assert (await client.get("/api/v1/healthz")).json() == {"status": "ok"}
    capabilities = (await client.get("/api/v1/get_server_capabilities")).json()
    assert capabilities["supported_models"][0]["model_name"] == "Qwen/Qwen3-0.6B"
    config = (await client.post("/api/v1/client/config", json={"sdk_version": "test"})).json()
    assert config["proto_write_fwdbwd"] is False
    weights_info = (
        await client.post(
            "/api/v1/weights_info",
            json={"tinker_path": "beam://tinker-checkpoints/checkpoints/model/weights/final"},
        )
    ).json()
    assert weights_info["base_model"] == "Qwen/Qwen3-0.6B"
    assert weights_info["lora_rank"] == 8
    session = (await client.post("/api/v1/create_session", json={})).json()
    assert session["type"] == "create_session"
    telemetry = (await client.post("/api/v1/telemetry", json={"events": []})).json()
    assert telemetry == {"status": "accepted"}

    create_future = (
        await client.post(
            "/api/v1/create_model",
            json={"base_model": "Qwen/Qwen3-0.6B", "lora_config": {"rank": 8}},
        )
    ).json()
    created = await retrieve(client, create_future)
    model_id = created["model_id"]
    assert create_future["model_id"] == model_id

    request = {
        "model_id": model_id,
        "forward_backward_input": {
            "loss_fn": "cross_entropy",
            "data": [
                {
                    "model_input": {"chunks": [{"tokens": [1]}]},
                    "loss_fn_inputs": {
                        "target_tokens": {"data": [2], "dtype": "int64", "shape": [1]},
                        "weights": {"data": [1.0], "dtype": "float32", "shape": [1]},
                    },
                }
            ],
        },
    }
    output = await retrieve(
        client, (await client.post("/api/v1/forward_backward", json=request)).json()
    )
    assert output["metrics"] == {"loss:mean": 1.0}
    optim = await retrieve(
        client,
        (
            await client.post(
                "/api/v1/optim_step",
                json={"model_id": model_id, "adam_params": {"learning_rate": 0.001}},
            )
        ).json(),
    )
    assert optim["metrics"] == {"learning_rate": 0.001}
    saved = await retrieve(
        client,
        (await client.post("/api/v1/save_weights", json={"model_id": model_id})).json(),
    )
    assert saved["path"].startswith("tinker://")
    sampling_session = (
        await client.post(
            "/api/v1/create_sampling_session",
            json={"session_id": session["session_id"], "base_model": "Qwen/Qwen3-0.6B"},
        )
    ).json()
    sample = await retrieve(
        client,
        (
            await client.post(
                "/api/v1/asample",
                json={
                    "sampling_session_id": sampling_session["sampling_session_id"],
                    "prompt": {"chunks": [{"tokens": [1]}]},
                    "sampling_params": {"max_tokens": 1},
                },
            )
        ).json(),
    )
    assert sample["sequences"][0]["tokens"] == [42]
    await client.aclose()


async def test_pending_future_uses_quiet_http_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def pending(
        _self: FutureStore,
        request_id: str,
        *,
        wait_timeout: float = 30,
    ) -> dict[str, Any]:
        assert request_id == "still-running"
        assert wait_timeout == 30
        return {
            "type": "try_again",
            "request_id": request_id,
            "queue_state": "active",
        }

    monkeypatch.setattr(FutureStore, "retrieve", pending)
    transport = httpx.ASGITransport(app=create_app(ContractEngine()))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/retrieve_future",
            json={"request_id": "still-running"},
        )

    assert response.status_code == 408
    assert response.json()["type"] == "try_again"


def test_checkpoint_is_staged_locally_before_volume_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_id = "model-1"
    engine = TransformersEngine(
        base_model="Qwen/Qwen3-0.6B",
        checkpoint_root=str(tmp_path),
        sampling_gpu=False,
    )

    class FakeModel:
        def save_pretrained(
            self,
            path: Path,
            *,
            selected_adapters: list[str],
            safe_serialization: bool,
        ) -> None:
            assert selected_adapters == [model_id]
            assert safe_serialization is False
            assert not str(path).startswith(str(tmp_path))
            adapter_path = path / model_id
            adapter_path.mkdir(parents=True)
            (adapter_path / "adapter_config.json").write_text("{}")
            (adapter_path / "adapter_model.bin").write_bytes(b"weights")

    monkeypatch.setattr(engine, "_imports", lambda: (None, None, None, None))
    monkeypatch.setattr(engine, "_activate", lambda _model_id: FakeModel())
    engine._models[model_id] = {
        "rank": 8,
        "train_mlp": True,
        "train_attn": True,
        "train_unembed": True,
    }

    response = engine.save_weights({"model_id": model_id, "path": "step-1"}, for_sampler=True)

    checkpoint = tmp_path / model_id / "sampler_weights" / "step-1" / model_id
    assert response["path"] == (
        "beam://tinker-checkpoints/checkpoints/model-1/sampler_weights/step-1"
    )
    assert (checkpoint / "adapter_config.json").read_text() == "{}"
    assert (checkpoint / "adapter_model.bin").read_bytes() == b"weights"
