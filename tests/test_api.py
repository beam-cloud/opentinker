from __future__ import annotations

import time
from typing import Any, cast

import httpx
import pytest

from opentinker._api import FutureStore, create_app


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

    def get_sampler(self, sampling_session_id: str) -> dict[str, Any]:
        return {
            "sampler_id": sampling_session_id,
            "base_model": self.base_model,
            "model_path": None,
        }

    def sample(self, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "sample",
            "sequences": [{"stop_reason": "length", "tokens": [42], "logprobs": [-0.1]}],
            "prompt_cache_hit_tokens": 0,
        }

    def runtime_status(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"strategy": "single", "request": request}

    def prepare_shutdown(self) -> dict[str, Any]:
        return {
            "checkpoint_saved": False,
            "volume_paths": [],
            "checkpoints": [],
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
    shutdown_requests = 0

    def request_shutdown() -> None:
        nonlocal shutdown_requests
        shutdown_requests += 1

    transport = httpx.ASGITransport(
        app=create_app(ContractEngine(), request_shutdown=request_shutdown)
    )
    client = httpx.AsyncClient(transport=transport, base_url="http://test")

    health = (await client.get("/api/v1/healthz")).json()
    assert health["status"] == "ok"
    assert health["runtime"]["strategy"] == "unknown"
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
    shutdown = (await client.post("/opentinker/prepare-shutdown")).json()
    assert shutdown == {
        "checkpoint_saved": False,
        "volume_paths": [],
        "checkpoints": [],
    }
    assert shutdown_requests == 0
    finish = (await client.post("/opentinker/finish")).json()
    assert finish == shutdown
    assert shutdown_requests == 1

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
    sampler = (
        await client.get(
            f"/api/v1/samplers/{sampling_session['sampling_session_id']}",
        )
    ).json()
    assert sampler == {
        "sampler_id": "sampler-1",
        "base_model": "Qwen/Qwen3-0.6B",
        "model_path": None,
    }
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


def test_future_store_close_releases_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FutureStore()
    shutdown_calls: list[dict[str, bool]] = []
    store._futures["queued"] = cast(Any, object())
    monkeypatch.setattr(
        store._executor,
        "shutdown",
        lambda **kwargs: shutdown_calls.append(kwargs),
    )

    store.close()

    assert store._futures == {}
    assert shutdown_calls == [{"wait": False, "cancel_futures": True}]
