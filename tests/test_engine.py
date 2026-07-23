from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from opentinker import _engine as engine_module
from opentinker._engine import TransformersEngine


def test_sampling_session_accepts_distillation_teacher(tmp_path: Path) -> None:
    engine = TransformersEngine(
        base_model="Qwen/Qwen3-0.6B",
        checkpoint_root=str(tmp_path),
        sampling_gpu=False,
    )

    response = engine.create_sampling_session({"base_model": "Qwen/Qwen3-4B-Instruct-2507"})

    session = engine._sampling_sessions[response["sampling_session_id"]]
    assert session == {
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "model_path": None,
    }


def test_forward_backward_batches_variable_length_datums(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = TransformersEngine(
        base_model="test/model",
        checkpoint_root=str(tmp_path),
        sampling_gpu=False,
        device="cpu",
    )

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(4, 4, bias=False)

        def forward(self, *, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
            inputs = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
            return SimpleNamespace(logits=self.projection(inputs))

        def set_adapter(self, _model_id: str) -> None:
            return

    model = TinyModel()
    engine._training_model = model
    engine._models["model-1"] = {}
    monkeypatch.setattr(engine, "_imports", lambda: (torch, None, None, None))

    def tensor(values: list[int] | list[float], dtype: str) -> dict[str, object]:
        return {"data": values, "dtype": dtype, "shape": [len(values)]}

    request = {
        "model_id": "model-1",
        "forward_backward_input": {
            "loss_fn": "cross_entropy",
            "data": [
                {
                    "model_input": {"chunks": [{"tokens": [1, 2]}]},
                    "loss_fn_inputs": {
                        "target_tokens": tensor([2, 3], "int64"),
                        "weights": tensor([1.0, 1.0], "float32"),
                    },
                },
                {
                    "model_input": {"chunks": [{"tokens": [3]}]},
                    "loss_fn_inputs": {
                        "target_tokens": tensor([0], "int64"),
                        "weights": tensor([1.0], "float32"),
                    },
                },
            ],
        },
    }

    response = engine.forward_backward(request, backward=True)

    assert [item["logprobs"]["shape"] for item in response["loss_fn_outputs"]] == [[2], [1]]
    assert response["metrics"]["loss:mean"] > 0
    assert model.projection.weight.grad is not None
    assert "_distributed_indices" not in response


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
    monkeypatch.setattr(engine_module, "_geesefs_version", lambda _path: "0.42-test")
    checkpoint_xattrs: dict[str, dict[str, bytes]] = {}

    def setxattr(path: Path, name: str, value: bytes) -> None:
        checkpoint_xattrs.setdefault(str(path), {})[name] = value

    def getxattr(path: Path, name: str) -> bytes:
        if name == "s3.etag":
            return b'"remote-etag"'
        return checkpoint_xattrs[str(path)][name]

    monkeypatch.setattr(engine_module.os, "setxattr", setxattr, raising=False)
    monkeypatch.setattr(engine_module.os, "getxattr", getxattr, raising=False)
    monkeypatch.setattr(
        engine_module.os,
        "listxattr",
        lambda _path: ["user.--content-sha256", "s3.etag"],
        raising=False,
    )
    engine._models[model_id] = {
        "rank": 8,
        "train_mlp": True,
        "train_attn": True,
        "train_unembed": True,
    }

    response = engine.save_weights({"model_id": model_id, "path": "step-1"}, for_sampler=True)

    checkpoint = tmp_path / model_id / "sampler_weights" / "step-1" / model_id
    assert response["path"] == "tinker://model-1/sampler_weights/step-1"
    assert engine._uri_path(response["path"]) == tmp_path / "model-1/sampler_weights/step-1"
    shutdown = engine.prepare_shutdown()
    assert shutdown["checkpoint_saved"] is True
    assert shutdown["volume_paths"] == [
        "tinker-checkpoints/checkpoints/model-1/sampler_weights/step-1"
    ]
    saved = shutdown["checkpoints"][0]
    assert saved["uri"] == "tinker://model-1/sampler_weights/step-1"
    assert saved["volume_path"] == ("tinker-checkpoints/checkpoints/model-1/sampler_weights/step-1")
    assert len(saved["manifest_sha256"]) == 64
    assert saved["verified"] is True
    assert saved["verification"] == "geesefs-fsync-sha256"
    assert saved["geesefs_version"] == "0.42-test"
    assert saved["file_count"] == 4
    assert all(item["etag"] == '"remote-etag"' for item in saved["files"])
    assert (checkpoint / "adapter_config.json").read_text() == "{}"
    assert (checkpoint / "adapter_model.bin").read_bytes() == b"weights"
    assert (checkpoint.parent / "opentinker-checksums.json").is_file()
