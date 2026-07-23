from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from opentinker import _checkpoint as checkpoint_module
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
    assert engine.get_sampler(response["sampling_session_id"]) == {
        "sampler_id": response["sampling_session_id"],
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
    expected_model = TinyModel()
    expected_model.load_state_dict(model.state_dict())
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

    expected_logits = expected_model(
        input_ids=torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
    ).logits
    expected_logprobs = torch.log_softmax(expected_logits, dim=-1).gather(
        -1,
        torch.tensor([[2, 3], [0, 0]], dtype=torch.long).unsqueeze(-1),
    )
    expected_loss = -(
        expected_logprobs.squeeze(-1)
        * torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    ).sum()
    expected_loss.backward()

    response = engine.forward_backward(request, backward=True)

    assert [item["logprobs"]["shape"] for item in response["loss_fn_outputs"]] == [[2], [1]]
    assert response["metrics"]["loss:sum"] == pytest.approx(float(expected_loss.detach()))
    assert response["metrics"]["loss:mean"] == pytest.approx(float(expected_loss.detach()) / 2)
    assert model.projection.weight.grad is not None
    assert expected_model.projection.weight.grad is not None
    torch.testing.assert_close(
        model.projection.weight.grad,
        expected_model.projection.weight.grad,
    )
    assert "_distributed_indices" not in response


def test_importance_sampling_sums_token_objectives_without_weights(
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
            torch.nn.init.zeros_(self.projection.weight)

        def forward(self, *, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
            inputs = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
            return SimpleNamespace(logits=self.projection(inputs))

        def set_adapter(self, _model_id: str) -> None:
            return

    model = TinyModel()
    expected_model = TinyModel()
    expected_model.load_state_dict(model.state_dict())
    engine._training_model = model
    engine._models["model-1"] = {}
    monkeypatch.setattr(engine, "_imports", lambda: (torch, None, None, None))

    def tensor(values: list[int] | list[float], dtype: str) -> dict[str, object]:
        return {"data": values, "dtype": dtype, "shape": [len(values)]}

    old_logprob = -float(torch.log(torch.tensor(4.0)))
    request = {
        "model_id": "model-1",
        "forward_backward_input": {
            "loss_fn": "importance_sampling",
            "data": [
                {
                    "model_input": {"chunks": [{"tokens": [0, 1, 2]}]},
                    "loss_fn_inputs": {
                        "target_tokens": tensor([1, 2, 3], "int64"),
                        "logprobs": tensor([old_logprob, old_logprob, old_logprob], "float32"),
                        "advantages": tensor([1.5, -0.25, 2.0], "float32"),
                        # Importance sampling has no weights input. An extra
                        # weights field must not alter its token-sum objective.
                        "weights": tensor([10.0, 0.0, 0.0], "float32"),
                    },
                },
            ],
        },
    }

    expected_logits = expected_model(input_ids=torch.tensor([[0, 1, 2]])).logits
    expected_logprobs = torch.log_softmax(expected_logits, dim=-1).gather(
        -1,
        torch.tensor([[1, 2, 3]], dtype=torch.long).unsqueeze(-1),
    )
    expected_loss = -(
        torch.exp(expected_logprobs.squeeze(-1) - old_logprob)
        * torch.tensor([[1.5, -0.25, 2.0]])
    ).sum()
    expected_loss.backward()

    response = engine.forward_backward(request, backward=True)

    assert response["loss_fn_output_type"] == "ImportanceSamplingLossReturn"
    assert response["metrics"]["loss:sum"] == pytest.approx(float(expected_loss.detach()))
    assert response["metrics"]["loss:mean"] == pytest.approx(float(expected_loss.detach()))
    assert model.projection.weight.grad is not None
    assert expected_model.projection.weight.grad is not None
    torch.testing.assert_close(
        model.projection.weight.grad,
        expected_model.projection.weight.grad,
    )


def test_distributed_backward_preserves_global_summed_objective(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BiasModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.1, 0.0]))

        def forward(self, *, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                logits=self.logits.expand(*input_ids.shape, self.logits.shape[0])
            )

        def set_adapter(self, _model_id: str) -> None:
            return

    def tensor(value: int | float, dtype: str) -> dict[str, object]:
        return {"data": [value], "dtype": dtype, "shape": [1]}

    data = [
        {
            "model_input": {"chunks": [{"tokens": [0]}]},
            "loss_fn_inputs": {
                "target_tokens": tensor(target, "int64"),
                "weights": tensor(weight, "float32"),
            },
        }
        for target, weight in [(0, 2.0), (1, 3.0), (2, 4.0)]
    ]
    request = {
        "model_id": "model-1",
        "forward_backward_input": {"loss_fn": "cross_entropy", "data": data},
    }

    expected_model = BiasModel()
    expected_logprobs = torch.log_softmax(expected_model.logits, dim=-1)
    expected_loss = -(
        expected_logprobs[torch.tensor([0, 1, 2])] * torch.tensor([2.0, 3.0, 4.0])
    ).sum()
    expected_loss.backward()
    global_loss = float(expected_loss.detach())

    reduced_local_losses: list[float] = []

    def all_reduce(value: torch.Tensor) -> None:
        reduced_local_losses.append(float(value))
        value.fill_(global_loss)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    models: list[BiasModel] = []
    responses: list[dict[str, object]] = []
    for rank in range(2):
        engine = TransformersEngine(
            base_model="test/model",
            checkpoint_root=str(tmp_path / str(rank)),
            sampling_gpu=False,
            device="cpu",
            distributed_rank=rank,
            distributed_world_size=2,
        )
        model = BiasModel()
        model.load_state_dict(expected_model.state_dict())
        engine._training_model = model
        engine._models["model-1"] = {}
        engine._distributed_adapter = "model-1"
        monkeypatch.setattr(engine, "_imports", lambda: (torch, None, None, None))

        responses.append(engine.forward_backward(request, backward=True))
        models.append(model)

    expected_local_losses = [
        -float((expected_logprobs[0] * 2.0 + expected_logprobs[2] * 4.0).detach()),
        -float((expected_logprobs[1] * 3.0).detach()),
    ]
    assert reduced_local_losses == pytest.approx(expected_local_losses)
    for response in responses:
        metrics = response["metrics"]
        assert isinstance(metrics, dict)
        assert metrics["loss:sum"] == pytest.approx(global_loss)
        assert metrics["loss:mean"] == pytest.approx(global_loss / len(data))

    # These are the pre-reduction gradients from each rank. DDP averages them,
    # so the engine's world-size compensation must make their average equal
    # the gradient of the global summed objective.
    rank_zero_gradient = models[0].logits.grad
    rank_one_gradient = models[1].logits.grad
    expected_gradient = expected_model.logits.grad
    assert rank_zero_gradient is not None
    assert rank_one_gradient is not None
    assert expected_gradient is not None
    ddp_averaged_gradient = (rank_zero_gradient + rank_one_gradient) / 2
    torch.testing.assert_close(ddp_averaged_gradient, expected_gradient)


def test_qwen35_linear_attention_projections_are_lora_targets(tmp_path: Path) -> None:
    engine = TransformersEngine(
        base_model="test/qwen3.5",
        checkpoint_root=str(tmp_path),
        sampling_gpu=False,
        device="cpu",
    )

    class TinyQwen35(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(2, 2)
            self.in_proj_qkv = torch.nn.Linear(2, 2)
            self.in_proj_z = torch.nn.Linear(2, 2)
            self.in_proj_b = torch.nn.Linear(2, 2)
            self.in_proj_a = torch.nn.Linear(2, 2)
            self.out_proj = torch.nn.Linear(2, 2)
            self.gate_proj = torch.nn.Linear(2, 2)
            self.lm_head = torch.nn.Linear(2, 2)

    assert engine._target_modules(TinyQwen35(), {}) == [
        "gate_proj",
        "in_proj_a",
        "in_proj_b",
        "in_proj_qkv",
        "in_proj_z",
        "lm_head",
        "out_proj",
        "q_proj",
    ]


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
    monkeypatch.setattr(checkpoint_module, "_geesefs_version", lambda _path: "0.42-test")
    checkpoint_xattrs: dict[str, dict[str, bytes]] = {}

    def setxattr(path: Path, name: str, value: bytes) -> None:
        checkpoint_xattrs.setdefault(str(path), {})[name] = value

    def getxattr(path: Path, name: str) -> bytes:
        if name == "s3.etag":
            return b'"remote-etag"'
        return checkpoint_xattrs[str(path)][name]

    monkeypatch.setattr(checkpoint_module.os, "setxattr", setxattr, raising=False)
    monkeypatch.setattr(checkpoint_module.os, "getxattr", getxattr, raising=False)
    monkeypatch.setattr(
        checkpoint_module.os,
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
