from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

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
    assert response["path"] == "tinker://model-1/sampler_weights/step-1"
    assert engine._uri_path(response["path"]) == tmp_path / "model-1/sampler_weights/step-1"
    shutdown = engine.prepare_shutdown()
    assert shutdown["checkpoint_saved"] is True
    assert 0 < shutdown["wait_seconds"] <= 60
    assert shutdown["volume_paths"] == [
        "tinker-checkpoints/checkpoints/model-1/sampler_weights/step-1"
    ]
    assert shutdown["checkpoints"] == [
        {
            "uri": "tinker://model-1/sampler_weights/step-1",
            "volume_path": ("tinker-checkpoints/checkpoints/model-1/sampler_weights/step-1"),
        }
    ]
    assert (checkpoint / "adapter_config.json").read_text() == "{}"
    assert (checkpoint / "adapter_model.bin").read_bytes() == b"weights"

    archive = engine.export_checkpoint({"path": response["path"]})
    try:
        with tarfile.open(archive, "r:gz") as handle:
            assert sorted(handle.getnames()) == [
                "checkpoint",
                "checkpoint/model-1",
                "checkpoint/model-1/adapter_config.json",
                "checkpoint/model-1/adapter_model.bin",
                "checkpoint/opentinker.json",
            ]
    finally:
        archive.unlink(missing_ok=True)
