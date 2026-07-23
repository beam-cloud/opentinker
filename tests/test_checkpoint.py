from __future__ import annotations

import json
from pathlib import Path

import pytest

from opentinker._checkpoint import CheckpointStore, safe_checkpoint_name


def test_publishes_resolves_and_reports_local_checkpoint(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, volume_name="training-output")

    def write(staging: Path) -> None:
        adapter = staging / "student"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.bin").write_bytes(b"weights")

    checkpoint = store.publish(
        model_id="student",
        kind="weights",
        name="step-8",
        overwrite=False,
        write=write,
    )

    assert checkpoint.uri == "tinker://student/weights/step-8"
    assert store.resolve(checkpoint.uri) == checkpoint.path
    assert (
        store.resolve("beam://training-output/checkpoints/student/weights/step-8")
        == checkpoint.path
    )
    manifest = json.loads((checkpoint.path / "opentinker-checksums.json").read_text())
    assert [item["path"] for item in manifest["files"]] == [
        "student/adapter_config.json",
        "student/adapter_model.bin",
    ]
    report = store.shutdown_report()
    assert report["checkpoint_saved"] is True
    assert report["checkpoints"][0]["verification"] == "local-fsync-sha256"
    assert report["checkpoints"][0]["file_count"] == 3


@pytest.mark.parametrize("value", ["", ".", "..", "../escape", "two/parts", r"two\parts"])
def test_checkpoint_names_are_single_path_components(value: str) -> None:
    with pytest.raises(ValueError, match="single non-empty path component"):
        safe_checkpoint_name(value)


def test_publish_rejects_model_id_traversal(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, volume_name="training-output")

    with pytest.raises(ValueError, match="model_id"):
        store.publish(
            model_id="../escape",
            kind="weights",
            name="step-1",
            overwrite=False,
            write=lambda _path: None,
        )


def test_resolve_rejects_foreign_and_malformed_handles(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, volume_name="training-output")

    for uri in (
        "beam://other/checkpoints/student/weights/step-1",
        "tinker://student/unknown/step-1",
        "tinker://../weights/step-1",
        "tinker://student/weights",
    ):
        with pytest.raises(ValueError):
            store.resolve(uri)
