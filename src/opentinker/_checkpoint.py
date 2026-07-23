# pyright: reportUnknownMemberType=false

"""Checkpoint naming, publication, resolution, and durability reporting."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

CheckpointKind = Literal["weights", "sampler_weights"]
CheckpointWriter = Callable[[Path], None]

_CHECKSUM_MANIFEST = "opentinker-checksums.json"
_GEESFS_HASH_XATTR = "user.--content-sha256"


@dataclass(frozen=True)
class PublishedCheckpoint:
    """A durable checkpoint and the proof returned to the adapter."""

    uri: str
    path: Path
    proof: Mapping[str, Any]


class CheckpointStore:
    """Own checkpoint paths and publish local staging trees to a mounted Volume."""

    def __init__(self, root: str | Path, *, volume_name: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.volume_name = volume_name
        self._published: dict[str, dict[str, Any]] = {}

    def publish(
        self,
        *,
        model_id: str,
        kind: CheckpointKind,
        name: object,
        overwrite: bool,
        write: CheckpointWriter,
    ) -> PublishedCheckpoint:
        """Stage one checkpoint locally, then synchronously publish it."""

        model_component = _safe_path_component(model_id, label="model_id")
        checkpoint_name = safe_checkpoint_name(name)
        destination = self.root / model_component / kind / checkpoint_name
        if destination.exists() and not overwrite:
            raise FileExistsError(f"checkpoint already exists: {checkpoint_name}")
        destination.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="tinker-checkpoint-") as temp_dir:
            staging = Path(temp_dir)
            write(staging)
            expected_files = _write_manifest(staging)
            if destination.exists():
                shutil.rmtree(destination)
            try:
                files = _publish_tree(
                    staging,
                    destination,
                    geesefs_version=_geesefs_version(self.root),
                    expected_files=expected_files,
                )
            except BaseException:
                if destination.exists():
                    shutil.rmtree(destination)
                raise

        uri = f"tinker://{model_component}/{kind}/{checkpoint_name}"
        volume_path = f"{self.volume_name}/checkpoints/{model_component}/{kind}/{checkpoint_name}"
        manifest = next(item for item in files if item["path"] == _CHECKSUM_MANIFEST)
        proof: dict[str, Any] = {
            "volume_path": volume_path,
            "manifest_sha256": manifest["sha256"],
            "verified": True,
            "verification": (
                "geesefs-fsync-sha256" if manifest["etag"] is not None else "local-fsync-sha256"
            ),
            "geesefs_version": manifest["geesefs_version"],
            "file_count": len(files),
            "total_bytes": sum(int(item["size"]) for item in files),
            "files": files,
        }
        self._published[uri] = proof
        return PublishedCheckpoint(uri=uri, path=destination, proof=proof)

    def resolve(self, uri: str) -> Path:
        """Resolve and validate a public checkpoint handle."""

        beam_prefix = f"beam://{self.volume_name}/checkpoints/"
        if uri.startswith(beam_prefix):
            parts = uri.removeprefix(beam_prefix).split("/")
        elif uri.startswith("tinker://"):
            parts = uri.removeprefix("tinker://").split("/")
        else:
            raise ValueError(f"checkpoint path must start with {beam_prefix!r} or 'tinker://'")
        if len(parts) != 3 or parts[1] not in {"weights", "sampler_weights"}:
            raise ValueError(f"invalid checkpoint path: {uri}")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"invalid checkpoint path: {uri}")
        path = self.root.joinpath(*parts)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {uri}")
        return path

    def shutdown_report(self) -> dict[str, Any]:
        """Return durability proofs for all checkpoints published by this process."""

        checkpoints = [{"uri": uri, **proof} for uri, proof in sorted(self._published.items())]
        return {
            "checkpoint_saved": bool(checkpoints),
            "volume_paths": [str(checkpoint["volume_path"]) for checkpoint in checkpoints],
            "checkpoints": checkpoints,
        }


def safe_checkpoint_name(value: object) -> str:
    """Validate the one-component checkpoint names accepted by Tinker."""

    return _safe_path_component(value, label="checkpoint name")


def _safe_path_component(value: object, *, label: str) -> str:
    name = str(value)
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"{label} must be a single non-empty path component")
    return name


def _write_manifest(staging: Path) -> list[dict[str, Any]]:
    payload_files = [
        _file_record(item, relative_to=staging)
        for item in sorted(staging.rglob("*"))
        if item.is_file()
    ]
    manifest_path = staging / _CHECKSUM_MANIFEST
    manifest_path.write_text(
        json.dumps({"format_version": 1, "files": payload_files}, indent=2) + "\n"
    )
    return [*payload_files, _file_record(manifest_path, relative_to=staging)]


def _geesefs_version(path: Path) -> str | None:
    try:
        return os.getxattr(path, "geesefs").decode()
    except (AttributeError, OSError):
        return None


def _file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _publish_tree(
    source: Path,
    destination: Path,
    *,
    geesefs_version: str | None,
    expected_files: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_by_path = {str(item["path"]): item for item in expected_files}
    if len(expected_by_path) != len(expected_files):
        raise RuntimeError("checkpoint contains duplicate manifest paths")

    destination.mkdir(parents=True)
    records = [
        _publish_file(
            source_file,
            source=source,
            destination=destination,
            expected_by_path=expected_by_path,
            geesefs_version=geesefs_version,
        )
        for source_file in sorted(source.rglob("*"))
        if source_file.is_file()
    ]
    if expected_by_path:
        missing = ", ".join(sorted(expected_by_path))
        raise RuntimeError(f"checkpoint manifest referenced missing files: {missing}")
    _fsync_directory(destination)
    return records


def _publish_file(
    source_file: Path,
    *,
    source: Path,
    destination: Path,
    expected_by_path: dict[str, dict[str, Any]],
    geesefs_version: str | None,
) -> dict[str, Any]:
    relative = source_file.relative_to(source)
    destination_file = destination / relative
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    checksum, size = _copy_and_fsync(
        source_file,
        destination_file,
        attach_checksum=geesefs_version is not None,
    )
    etag = (
        _verify_geesefs_metadata(destination_file, checksum)
        if geesefs_version is not None
        else None
    )
    expected = expected_by_path.pop(relative.as_posix(), None)
    if expected is None:
        raise RuntimeError(f"checkpoint manifest omitted {relative.as_posix()}")
    if size != expected["size"] or checksum != expected["sha256"]:
        raise RuntimeError(f"checkpoint copy changed {relative.as_posix()}")
    return {**expected, "etag": etag, "geesefs_version": geesefs_version}


def _copy_and_fsync(
    source: Path,
    destination: Path,
    *,
    attach_checksum: bool,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as reader, destination.open("wb") as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        writer.flush()
        checksum = digest.hexdigest()
        if attach_checksum:
            os.setxattr(destination, _GEESFS_HASH_XATTR, checksum.encode())
        os.fsync(writer.fileno())
    return checksum, size


def _verify_geesefs_metadata(path: Path, checksum: str) -> str:
    stored_checksum = os.getxattr(path, _GEESFS_HASH_XATTR).decode()
    if stored_checksum != checksum:
        raise RuntimeError(f"geesefs checksum metadata mismatch for {path.name}")
    etag_names = [name for name in os.listxattr(path) if name.endswith(".etag")]
    if len(etag_names) != 1:
        raise RuntimeError(f"geesefs did not return one remote ETag for {path.name}")
    etag = os.getxattr(path, etag_names[0]).decode()
    if not etag:
        raise RuntimeError(f"geesefs returned an empty remote ETag for {path.name}")
    return etag


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["CheckpointKind", "CheckpointStore", "PublishedCheckpoint"]
