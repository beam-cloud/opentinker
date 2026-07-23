"""Reproducible Beam/Beta9 backend image construction."""

from __future__ import annotations

import importlib.metadata
import os
import shlex
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

_REPRODUCIBLE_MTIME = 315619200  # 1980-01-02 UTC; ZIP-safe in western timezones.


@dataclass(frozen=True)
class BackendImageSpec:
    base_image: str
    tinker_requirement: str | None
    python_packages: Sequence[str]
    commands: Sequence[str]


def build_backend_image(
    provider: ModuleType,
    spec: BackendImageSpec,
    *,
    package_source: Path | None = None,
) -> Any:
    """Build an SDK image containing the OpenTinker server package."""

    if any(character in spec.base_image for character in "\r\n"):
        raise ValueError("base_image must not contain newlines")
    packages = [
        spec.tinker_requirement or f"tinker=={importlib.metadata.version('tinker')}",
        # Match the current Cookbook runtime. Transformers 4.x predates
        # Qwen3.5; PEFT 0.18.1 is the first patch release with its complete
        # Transformers 5 compatibility fixes.
        "peft>=0.18.1,<0.19",
        "transformers==5.5.4",
        "fastapi>=0.115,<1",
        "uvicorn[standard]>=0.34,<1",
        *spec.python_packages,
    ]

    source = package_source or Path(__file__).parent
    with tempfile.TemporaryDirectory(prefix="opentinker-image-") as temp_directory:
        context = Path(temp_directory)
        shutil.copytree(
            source,
            context / "opentinker",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        install = " ".join(shlex.quote(package) for package in packages)
        dockerfile_lines = [
            f"FROM {spec.base_image}",
            f"RUN python -m pip install --no-cache-dir {install}",
            *(f"RUN /bin/bash -lc {shlex.quote(command)}" for command in spec.commands),
            "COPY opentinker /opt/opentinker/opentinker",
            "ENV PYTHONPATH=/opt/opentinker",
        ]
        dockerfile = context / "Dockerfile"
        dockerfile.write_text("\n".join(dockerfile_lines) + "\n")
        _normalize_timestamps(context)
        image = provider.Image.from_dockerfile(
            str(dockerfile),
            context_dir=str(context),
        )
        # A Dockerfile owns its Python runtime; the SDK must not layer the
        # notebook kernel's interpreter on top.
        image.ignore_python = True
        if _is_notebook_process():
            _patch_sdk_notebook_version(image)
        return image


def _normalize_timestamps(context: Path) -> None:
    """Keep provider build-context hashes stable across equivalent runs."""

    paths = sorted(context.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in paths:
        os.utime(path, (_REPRODUCIBLE_MTIME, _REPRODUCIBLE_MTIME))
    os.utime(context, (_REPRODUCIBLE_MTIME, _REPRODUCIBLE_MTIME))


def _is_notebook_process() -> bool:
    return any(
        module in sys.modules
        for module in ("google.colab", "ipykernel.zmqshell", "marimo")
    )


class _PythonVersionString(str):
    """Compatibility value for SDKs that call ``.value`` on a string."""

    @property
    def value(self) -> str:
        return str(self)


def _patch_sdk_notebook_version(image: Any) -> None:
    """Avoid the Beta9 0.1.265 notebook warning crash on Python 3.13."""

    sdk_image_module = sys.modules.get(type(image).__module__)
    if sdk_image_module is None:
        return
    local_version = getattr(sdk_image_module, "LOCAL_PYTHON_VERSION", None)
    if isinstance(local_version, str) and not hasattr(local_version, "value"):
        vars(sdk_image_module)["LOCAL_PYTHON_VERSION"] = _PythonVersionString(local_version)


__all__ = ["BackendImageSpec", "build_backend_image"]
