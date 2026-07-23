# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUntypedFunctionDecorator=false, reportUnusedFunction=false

"""Tinker HTTP compatibility application.

The application factory depends only on the :class:`ComputeEngine` protocol,
which keeps transport concerns independent from the GPU implementation.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Protocol


class ComputeEngine(Protocol):
    """Operations exposed by the HTTP compatibility layer."""

    base_model: str
    max_length: int

    def create_model(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def get_info(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def weights_info(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def forward_backward(self, request: dict[str, Any], *, backward: bool) -> dict[str, Any]: ...

    def optim_step(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def save_weights(self, request: dict[str, Any], *, for_sampler: bool) -> dict[str, Any]: ...

    def load_weights(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def unload_model(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def create_sampling_session(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def sample(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def runtime_status(self, request: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def prepare_shutdown(self) -> dict[str, Any]: ...


class FutureStore:
    """Turn blocking model calls into Tinker's polling future protocol."""

    def __init__(self, workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tinker-beam")
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        operation: Callable[..., dict[str, Any]],
        *args: Any,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        future = self._executor.submit(operation, *args, **kwargs)
        with self._lock:
            self._futures[request_id] = future
        response: dict[str, Any] = {"request_id": request_id}
        if model_id is not None:
            response["model_id"] = model_id
        return response

    def retrieve(self, request_id: str, *, wait_timeout: float = 30) -> dict[str, Any]:
        with self._lock:
            future = self._futures.get(request_id)
        if future is None:
            return {"error": f"unknown request_id: {request_id}", "category": "User"}
        try:
            result = future.result(timeout=wait_timeout)
        except TimeoutError:
            if not future.done():
                return {
                    "type": "try_again",
                    "request_id": request_id,
                    "queue_state": "active",
                }
            try:
                result = future.result()
            except Exception as exc:
                with self._lock:
                    self._futures.pop(request_id, None)
                return {"error": f"{type(exc).__name__}: {exc}", "category": "Server"}
        except Exception as exc:
            with self._lock:
                self._futures.pop(request_id, None)
            return {"error": f"{type(exc).__name__}: {exc}", "category": "Server"}
        with self._lock:
            self._futures.pop(request_id, None)
        return result

    def close(self) -> None:
        """Cancel queued work and release worker threads."""

        with self._lock:
            self._futures.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)


def create_app(engine: ComputeEngine) -> Any:
    """Build the FastAPI application around a compute engine."""

    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - only possible in a malformed Pod image
        raise ImportError("The Beam compute server requires fastapi") from exc

    futures = FutureStore()

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        try:
            yield
        finally:
            futures.close()
            close = getattr(engine, "close", None)
            if close is not None:
                close()

    app = FastAPI(
        title="OpenTinker Beam compute backend",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.get("/api/v1/healthz")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": getattr(engine, "runtime_info", {"strategy": "unknown"}),
        }

    @app.get("/api/v1/get_server_capabilities")
    def capabilities() -> dict[str, list[dict[str, Any]]]:
        return {
            "supported_models": [
                {
                    "model_name": engine.base_model,
                    "max_context_length": engine.max_length,
                }
            ]
        }

    @app.post("/api/v1/client/config")
    def client_config(_request: dict[str, Any]) -> dict[str, Any]:
        return {
            "parallel_fwdbwd_chunks": False,
            "proto_write_fwdbwd": False,
            "sample_enable_stuck_detection": False,
            "use_pyqwest_transport": False,
        }

    @app.post("/api/v1/create_session")
    def create_session(_request: dict[str, Any]) -> dict[str, str]:
        return {"type": "create_session", "session_id": str(uuid.uuid4())}

    @app.post("/api/v1/session_heartbeat")
    def session_heartbeat(_request: dict[str, Any]) -> dict[str, str]:
        return {"type": "session_heartbeat"}

    @app.post("/api/v1/telemetry")
    def telemetry(_request: dict[str, Any]) -> dict[str, str]:
        return {"status": "accepted"}

    @app.post("/opentinker/prepare-shutdown")
    def prepare_shutdown() -> dict[str, Any]:
        return engine.prepare_shutdown()

    @app.get("/opentinker/runtime")
    def runtime_status() -> dict[str, Any]:
        return engine.runtime_status()

    @app.post("/api/v1/create_sampling_session")
    def create_sampling_session(request: dict[str, Any]) -> dict[str, Any]:
        return engine.create_sampling_session(request)

    @app.post("/api/v1/create_model")
    def create_model(request: dict[str, Any]) -> dict[str, Any]:
        model_id = str(uuid.uuid4())
        request = {**request, "_model_id": model_id}
        return futures.submit(engine.create_model, request, model_id=model_id)

    @app.post("/api/v1/get_info")
    def get_info(request: dict[str, Any]) -> dict[str, Any]:
        return engine.get_info(request)

    @app.post("/api/v1/weights_info")
    def weights_info(request: dict[str, Any]) -> dict[str, Any]:
        return engine.weights_info(request)

    @app.post("/api/v1/forward")
    def forward(request: dict[str, Any]) -> dict[str, Any]:
        return futures.submit(
            engine.forward_backward,
            request,
            backward=False,
            model_id=request.get("model_id"),
        )

    @app.post("/api/v1/forward_backward")
    def forward_backward(request: dict[str, Any]) -> dict[str, Any]:
        return futures.submit(
            engine.forward_backward,
            request,
            backward=True,
            model_id=request.get("model_id"),
        )

    @app.post("/api/v1/optim_step")
    def optim_step(request: dict[str, Any]) -> dict[str, Any]:
        return futures.submit(engine.optim_step, request, model_id=request.get("model_id"))

    @app.post("/api/v1/save_weights")
    def save_weights(request: dict[str, Any]) -> dict[str, Any]:
        return futures.submit(
            engine.save_weights,
            request,
            for_sampler=False,
            model_id=request.get("model_id"),
        )

    @app.post("/api/v1/save_weights_for_sampler")
    def save_weights_for_sampler(request: dict[str, Any]) -> dict[str, Any]:
        return futures.submit(
            engine.save_weights,
            request,
            for_sampler=True,
            model_id=request.get("model_id"),
        )

    @app.post("/api/v1/load_weights")
    def load_weights(request: dict[str, Any]) -> dict[str, Any]:
        return futures.submit(engine.load_weights, request, model_id=request.get("model_id"))

    @app.post("/api/v1/unload_model")
    def unload_model(request: dict[str, Any]) -> dict[str, Any]:
        return futures.submit(engine.unload_model, request, model_id=request.get("model_id"))

    @app.post("/api/v1/asample")
    def sample(request: dict[str, Any]) -> dict[str, Any]:
        return futures.submit(engine.sample, request)

    @app.post("/api/v1/retrieve_future")
    def retrieve_future(request: dict[str, Any]) -> Any:
        result = futures.retrieve(str(request.get("request_id", "")))
        if result.get("type") == "try_again":
            # Upstream Tinker treats 408 as its quiet long-poll continuation
            # path. Returning try_again with HTTP 200 logs a warning per poll.
            return JSONResponse(content=result, status_code=408)
        return result

    return app


__all__ = ["ComputeEngine", "FutureStore", "create_app"]
