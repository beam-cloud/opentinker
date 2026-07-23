# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

"""Transformers/PEFT implementation of OpenTinker's compute protocol."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any


class TransformersEngine:
    """Single-node LoRA trainer and sampler backed by Transformers/PEFT."""

    def __init__(
        self,
        *,
        base_model: str,
        checkpoint_root: str,
        volume_name: str = "tinker-checkpoints",
        max_length: int = 8192,
        trust_remote_code: bool = False,
        sampling_gpu: bool = True,
        device: str | None = None,
        distributed_rank: int = 0,
        distributed_world_size: int = 1,
    ) -> None:
        self.base_model = _model_name(base_model)
        self.max_length = max_length
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.volume_name = volume_name
        self.trust_remote_code = trust_remote_code
        self.distributed_rank = distributed_rank
        self.distributed_world_size = distributed_world_size
        self.training_device = device or "cuda:0"
        self.sampling_device = device or (
            "cuda:1" if sampling_gpu and distributed_world_size == 1 else "cuda:0"
        )
        self._training_model: Any = None
        self._distributed_model: Any = None
        self._distributed_adapter: str | None = None
        self._sampling_model: Any = None
        self._sampling_model_name: str | None = None
        self._sampling_checkpoint: Path | None = None
        self._tokenizer: Any = None
        self._sampling_tokenizer: Any = None
        self._sampling_tokenizer_name: str | None = None
        self._models: dict[str, dict[str, Any]] = {}
        self._optimizers: dict[str, Any] = {}
        self._sampling_sessions: dict[str, dict[str, Any]] = {}
        self._saved_checkpoints: dict[str, dict[str, Any]] = {}
        self._forward_calls = 0
        self._backward_calls = 0
        self._forward_datums = 0
        self._optimizer_steps = 0
        self._lock = threading.RLock()
        self.runtime_info: dict[str, Any] = {
            "gpu_count": distributed_world_size,
            "distributed": distributed_world_size > 1,
            "strategy": "ddp" if distributed_world_size > 1 else "single",
        }

    def _imports(self) -> tuple[Any, Any, Any, Any]:
        try:
            import peft
            import torch
            import transformers
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - remote image contract
            raise ImportError(
                "The Beam compute image requires torch, peft, and transformers"
            ) from exc
        return torch, peft, transformers, PeftModel

    def _load_training_model(self) -> Any:
        if self._training_model is not None:
            return self._training_model
        torch, _peft, transformers, _peft_model = self._imports()
        if not torch.cuda.is_available():
            raise RuntimeError("Beam compute backend requires a CUDA GPU")
        if self.sampling_device == self.training_device:
            self._clear_sampling_model(torch)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._training_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model,
            dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        ).to(self.training_device)
        self._training_model.config.use_cache = False
        if hasattr(self._training_model, "gradient_checkpointing_enable"):
            self._training_model.gradient_checkpointing_enable()
        return self._training_model

    def configure_runtime(self, topology: dict[str, Any]) -> None:
        """Attach validated CUDA topology details for health reporting."""

        self.runtime_info = {
            **topology,
            "distributed": self.distributed_world_size > 1,
            "strategy": "ddp" if self.distributed_world_size > 1 else "single",
        }

    def _wrap_distributed_model(self, model_id: str) -> None:
        if self.distributed_world_size == 1:
            return
        torch, _peft, _transformers, _peft_model = self._imports()
        self._distributed_model = torch.nn.parallel.DistributedDataParallel(
            self._training_model,
            device_ids=[self.distributed_rank],
            output_device=self.distributed_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        self._distributed_adapter = model_id

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            _torch, _peft, transformers, _peft_model = self._imports()
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=self.trust_remote_code,
            )
        return self._tokenizer

    def _load_sampling_tokenizer(self, base_model: str) -> Any:
        if self._sampling_tokenizer is None or self._sampling_tokenizer_name != base_model:
            _torch, _peft, transformers, _peft_model = self._imports()
            self._sampling_tokenizer = transformers.AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=self.trust_remote_code,
            )
            self._sampling_tokenizer_name = base_model
        return self._sampling_tokenizer

    def _activate(self, model_id: str) -> Any:
        model = self._load_training_model()
        if model_id not in self._models:
            raise ValueError(f"unknown model_id: {model_id}")
        model.set_adapter(model_id)
        if self.distributed_world_size > 1 and self._distributed_adapter != model_id:
            self._wrap_distributed_model(model_id)
        return model

    def create_model(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            requested_model = _model_name(str(request.get("base_model", "")))
            if requested_model != self.base_model:
                raise ValueError(
                    f"this Beam backend serves {self.base_model!r}, not {requested_model!r}"
                )
            model_id = str(request["_model_id"])
            lora = request.get("lora_config") or {}
            rank = int(lora.get("rank", 32))
            seed = lora.get("seed")
            torch, peft, _transformers, _peft_model = self._imports()
            if seed is not None:
                torch.manual_seed(int(seed))
                torch.cuda.manual_seed_all(int(seed))
            model = self._load_training_model()
            target_modules = self._target_modules(model, lora)
            config = peft.LoraConfig(
                r=rank,
                lora_alpha=rank,
                target_modules=target_modules,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if not self._models:
                model = peft.get_peft_model(model, config, adapter_name=model_id)
                self._training_model = model
            else:
                model.add_adapter(model_id, config)
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model.set_adapter(model_id)
            self._wrap_distributed_model(model_id)
            self._models[model_id] = {
                "base_model": self.base_model,
                "rank": rank,
                "target_modules": target_modules,
                "train_mlp": bool(lora.get("train_mlp", True)),
                "train_attn": bool(lora.get("train_attn", True)),
                "train_unembed": bool(lora.get("train_unembed", True)),
            }
            return {"type": "create_model", "model_id": model_id}

    def _target_modules(self, model: Any, config: dict[str, Any]) -> list[str]:
        wanted: set[str] = set()
        if config.get("train_attn", True):
            wanted.update({"q_proj", "k_proj", "v_proj", "o_proj", "query_key_value"})
        if config.get("train_mlp", True):
            wanted.update(
                {
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "fc1",
                    "fc2",
                    "dense_h_to_4h",
                    "dense_4h_to_h",
                }
            )
        if config.get("train_unembed", True):
            wanted.add("lm_head")
        present = {name.rsplit(".", 1)[-1] for name, _module in model.named_modules()}
        targets = sorted(wanted & present)
        if not targets:
            raise ValueError("no requested LoRA target modules exist in this model architecture")
        return targets

    def get_info(self, request: dict[str, Any]) -> dict[str, Any]:
        model_id = str(request.get("model_id", ""))
        state = self._models.get(model_id)
        if state is None:
            raise ValueError(f"unknown model_id: {model_id}")
        return {
            "type": "get_info",
            "model_id": model_id,
            "is_lora": True,
            "lora_rank": state["rank"],
            "model_name": self.base_model,
            "model_data": {
                "model_name": self.base_model,
                "tokenizer_id": self.base_model,
            },
        }

    def weights_info(self, request: dict[str, Any]) -> dict[str, Any]:
        path = self._uri_path(str(request.get("tinker_path", "")))
        manifest_path = path / "opentinker.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"checkpoint metadata does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        return {
            "base_model": manifest["base_model"],
            "is_lora": True,
            "lora_rank": manifest["lora_rank"],
            "train_mlp": manifest.get("train_mlp", True),
            "train_attn": manifest.get("train_attn", True),
            "train_unembed": manifest.get("train_unembed", True),
        }

    def forward_backward(self, request: dict[str, Any], *, backward: bool) -> dict[str, Any]:
        with self._lock:
            torch, _peft, _transformers, _peft_model = self._imports()
            model_id = str(request.get("model_id", ""))
            model = self._activate(model_id)
            payload = request.get("forward_backward_input") or request.get("forward_input")
            if not isinstance(payload, dict):
                raise ValueError("missing forward input")
            loss_name = str(payload.get("loss_fn", "cross_entropy"))
            if loss_name not in {"cross_entropy", "importance_sampling"}:
                raise NotImplementedError(
                    f"BeamComputeAdapter currently supports cross_entropy and importance_sampling, not {loss_name}"
                )
            data = payload.get("data") or []
            if not data:
                raise ValueError("forward pass requires at least one datum")
            self._validate_forward_data(data, loss_name)
            indices = list(range(self.distributed_rank, len(data), self.distributed_world_size))
            local_data = [data[index] for index in indices]
            output_type = (
                "CrossEntropyLossReturn"
                if loss_name == "cross_entropy"
                else "ImportanceSamplingLossReturn"
            )
            model.train(mode=backward)
            batch = self._training_batch(local_data, loss_name, torch)
            grad_context = contextlib.nullcontext() if backward else torch.no_grad()
            with grad_context:
                forward_model = self._distributed_model or model
                logits = forward_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits.float()
                logprobs = (
                    torch.log_softmax(logits, dim=-1)
                    .gather(-1, batch["targets"].unsqueeze(-1))
                    .squeeze(-1)
                )
                if loss_name == "cross_entropy":
                    per_token_loss = -logprobs
                else:
                    per_token_loss = -(
                        torch.exp(logprobs - batch["old_logprobs"]) * batch["advantages"]
                    )
                denominators = batch["weights"].sum(dim=1).clamp_min(1.0)
                datum_losses = (per_token_loss * batch["weights"]).sum(dim=1) / denominators
                local_loss_sum = datum_losses[: len(local_data)].sum()
                outputs = [
                    {"logprobs": _tensor_json(logprobs[row, :length])}
                    for row, length in enumerate(batch["lengths"])
                ]
            metric_loss_sum = local_loss_sum.detach().clone()
            if self.distributed_world_size > 1:
                torch.distributed.all_reduce(metric_loss_sum)
            if backward:
                scaled_loss = local_loss_sum * self.distributed_world_size / max(1, len(data))
                scaled_loss.backward()
            response: dict[str, Any] = {
                "loss_fn_output_type": output_type,
                "loss_fn_outputs": outputs,
                "metrics": {"loss:mean": float((metric_loss_sum / max(1, len(data))).cpu())},
            }
            if self.distributed_world_size > 1:
                response["_distributed_indices"] = indices
            self._forward_calls += 1
            self._forward_datums += len(local_data)
            if backward:
                self._backward_calls += 1
            return response

    def _validate_forward_data(self, data: list[dict[str, Any]], loss_name: str) -> None:
        """Run deterministic request checks on every rank before a DDP collective."""

        for datum in data:
            tokens = _input_tokens(datum.get("model_input") or {})
            if len(tokens) > self.max_length:
                raise ValueError(
                    f"input has {len(tokens)} tokens, exceeding max_length={self.max_length}"
                )
            loss_inputs = datum.get("loss_fn_inputs") or {}
            if _serialized_tensor_size(loss_inputs.get("target_tokens")) != len(tokens):
                raise ValueError("target_tokens must align one-to-one with model_input tokens")
            weights = loss_inputs.get("weights")
            if weights is not None and _serialized_tensor_size(weights) != len(tokens):
                raise ValueError("weights must match target_tokens length")
            if loss_name == "importance_sampling" and (
                _serialized_tensor_size(loss_inputs.get("logprobs")) != len(tokens)
                or _serialized_tensor_size(loss_inputs.get("advantages")) != len(tokens)
            ):
                raise ValueError("logprobs and advantages must match target_tokens length")

    def _training_batch(
        self,
        data: list[dict[str, Any]],
        loss_name: str,
        torch: Any,
    ) -> dict[str, Any]:
        """Pad a rank-local batch while retaining one backward call per rank."""

        rows: list[dict[str, Any]] = []
        for datum in data:
            tokens = _input_tokens(datum.get("model_input") or {})
            if len(tokens) > self.max_length:
                raise ValueError(
                    f"input has {len(tokens)} tokens, exceeding max_length={self.max_length}"
                )
            loss_inputs = datum.get("loss_fn_inputs") or {}
            targets = (
                _tensor(loss_inputs.get("target_tokens"), torch, self.training_device)
                .long()
                .flatten()
            )
            if targets.numel() != len(tokens):
                raise ValueError("target_tokens must align one-to-one with model_input tokens")
            weights_value = loss_inputs.get("weights")
            weights = (
                _tensor(weights_value, torch, self.training_device).float().flatten()
                if weights_value is not None
                else torch.ones_like(targets, dtype=torch.float32)
            )
            if weights.numel() != len(tokens):
                raise ValueError("weights must match target_tokens length")
            row = {
                "tokens": tokens,
                "targets": targets,
                "weights": weights,
            }
            if loss_name == "importance_sampling":
                old_logprobs = (
                    _tensor(loss_inputs.get("logprobs"), torch, self.training_device)
                    .float()
                    .flatten()
                )
                advantages = (
                    _tensor(loss_inputs.get("advantages"), torch, self.training_device)
                    .float()
                    .flatten()
                )
                if old_logprobs.numel() != len(tokens) or advantages.numel() != len(tokens):
                    raise ValueError("logprobs and advantages must match target_tokens length")
                row["old_logprobs"] = old_logprobs
                row["advantages"] = advantages
            rows.append(row)

        # When the global batch is smaller than the world size, idle ranks still
        # need one graph-bearing forward/backward call to satisfy DDP collectives.
        if not rows:
            rows.append(
                {
                    "tokens": [0],
                    "targets": torch.zeros(1, dtype=torch.long, device=self.training_device),
                    "weights": torch.zeros(1, dtype=torch.float32, device=self.training_device),
                    "old_logprobs": torch.zeros(
                        1, dtype=torch.float32, device=self.training_device
                    ),
                    "advantages": torch.zeros(1, dtype=torch.float32, device=self.training_device),
                }
            )
        width = max(len(row["tokens"]) for row in rows)
        batch_size = len(rows)
        input_ids = torch.zeros(
            (batch_size, width),
            dtype=torch.long,
            device=self.training_device,
        )
        attention_mask = torch.zeros_like(input_ids)
        targets = torch.zeros_like(input_ids)
        weights = torch.zeros(
            (batch_size, width),
            dtype=torch.float32,
            device=self.training_device,
        )
        old_logprobs = torch.zeros_like(weights)
        advantages = torch.zeros_like(weights)
        if not data:
            attention_mask[0, 0] = 1
        lengths: list[int] = []
        for index, row in enumerate(rows[: len(data)]):
            length = len(row["tokens"])
            lengths.append(length)
            input_ids[index, :length] = torch.tensor(
                row["tokens"],
                dtype=torch.long,
                device=self.training_device,
            )
            attention_mask[index, :length] = 1
            targets[index, :length] = row["targets"]
            weights[index, :length] = row["weights"]
            if loss_name == "importance_sampling":
                old_logprobs[index, :length] = row["old_logprobs"]
                advantages[index, :length] = row["advantages"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
            "weights": weights,
            "old_logprobs": old_logprobs,
            "advantages": advantages,
            "lengths": lengths,
        }

    def optim_step(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            torch, _peft, _transformers, _peft_model = self._imports()
            model_id = str(request.get("model_id", ""))
            model = self._activate(model_id)
            values = request.get("adam_params") or {}
            learning_rate = float(values.get("learning_rate", 1e-4))
            betas = (
                float(values.get("beta1", 0.9)),
                float(values.get("beta2", 0.95)),
            )
            eps = float(values.get("eps", 1e-12))
            weight_decay = float(values.get("weight_decay", 0.0))
            optimizer = self._optimizers.get(model_id)
            if optimizer is None:
                parameters = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
                optimizer = torch.optim.AdamW(
                    parameters,
                    lr=learning_rate,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                )
                self._optimizers[model_id] = optimizer
            else:
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                    group["betas"] = betas
                    group["eps"] = eps
                    group["weight_decay"] = weight_decay
            clip = float(values.get("grad_clip_norm", 0.0))
            metrics: dict[str, float] = {}
            if clip > 0:
                norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], clip
                )
                metrics["grad_norm"] = float(norm.detach().cpu())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self._optimizer_steps += 1
            return {"metrics": metrics}

    def runtime_status(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return rank-local evidence that CUDA training work actually ran."""

        del request
        torch, _peft, _transformers, _peft_model = self._imports()
        allocated = 0
        peak = 0
        if torch.cuda.is_available():
            allocated = int(torch.cuda.memory_allocated(self.training_device))
            peak = int(torch.cuda.max_memory_allocated(self.training_device))
        return {
            "rank": self.distributed_rank,
            "device": self.training_device,
            "forward_calls": self._forward_calls,
            "backward_calls": self._backward_calls,
            "forward_datums": self._forward_datums,
            "optimizer_steps": self._optimizer_steps,
            "cuda_memory_allocated_bytes": allocated,
            "cuda_peak_memory_allocated_bytes": peak,
        }

    def save_weights(self, request: dict[str, Any], *, for_sampler: bool) -> dict[str, Any]:
        with self._lock:
            torch, _peft, _transformers, _peft_model = self._imports()
            model_id = str(request.get("model_id", ""))
            model = self._activate(model_id)
            checkpoint_type = "sampler_weights" if for_sampler else "weights"
            name = _safe_name(request.get("path") or f"step-{uuid.uuid4().hex[:12]}")
            path = self.checkpoint_root / model_id / checkpoint_type / name
            if path.exists() and not request.get("overwrite", False):
                raise FileExistsError(f"checkpoint already exists: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            # Serialize on the Pod's local disk first. Beam volume directory renames
            # do not preserve nested PEFT adapter directories, and direct safetensor
            # writes can be observed before their header has been fully flushed.
            with tempfile.TemporaryDirectory(prefix="tinker-checkpoint-") as temp_dir:
                temporary = Path(temp_dir)
                model.save_pretrained(
                    temporary,
                    selected_adapters=[model_id],
                    safe_serialization=False,
                )
                optimizer = self._optimizers.get(model_id)
                if optimizer is not None and not for_sampler:
                    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
                model_state = self._models[model_id]
                (temporary / "opentinker.json").write_text(
                    json.dumps(
                        {
                            "format_version": 1,
                            "base_model": self.base_model,
                            "model_id": model_id,
                            "checkpoint_type": checkpoint_type,
                            "name": name,
                            "lora_rank": model_state["rank"],
                            "train_mlp": model_state["train_mlp"],
                            "train_attn": model_state["train_attn"],
                            "train_unembed": model_state["train_unembed"],
                        },
                        indent=2,
                    )
                    + "\n"
                )
                manifest_files = [
                    _file_record(item, relative_to=temporary)
                    for item in sorted(temporary.rglob("*"))
                    if item.is_file()
                ]
                manifest = {"format_version": 1, "files": manifest_files}
                manifest_path = temporary / "opentinker-checksums.json"
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                expected_files = [
                    *manifest_files,
                    _file_record(manifest_path, relative_to=temporary),
                ]
                if path.exists():
                    shutil.rmtree(path)
                try:
                    published = _publish_checkpoint(
                        temporary,
                        path,
                        geesefs_version=_geesefs_version(self.checkpoint_root),
                        expected_files=expected_files,
                    )
                except BaseException:
                    if path.exists():
                        shutil.rmtree(path)
                    raise
            volume_path = f"{self.volume_name}/checkpoints/{model_id}/{checkpoint_type}/{name}"
            uri = f"tinker://{model_id}/{checkpoint_type}/{name}"
            manifest_record = next(
                item for item in published if item["path"] == "opentinker-checksums.json"
            )
            self._saved_checkpoints[uri] = {
                "volume_path": volume_path,
                "manifest_sha256": manifest_record["sha256"],
                "verified": True,
                "verification": (
                    "geesefs-fsync-sha256"
                    if manifest_record["etag"] is not None
                    else "local-fsync-sha256"
                ),
                "geesefs_version": manifest_record["geesefs_version"],
                "file_count": len(published),
                "total_bytes": sum(int(item["size"]) for item in published),
                "files": published,
            }
            # The official Tinker client validates model_path locally and only
            # accepts tinker:// handles. The files still live at the equivalent
            # beam://<volume>/checkpoints/... path for direct Volume access.
            if for_sampler and request.get("sampling_session_seq_id") is not None:
                session_id = str(uuid.uuid4())
                self._sampling_sessions[session_id] = {
                    "base_model": self.base_model,
                    "model_path": uri,
                }
                return {
                    "type": "save_weights_for_sampler",
                    "sampling_session_id": session_id,
                }
            return {
                "type": "save_weights_for_sampler" if for_sampler else "save_weights",
                "path": uri,
            }

    def prepare_shutdown(self) -> dict[str, Any]:
        """Return in-Pod durability proofs for every completed checkpoint."""

        with self._lock:
            checkpoints = [
                {"uri": uri, **checkpoint}
                for uri, checkpoint in sorted(self._saved_checkpoints.items())
            ]
            return {
                "checkpoint_saved": bool(checkpoints),
                "volume_paths": [str(checkpoint["volume_path"]) for checkpoint in checkpoints],
                "checkpoints": checkpoints,
            }

    def close(self) -> None:
        """Release rank-local model memory during an orderly server shutdown."""

        with self._lock:
            self._training_model = None
            self._distributed_model = None
            self._sampling_model = None
            self._optimizers.clear()

    def load_weights(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            torch, peft, _transformers, _peft_model = self._imports()
            model_id = str(request.get("model_id", ""))
            model = self._activate(model_id)
            path = self._uri_path(str(request.get("path", "")))
            adapter_path = _adapter_path(path)
            weights = peft.load_peft_weights(str(adapter_path), device=self.training_device)
            peft.set_peft_model_state_dict(model, weights, adapter_name=model_id)
            if request.get("optimizer"):
                optimizer_path = path / "optimizer.pt"
                if not optimizer_path.exists():
                    raise FileNotFoundError(f"checkpoint has no optimizer state: {request['path']}")
                parameters = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
                optimizer = torch.optim.AdamW(parameters, lr=1e-4)
                optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=self.training_device, weights_only=True)
                )
                self._optimizers[model_id] = optimizer
            return {"type": "load_weights", "path": request["path"]}

    def unload_model(self, request: dict[str, Any]) -> dict[str, Any]:
        model_id = str(request.get("model_id", ""))
        self._models.pop(model_id, None)
        self._optimizers.pop(model_id, None)
        return {"type": "unload_model", "model_id": model_id}

    def create_sampling_session(self, request: dict[str, Any]) -> dict[str, Any]:
        base_model = request.get("base_model")
        model_path = request.get("model_path")
        resolved_base_model = (
            _model_name(str(base_model)) if base_model is not None else self.base_model
        )
        if model_path is not None:
            self._uri_path(str(model_path))
            if resolved_base_model != self.base_model:
                raise ValueError(
                    "Beam Volume checkpoints can only be loaded with their student base model"
                )
        if base_model is None and model_path is None:
            raise ValueError("create_sampling_session requires base_model or model_path")
        session_id = str(request.get("_sampling_session_id") or uuid.uuid4())
        self._sampling_sessions[session_id] = {
            "base_model": resolved_base_model,
            "model_path": model_path,
        }
        return {
            "type": "create_sampling_session",
            "sampling_session_id": session_id,
        }

    def sampling_session(self, session_id: str) -> dict[str, Any]:
        """Return one session for rank-to-rank synchronization."""

        try:
            return dict(self._sampling_sessions[session_id])
        except KeyError as exc:
            raise ValueError(f"unknown sampling_session_id: {session_id}") from exc

    def register_sampling_session(self, request: dict[str, Any]) -> dict[str, Any]:
        """Install a session created by rank zero after checkpoint serialization."""

        session_id = str(request["sampling_session_id"])
        state = request.get("state")
        if not isinstance(state, dict):
            raise ValueError("sampling session state must be a mapping")
        self._sampling_sessions[session_id] = dict(state)
        return {"type": "register_sampling_session", "sampling_session_id": session_id}

    def _clear_sampling_model(self, torch: Any) -> None:
        if self._sampling_model is not None:
            del self._sampling_model
            self._sampling_model = None
            self._sampling_model_name = None
            self._sampling_checkpoint = None
            torch.cuda.empty_cache()

    def _load_sampling_model(self, base_model: str, checkpoint: Path | None) -> Any:
        torch, _peft, transformers, PeftModel = self._imports()
        if (
            self._sampling_model is not None
            and self._sampling_model_name == base_model
            and self._sampling_checkpoint == checkpoint
        ):
            return self._sampling_model
        self._clear_sampling_model(torch)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        ).to(self.sampling_device)
        if checkpoint is not None:
            model = PeftModel.from_pretrained(
                model,
                str(_adapter_path(checkpoint)),
                is_trainable=False,
            )
        model.eval()
        self._sampling_model = model
        self._sampling_model_name = base_model
        self._sampling_checkpoint = checkpoint
        return model

    def sample(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            torch, _peft, _transformers, _peft_model = self._imports()
            total_samples = int(request.get("num_samples", 1))
            if total_samples <= 0:
                raise ValueError("num_samples must be greater than zero")
            sample_indices = list(
                range(self.distributed_rank, total_samples, self.distributed_world_size)
            )
            if not sample_indices:
                return {
                    "type": "sample",
                    "sequences": [],
                    "prompt_cache_hit_tokens": 0,
                    "_distributed_indices": [],
                }
            session_id = str(request.get("sampling_session_id", ""))
            session = self._sampling_sessions.get(session_id)
            if session is None:
                model_path = request.get("model_path")
                base_model = request.get("base_model")
                if model_path is None and base_model is None:
                    raise ValueError(f"unknown sampling_session_id: {session_id}")
                session = {"model_path": model_path, "base_model": base_model}
            checkpoint = (
                self._uri_path(str(session["model_path"])) if session.get("model_path") else None
            )
            sampling_base_model = _model_name(str(session.get("base_model") or self.base_model))
            model = self._load_sampling_model(sampling_base_model, checkpoint)
            tokenizer = self._load_sampling_tokenizer(sampling_base_model)
            prompt = _input_tokens(request.get("prompt") or {})
            params = request.get("sampling_params") or {}
            max_tokens = int(params.get("max_tokens") or 16)
            if len(prompt) + max_tokens > self.max_length:
                max_tokens = self.max_length - len(prompt)
            if max_tokens <= 0:
                raise ValueError("prompt leaves no room for generated tokens")
            temperature = float(params.get("temperature", 1.0))
            do_sample = temperature > 0
            generate_args: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "return_dict_in_generate": True,
                "output_scores": True,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if do_sample:
                generate_args["temperature"] = temperature
                top_p = float(params.get("top_p", 1.0))
                top_k = int(params.get("top_k", -1))
                if top_p < 1:
                    generate_args["top_p"] = top_p
                if top_k > 0:
                    generate_args["top_k"] = top_k
            input_ids = torch.tensor([prompt], dtype=torch.long, device=self.sampling_device)
            sequences: list[dict[str, Any]] = []
            seed_value = request.get("_sampling_seed", params.get("seed"))
            for sample_index in sample_indices:
                if seed_value is not None:
                    seed = int(seed_value) + sample_index
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    generated = model.generate(input_ids=input_ids, **generate_args)
                tokens = generated.sequences[0, len(prompt) :].tolist()
                transition = model.compute_transition_scores(
                    generated.sequences,
                    generated.scores,
                    normalize_logits=True,
                )[0]
                logprobs = transition[: len(tokens)].float().cpu().tolist()
                tokens, logprobs, stopped = _truncate_stop(
                    tokens,
                    logprobs,
                    params.get("stop"),
                    tokenizer,
                )
                if tokenizer.eos_token_id in tokens:
                    eos_index = tokens.index(tokenizer.eos_token_id)
                    tokens = tokens[: eos_index + 1]
                    logprobs = logprobs[: eos_index + 1]
                    stopped = True
                sequences.append(
                    {
                        "stop_reason": "stop" if stopped else "length",
                        "tokens": tokens,
                        "logprobs": logprobs,
                    }
                )
            response: dict[str, Any] = {
                "type": "sample",
                "sequences": sequences,
                "prompt_cache_hit_tokens": 0,
            }
            if self.distributed_world_size > 1:
                response["_distributed_indices"] = sample_indices
            return response

    def _uri_path(self, uri: str) -> Path:
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
        path = self.checkpoint_root.joinpath(*parts)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {uri}")
        return path


def _model_name(value: str) -> str:
    for prefix in ("hf://", "ms://"):
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return value


def _safe_name(value: Any) -> str:
    name = str(value)
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("checkpoint name must be a single non-empty path component")
    return name


def _input_tokens(model_input: dict[str, Any]) -> list[int]:
    tokens: list[int] = []
    for chunk in model_input.get("chunks") or []:
        chunk_tokens = chunk.get("tokens")
        if chunk_tokens is None and isinstance(chunk.get("encoded_text"), dict):
            chunk_tokens = chunk["encoded_text"].get("tokens")
        if chunk_tokens is None:
            raise NotImplementedError("BeamComputeAdapter currently supports token inputs only")
        tokens.extend(int(token) for token in chunk_tokens)
    if not tokens:
        raise ValueError("model_input contains no tokens")
    return tokens


def _tensor(value: Any, torch: Any, device: str) -> Any:
    if not isinstance(value, dict):
        raise ValueError("required tensor loss input is missing")
    dtype = torch.int64 if value.get("dtype") == "int64" else torch.float32
    shape = value.get("shape")
    if value.get("sparse_crow_indices") is not None:
        sparse = torch.sparse_csr_tensor(
            torch.tensor(value["sparse_crow_indices"], dtype=torch.int64),
            torch.tensor(value["sparse_col_indices"], dtype=torch.int64),
            torch.tensor(value.get("data") or [], dtype=dtype),
            size=shape,
        ).to_dense()
        return sparse.to(device)
    tensor = torch.tensor(value.get("data") or [], dtype=dtype, device=device)
    return tensor.reshape(shape) if shape else tensor


def _serialized_tensor_size(value: Any) -> int:
    if not isinstance(value, dict):
        return -1
    shape = value.get("shape")
    if isinstance(shape, list):
        size = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                return -1
            size *= dimension
        return size
    data = value.get("data")
    return len(data) if isinstance(data, list) else -1


def _tensor_json(tensor: Any) -> dict[str, Any]:
    value = tensor.detach().float().cpu()
    return {
        "data": value.flatten().tolist(),
        "dtype": "float32",
        "shape": list(value.shape),
    }


def _adapter_path(path: Path) -> Path:
    if (path / "adapter_config.json").exists():
        return path
    configs = list(path.glob("*/adapter_config.json"))
    if len(configs) != 1:
        raise FileNotFoundError(f"could not locate one PEFT adapter under {path}")
    return configs[0].parent


def _geesefs_version(path: Path) -> str | None:
    """Return the mounted geesefs version, or ``None`` for ordinary local storage."""

    try:
        return os.getxattr(path, "geesefs").decode()
    except OSError:
        return None


def _file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _publish_checkpoint(
    source: Path,
    destination: Path,
    *,
    geesefs_version: str | None,
    expected_files: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy, hash, and synchronously persist a checkpoint on its mounted Volume.

    Beam Volumes use geesefs staged writes. Setting the digest xattr before
    ``fsync`` places it on the same object upload; ``fsync`` then blocks until
    that upload completes. The ETag returned by object storage proves the
    barrier reached the remote object without downloading it again.
    """

    expected_by_path = {str(item["path"]): item for item in expected_files}
    if len(expected_by_path) != len(expected_files):
        raise RuntimeError("checkpoint contains duplicate manifest paths")
    destination.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for source_file in sorted(source.rglob("*")):
        if not source_file.is_file():
            continue
        relative = source_file.relative_to(source)
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with source_file.open("rb") as reader, destination_file.open("wb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            writer.flush()
            checksum = digest.hexdigest()
            if geesefs_version is not None:
                os.setxattr(
                    destination_file,
                    "user.--content-sha256",
                    checksum.encode(),
                )
            os.fsync(writer.fileno())

        etag: str | None = None
        if geesefs_version is not None:
            stored_checksum = os.getxattr(
                destination_file,
                "user.--content-sha256",
            ).decode()
            if stored_checksum != checksum:
                raise RuntimeError(f"geesefs checksum metadata mismatch for {relative.as_posix()}")
            etag_names = [name for name in os.listxattr(destination_file) if name.endswith(".etag")]
            if len(etag_names) != 1:
                raise RuntimeError(
                    f"geesefs did not return one remote ETag for {relative.as_posix()}"
                )
            etag = os.getxattr(destination_file, etag_names[0]).decode()
            if not etag:
                raise RuntimeError(
                    f"geesefs returned an empty remote ETag for {relative.as_posix()}"
                )

        expected = expected_by_path.pop(relative.as_posix(), None)
        if expected is None:
            raise RuntimeError(f"checkpoint manifest omitted {relative.as_posix()}")
        if size != expected["size"] or checksum != expected["sha256"]:
            raise RuntimeError(f"checkpoint copy changed {relative.as_posix()}")
        records.append(
            {
                **expected,
                "etag": etag,
                "geesefs_version": geesefs_version,
            }
        )

    if expected_by_path:
        missing = ", ".join(sorted(expected_by_path))
        raise RuntimeError(f"checkpoint manifest referenced missing files: {missing}")
    directory_fd = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truncate_stop(
    tokens: list[int],
    logprobs: list[float],
    stop: Any,
    tokenizer: Any,
) -> tuple[list[int], list[float], bool]:
    if stop is None:
        return tokens, logprobs, False
    values = list(stop) if isinstance(stop, (list, tuple)) else [stop]
    patterns: list[list[int]] = []
    if values and all(isinstance(value, int) for value in values):
        patterns = [[int(value)] for value in values]
    else:
        patterns = [tokenizer.encode(str(value), add_special_tokens=False) for value in values]
    first: int | None = None
    for pattern in patterns:
        if not pattern:
            continue
        for index in range(len(tokens) - len(pattern) + 1):
            if tokens[index : index + len(pattern)] == pattern:
                first = index + len(pattern) if first is None else min(first, index + len(pattern))
                break
    if first is None:
        return tokens, logprobs, False
    return tokens[:first], logprobs[:first], True


__all__ = ["TransformersEngine"]
