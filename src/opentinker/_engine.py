# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

"""Transformers/PEFT implementation of OpenTinker's compute protocol."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tarfile
import tempfile
import threading
import time
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
        volume_sync_seconds: float = 60,
        max_length: int = 8192,
        trust_remote_code: bool = False,
        sampling_gpu: bool = True,
    ) -> None:
        self.base_model = _model_name(base_model)
        self.max_length = max_length
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.volume_name = volume_name
        self.volume_sync_seconds = volume_sync_seconds
        self.trust_remote_code = trust_remote_code
        self.training_device = "cuda:0"
        self.sampling_device = "cuda:1" if sampling_gpu else "cuda:0"
        self._training_model: Any = None
        self._sampling_model: Any = None
        self._sampling_model_name: str | None = None
        self._sampling_checkpoint: Path | None = None
        self._tokenizer: Any = None
        self._sampling_tokenizer: Any = None
        self._sampling_tokenizer_name: str | None = None
        self._models: dict[str, dict[str, Any]] = {}
        self._optimizers: dict[str, Any] = {}
        self._sampling_sessions: dict[str, dict[str, Any]] = {}
        self._last_checkpoint_saved_at: float | None = None
        self._saved_volume_paths: set[str] = set()
        self._saved_checkpoints: dict[str, str] = {}
        self._lock = threading.RLock()

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
            output_type = (
                "CrossEntropyLossReturn"
                if loss_name == "cross_entropy"
                else "ImportanceSamplingLossReturn"
            )
            outputs: list[dict[str, Any]] = []
            loss_values: list[Any] = []
            model.train(mode=backward)
            for datum in data:
                grad_context = contextlib.nullcontext() if backward else torch.no_grad()
                with grad_context:
                    tokens = _input_tokens(datum.get("model_input") or {})
                    if len(tokens) > self.max_length:
                        raise ValueError(
                            f"input has {len(tokens)} tokens, exceeding max_length={self.max_length}"
                        )
                    inputs = torch.tensor([tokens], dtype=torch.long, device=self.training_device)
                    loss_inputs = datum.get("loss_fn_inputs") or {}
                    targets = (
                        _tensor(loss_inputs.get("target_tokens"), torch, self.training_device)
                        .long()
                        .flatten()
                    )
                    if targets.numel() != len(tokens):
                        raise ValueError(
                            "target_tokens must align one-to-one with model_input tokens"
                        )
                    weights_value = loss_inputs.get("weights")
                    weights = (
                        _tensor(weights_value, torch, self.training_device).float().flatten()
                        if weights_value is not None
                        else torch.ones_like(targets, dtype=torch.float32)
                    )
                    logits = model(input_ids=inputs, use_cache=False).logits[0]
                    logprobs = (
                        torch.log_softmax(logits.float(), dim=-1)
                        .gather(-1, targets.unsqueeze(-1))
                        .squeeze(-1)
                    )
                    if loss_name == "cross_entropy":
                        per_token_loss = -logprobs
                    else:
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
                        if (
                            old_logprobs.shape != logprobs.shape
                            or advantages.shape != logprobs.shape
                        ):
                            raise ValueError(
                                "logprobs and advantages must match target_tokens length"
                            )
                        per_token_loss = -(torch.exp(logprobs - old_logprobs) * advantages)
                    denominator = weights.sum().clamp_min(1.0)
                    datum_loss = (per_token_loss * weights).sum() / denominator
                    outputs.append({"logprobs": _tensor_json(logprobs)})
                if backward:
                    (datum_loss / len(data)).backward()
                loss_values.append(datum_loss.detach())
            loss = torch.stack(loss_values).mean()
            return {
                "loss_fn_output_type": output_type,
                "loss_fn_outputs": outputs,
                "metrics": {"loss:mean": float(loss.detach().cpu())},
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
            return {"metrics": metrics}

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
                if path.exists():
                    shutil.rmtree(path)
                try:
                    shutil.copytree(temporary, path)
                except BaseException:
                    if path.exists():
                        shutil.rmtree(path)
                    raise
                _flush_tree(path)
            self._last_checkpoint_saved_at = time.monotonic()
            volume_path = f"{self.volume_name}/checkpoints/{model_id}/{checkpoint_type}/{name}"
            uri = f"tinker://{model_id}/{checkpoint_type}/{name}"
            self._saved_volume_paths.add(volume_path)
            self._saved_checkpoints[uri] = volume_path
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
        """Report how long the latest Volume write still needs to propagate."""

        with self._lock:
            if self._last_checkpoint_saved_at is None:
                return {
                    "checkpoint_saved": False,
                    "wait_seconds": 0.0,
                    "volume_paths": [],
                    "checkpoints": [],
                }
            age = time.monotonic() - self._last_checkpoint_saved_at
            return {
                "checkpoint_saved": True,
                "wait_seconds": max(0.0, self.volume_sync_seconds - age),
                "volume_paths": sorted(self._saved_volume_paths),
                "checkpoints": [
                    {"uri": uri, "volume_path": volume_path}
                    for uri, volume_path in sorted(self._saved_checkpoints.items())
                ],
            }

    def export_checkpoint(self, request: dict[str, Any]) -> Path:
        """Archive one live checkpoint for the adapter's Volume upload fallback."""

        with self._lock:
            uri = str(request.get("path", ""))
            if uri not in self._saved_checkpoints:
                raise ValueError("only checkpoints saved by this service can be exported")
            checkpoint = self._uri_path(uri)
            descriptor, archive_name = tempfile.mkstemp(
                prefix="opentinker-checkpoint-",
                suffix=".tar.gz",
            )
            os.close(descriptor)
            archive = Path(archive_name)
            try:
                with tarfile.open(archive, "w:gz") as tar:
                    tar.add(checkpoint, arcname="checkpoint")
            except BaseException:
                archive.unlink(missing_ok=True)
                raise
            return archive

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
        session_id = str(uuid.uuid4())
        self._sampling_sessions[session_id] = {
            "base_model": resolved_base_model,
            "model_path": model_path,
        }
        return {
            "type": "create_sampling_session",
            "sampling_session_id": session_id,
        }

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
            if params.get("seed") is not None:
                torch.manual_seed(int(params["seed"]))
                torch.cuda.manual_seed_all(int(params["seed"]))
            input_ids = torch.tensor([prompt], dtype=torch.long, device=self.sampling_device)
            sequences: list[dict[str, Any]] = []
            for _ in range(int(request.get("num_samples", 1))):
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
            return {
                "type": "sample",
                "sequences": sequences,
                "prompt_cache_hit_tokens": 0,
            }

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


def _flush_tree(path: Path) -> None:
    """Flush checkpoint files before a short-lived Pod releases its Volume mount."""

    for item in path.rglob("*"):
        if not item.is_file():
            continue
        with contextlib.suppress(OSError), item.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
    with contextlib.suppress(OSError):
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    with contextlib.suppress(OSError):
        os.sync()


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
