"""Shared sampling and LoRA training helpers for runnable examples."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, cast

import tinker

from ._examples import mean_nll


@dataclass(frozen=True)
class TrainingArtifacts:
    """Checkpoints and losses produced by one LoRA training run."""

    state_checkpoint: str
    sampler_checkpoint: str
    losses: list[float]


def sample_text(
    client: Any,
    renderer: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
) -> str:
    """Sample one decoded response from a Tinker sampling client."""

    prompt = renderer.build_generation_prompt(cast(Any, messages))
    response = client.sample(
        prompt=prompt,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            stop=renderer.get_stop_sequences(),
        ),
        num_samples=1,
    ).result()
    return tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object in a model response."""

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
    return None


def train_lora(
    service_client: Any,
    *,
    base_model: str,
    datums: list[Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_name: str,
    rank: int = 8,
    seed: int = 0,
    loss_label: str = "teacher_ce_nll",
) -> TrainingArtifacts:
    """Train a shuffled LoRA loop and save state and sampler checkpoints."""

    if not datums:
        raise ValueError("distillation requires at least one training datum")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=rank,
        seed=seed,
    )
    rng = random.Random(seed)
    total_steps = epochs * ((len(datums) + batch_size - 1) // batch_size)
    step = 0
    losses: list[float] = []
    for epoch in range(epochs):
        epoch_datums = list(datums)
        rng.shuffle(epoch_datums)
        for offset in range(0, len(epoch_datums), batch_size):
            batch = epoch_datums[offset : offset + batch_size]
            step += 1
            forward_backward = training_client.forward_backward(
                batch,
                loss_fn="cross_entropy",
            )
            optimizer = training_client.optim_step(
                tinker.AdamParams(
                    learning_rate=learning_rate,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )
            )
            loss = mean_nll(forward_backward.result(), batch)
            optimizer.result()
            losses.append(loss)
            print(
                f"epoch={epoch + 1}/{epochs} step={step}/{total_steps} {loss_label}={loss:.6f}",
                flush=True,
            )

    state_checkpoint = training_client.save_state(checkpoint_name).result().path
    sampler_checkpoint = training_client.save_weights_for_sampler(checkpoint_name).result().path
    return TrainingArtifacts(
        state_checkpoint=state_checkpoint,
        sampler_checkpoint=sampler_checkpoint,
        losses=losses,
    )


__all__ = [
    "TrainingArtifacts",
    "extract_json_object",
    "sample_text",
    "train_lora",
]
