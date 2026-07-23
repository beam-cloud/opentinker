"""Fine-tune on a real dataset and verify a multi-GPU checkpoint reload."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, cast

import tinker
from datasets import load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opentinker import BeamComputeAdapter
from opentinker._distillation import sample_text
from opentinker._examples import add_compute_arguments, compute_options_from_args, mean_nll
from opentinker.data import conversations_to_datums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--renderer", default="qwen3")
    parser.add_argument("--train-examples", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", default="./runs/multigpu-finetune")
    add_compute_arguments(parser, machine_ttl="2h")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("train_examples", "eval_examples", "batch_size", "steps"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.gpu_count > 1 and args.batch_size < args.gpu_count:
        raise ValueError("--batch-size must be at least --gpu-count")

    requested = args.train_examples + args.eval_examples
    dataset = load_dataset("HuggingFaceH4/no_robots", split="train")
    if len(dataset) < requested:
        raise ValueError(f"no_robots contains fewer than {requested} requested examples")
    rows = [
        cast(dict[str, Any], dataset[index])
        for index in random.Random(0).sample(range(len(dataset)), requested)
    ]
    evaluation_rows = rows[: args.eval_examples]
    training_rows = rows[args.eval_examples :]
    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tokenizer)
    training_datums = conversations_to_datums(
        training_rows,
        renderer=renderer,
        max_length=args.max_length,
        train_on="last_assistant_message",
    )
    evaluation_datums = conversations_to_datums(
        evaluation_rows,
        renderer=renderer,
        max_length=args.max_length,
        train_on="last_assistant_message",
    )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    adapter = BeamComputeAdapter(
        base_model=args.model,
        **compute_options_from_args(args),
    )
    with adapter as service_client:
        training_client = service_client.create_lora_training_client(
            base_model=args.model,
            rank=16,
            seed=0,
            train_unembed=False,
        )
        initial_nll = mean_nll(
            training_client.forward(evaluation_datums, loss_fn="cross_entropy").result(),
            evaluation_datums,
        )
        rng = random.Random(1)
        shuffled = list(training_datums)
        losses: list[float] = []
        for step in range(args.steps):
            offset = (step * args.batch_size) % len(shuffled)
            if offset + args.batch_size > len(shuffled):
                rng.shuffle(shuffled)
                offset = 0
            batch = shuffled[offset : offset + args.batch_size]
            backward = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optimizer = training_client.optim_step(
                tinker.AdamParams(
                    learning_rate=args.learning_rate,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )
            )
            loss = mean_nll(backward.result(), batch)
            optimizer.result()
            losses.append(loss)
            print(
                f"step={step + 1}/{args.steps} batch={len(batch)} mean_nll={loss:.6f}",
                flush=True,
            )

        final_nll = mean_nll(
            training_client.forward(evaluation_datums, loss_fn="cross_entropy").result(),
            evaluation_datums,
        )
        checkpoint_name = f"multigpu-no-robots-{int(time.time())}"
        state_path = training_client.save_state(checkpoint_name).result().path
        sampler_path = training_client.save_weights_for_sampler(checkpoint_name).result().path

        reloaded_client = service_client.create_training_client_from_state_with_optimizer(
            state_path
        )
        reloaded_nll = mean_nll(
            reloaded_client.forward(evaluation_datums, loss_fn="cross_entropy").result(),
            evaluation_datums,
        )
        sampler = service_client.create_sampling_client(model_path=sampler_path)
        inference_text = sample_text(
            sampler,
            renderer,
            tokenizer,
            cast(list[dict[str, str]], evaluation_rows[0]["messages"][:-1]),
            max_tokens=96,
        )
        if not inference_text.strip():
            raise RuntimeError("reloaded sampler checkpoint produced an empty response")
        runtime = adapter.refresh_runtime_info()
        if final_nll >= initial_nll:
            raise RuntimeError("held-out NLL did not improve")
        if not math.isclose(reloaded_nll, final_nll, rel_tol=5e-4, abs_tol=1e-3):
            raise RuntimeError(
                f"reloaded checkpoint changed held-out NLL: {final_nll} -> {reloaded_nll}"
            )

    summary = {
        "dataset": "HuggingFaceH4/no_robots",
        "model": args.model,
        "train_examples": len(training_datums),
        "eval_examples": len(evaluation_datums),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "gpu": adapter.gpu,
        "gpu_count": adapter.gpu_count,
        "runtime": runtime,
        "initial_nll": initial_nll,
        "training_nll": losses,
        "final_nll": final_nll,
        "reloaded_nll": reloaded_nll,
        "state_checkpoint": state_path,
        "sampler_checkpoint": sampler_path,
        "checkpoint_inference": inference_text,
        "checkpoint_verification": list(adapter.checkpoint_verification),
        "container_id": adapter.container_id,
        "dashboard_url": adapter.dashboard_url,
    }
    results_path = output / "results.json"
    results_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
