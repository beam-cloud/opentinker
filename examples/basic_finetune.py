"""A small, real Tinker SFT loop whose GPU work runs on Beam.

This follows the same public API and datum construction used by
``tinker_cookbook.recipes.sl_loop``. It deliberately uses a tiny in-memory
dataset so the example verifies training without downloading a full dataset.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, cast

from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

import opentinker as tinker

CONVERSATIONS: list[list[dict[str, str]]] = [
    [
        {"role": "user", "content": "What color is a clear daytime sky?"},
        {"role": "assistant", "content": "A clear daytime sky is blue."},
    ],
    [
        {"role": "user", "content": "Complete this sentence: grass is usually"},
        {"role": "assistant", "content": "Grass is usually green."},
    ],
    [
        {"role": "user", "content": "What is two plus two?"},
        {"role": "assistant", "content": "Two plus two is four."},
    ],
    [
        {"role": "user", "content": "Give the opposite of cold."},
        {"role": "assistant", "content": "The opposite of cold is hot."},
    ],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--renderer", default="qwen3")
    parser.add_argument("--provider", choices=("beam", "beta9"), default="beam")
    parser.add_argument("--profile")
    parser.add_argument("--gpu", default="A10G")
    parser.add_argument("--pool")
    parser.add_argument("--on-demand", action="store_true")
    parser.add_argument("--machine-ttl", default="1h")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def mean_nll(result: Any, batch: list[Any]) -> float:
    logprobs = [item["logprobs"] for item in result.loss_fn_outputs]
    weights = [datum.loss_fn_inputs["weights"] for datum in batch]
    return compute_mean_nll(logprobs, weights)


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")

    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tokenizer)
    batch = [
        conversation_to_datum(
            cast(Any, conversation),
            renderer,
            args.max_length,
            renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        for conversation in CONVERSATIONS
    ]

    checkpoint_name = f"basic-finetune-{int(time.time())}"
    adapter = tinker.BeamComputeAdapter(
        base_model=args.model,
        provider=args.provider,
        profile=args.profile,
        gpu=args.gpu,
        pool=args.pool,
        on_demand=args.on_demand,
        machine_ttl=args.machine_ttl,
        max_length=args.max_length,
        sampling_gpu=False,
    )
    with adapter as service_client:
        print(
            f"Beam backend ready: {adapter.endpoint_url} "
            f"(gpu={args.gpu}, volume={adapter.volume_name})",
            flush=True,
        )
        training_client = service_client.create_lora_training_client(
            base_model=args.model,
            rank=8,
            seed=0,
        )
        print(f"Loaded {args.model} with rank-8 LoRA", flush=True)

        initial = mean_nll(
            training_client.forward(batch, loss_fn="cross_entropy").result(),
            batch,
        )
        print(f"initial_eval mean_nll={initial:.6f}", flush=True)
        step_losses: list[float] = []
        for step in range(1, args.steps + 1):
            forward_backward = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optimizer = training_client.optim_step(
                tinker.AdamParams(
                    learning_rate=args.learning_rate,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )
            )
            step_loss = mean_nll(forward_backward.result(), batch)
            step_losses.append(step_loss)
            optimizer.result()
            print(
                f"step={step}/{args.steps} train_mean_nll={step_loss:.6f} "
                f"learning_rate={args.learning_rate:g}",
                flush=True,
            )

        final = mean_nll(
            training_client.forward(batch, loss_fn="cross_entropy").result(),
            batch,
        )
        checkpoint = training_client.save_state(checkpoint_name).result().path
        print(f"final_eval mean_nll={final:.6f}", flush=True)
        print(f"checkpoint={checkpoint}", flush=True)

    summary = {
        "model": args.model,
        "gpu": args.gpu,
        "steps": args.steps,
        "initial_nll": initial,
        "training_nll": step_losses,
        "final_nll": final,
        "improved": final < initial,
        "checkpoint": checkpoint,
    }
    print(json.dumps(summary, indent=2))
    if not summary["improved"]:
        raise RuntimeError("fine-tuning completed but the verification loss did not improve")


if __name__ == "__main__":
    main()
