"""Fine-tune a model from an OpenAI-style conversation JSONL file."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opentinker import BeamComputeAdapter
from opentinker._examples import add_compute_arguments, compute_options_from_args, mean_nll
from opentinker.data import TRAIN_ON_CHOICES, conversations_to_datums, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default="examples/data/sft.jsonl")
    parser.add_argument("--eval-data", help="optional held-out JSONL with the same schema")
    parser.add_argument("--messages-key", default="messages")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--renderer", default="qwen3")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-on", choices=TRAIN_ON_CHOICES, default="last_assistant_message")
    parser.add_argument("--output", default="./runs/jsonl-finetune")
    add_compute_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")

    records = read_jsonl(args.data)
    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tokenizer)
    datums = conversations_to_datums(
        records,
        renderer=renderer,
        max_length=args.max_length,
        messages_key=args.messages_key,
        train_on=args.train_on,
    )
    if not datums:
        raise ValueError("the dataset contains no examples")
    evaluation_datums = datums
    if args.eval_data:
        evaluation_datums = conversations_to_datums(
            read_jsonl(args.eval_data),
            renderer=renderer,
            max_length=args.max_length,
            messages_key=args.messages_key,
            train_on=args.train_on,
        )
        if not evaluation_datums:
            raise ValueError("the evaluation dataset contains no examples")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    adapter = BeamComputeAdapter(
        base_model=args.model,
        **compute_options_from_args(args),
    )
    with adapter as service_client:
        print(
            f"Fine-tuning {args.model} on {len(datums)} JSONL conversations "
            f"(gpu={adapter.gpu}, train_on={args.train_on})",
            flush=True,
        )
        training_client = service_client.create_lora_training_client(
            base_model=args.model,
            rank=8,
            seed=0,
        )
        evaluation_batch = evaluation_datums[: min(8, len(evaluation_datums))]
        initial_nll = mean_nll(
            training_client.forward(evaluation_batch, loss_fn="cross_entropy").result(),
            evaluation_batch,
        )
        rng = random.Random(0)
        losses: list[float] = []
        total_steps = args.epochs * ((len(datums) + args.batch_size - 1) // args.batch_size)
        step = 0
        for epoch in range(args.epochs):
            shuffled = list(datums)
            rng.shuffle(shuffled)
            for offset in range(0, len(shuffled), args.batch_size):
                batch = shuffled[offset : offset + args.batch_size]
                step += 1
                forward_backward = training_client.forward_backward(
                    batch,
                    loss_fn="cross_entropy",
                )
                optimizer = training_client.optim_step(
                    tinker.AdamParams(
                        learning_rate=args.learning_rate,
                        beta1=0.9,
                        beta2=0.95,
                        eps=1e-8,
                    )
                )
                loss = mean_nll(forward_backward.result(), batch)
                optimizer.result()
                losses.append(loss)
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{total_steps} "
                    f"mean_nll={loss:.6f}",
                    flush=True,
                )

        final_nll = mean_nll(
            training_client.forward(evaluation_batch, loss_fn="cross_entropy").result(),
            evaluation_batch,
        )
        checkpoint_name = f"jsonl-finetune-{int(time.time())}"
        state_path = training_client.save_state(checkpoint_name).result().path
        sampler_path = training_client.save_weights_for_sampler(checkpoint_name).result().path

    summary = {
        "data": str(Path(args.data).resolve()),
        "evaluation_data": str(Path(args.eval_data).resolve()) if args.eval_data else None,
        "examples": len(datums),
        "model": args.model,
        "gpu": adapter.gpu,
        "train_on": args.train_on,
        "initial_nll": initial_nll,
        "final_nll": final_nll,
        "improved": final_nll < initial_nll,
        "training_nll": losses,
        "state_checkpoint": state_path,
        "sampler_checkpoint": sampler_path,
        "container_id": adapter.container_id,
        "dashboard_url": adapter.dashboard_url,
    }
    results_path = output / "results.json"
    results_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["improved"]:
        raise RuntimeError("fine-tuning completed but evaluation NLL did not improve")


if __name__ == "__main__":
    main()
