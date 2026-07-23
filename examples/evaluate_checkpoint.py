"""Evaluate a Beam Volume checkpoint on held-out No Robots conversations."""

from __future__ import annotations

import argparse
import json
from typing import Any, cast

import datasets
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opentinker import BeamComputeAdapter
from opentinker._examples import add_compute_arguments, compute_options_from_args, mean_nll
from opentinker.data import conversation_to_datum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="tinker:// handle returned by save_state")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--resume-with-optimizer",
        action="store_true",
        help="load through ServiceClient's full state+optimizer resume path",
    )
    add_compute_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(args.model),
        tokenizer,
    )
    dataset = datasets.load_dataset("HuggingFaceH4/no_robots", split="test")
    batch = [
        conversation_to_datum(
            cast(dict[str, Any], row)["messages"],
            renderer=renderer,
            max_length=args.max_length,
            train_on="all_assistant_messages",
        )
        for row in dataset.select(range(args.examples))
    ]

    adapter = BeamComputeAdapter(
        base_model=args.model,
        **compute_options_from_args(args),
    )
    with adapter as service_client:
        if args.resume_with_optimizer:
            resumed_client = service_client.create_training_client_from_state_with_optimizer(
                args.checkpoint
            )
            resumed_nll = mean_nll(
                resumed_client.forward(batch[:1], loss_fn="cross_entropy").result(),
                batch[:1],
            )
            print(
                json.dumps(
                    {
                        "checkpoint": args.checkpoint,
                        "optimizer_resume": True,
                        "held_out_mean_nll": resumed_nll,
                    },
                    indent=2,
                )
            )
            return
        training_client = service_client.create_lora_training_client(
            base_model=args.model,
            rank=8,
            seed=0,
        )
        baseline = mean_nll(
            training_client.forward(batch, loss_fn="cross_entropy").result(),
            batch,
        )
        training_client.load_state(args.checkpoint).result()
        tuned = mean_nll(
            training_client.forward(batch, loss_fn="cross_entropy").result(),
            batch,
        )

    summary = {
        "dataset": "HuggingFaceH4/no_robots:test",
        "examples": args.examples,
        "checkpoint": args.checkpoint,
        "baseline_mean_nll": baseline,
        "tuned_mean_nll": tuned,
        "improved": tuned < baseline,
    }
    print(json.dumps(summary, indent=2))
    if not summary["improved"]:
        raise RuntimeError("checkpoint did not improve held-out No Robots mean NLL")


if __name__ == "__main__":
    main()
