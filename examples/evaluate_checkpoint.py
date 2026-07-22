"""Evaluate a Beam Volume checkpoint on held-out No Robots conversations."""

from __future__ import annotations

import argparse
import json
from typing import Any, cast

import datasets
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opentinker import BeamComputeAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="beam:// Volume URI returned by save_state")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--profile")
    parser.add_argument(
        "--gpu",
        help="GPU type (default: A10G serverless; omit with --on-demand to browse all)",
    )
    parser.add_argument("--pool")
    parser.add_argument(
        "--on-demand",
        action="store_true",
        help="open Beam's machine picker and release the reservation after evaluation",
    )
    parser.add_argument("--machine-ttl", default="1h")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--resume-with-optimizer",
        action="store_true",
        help="load through ServiceClient's full state+optimizer resume path",
    )
    return parser.parse_args()


def mean_nll(result: Any, batch: list[Any]) -> float:
    return compute_mean_nll(
        [item["logprobs"] for item in result.loss_fn_outputs],
        [datum.loss_fn_inputs["weights"] for datum in batch],
    )


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
            renderer,
            args.max_length,
            renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
        for row in dataset.select(range(args.examples))
    ]

    adapter = BeamComputeAdapter(
        base_model=args.model,
        profile=args.profile,
        gpu=args.gpu,
        pool=args.pool,
        on_demand=args.on_demand,
        machine_ttl=args.machine_ttl,
        sampling_gpu=False,
        max_length=args.max_length,
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
