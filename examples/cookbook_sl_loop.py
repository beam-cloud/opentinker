"""Run the official Tinker Cookbook supervised loop on OpenTinker's backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinker_cookbook.recipes.sl_loop import Config
from tinker_cookbook.recipes.sl_loop import main as cookbook_main

from opentinker import BeamComputeAdapter
from opentinker._examples import add_compute_arguments, compute_options_from_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-path", default="./runs/no-robots")
    add_compute_arguments(parser)
    args = parser.parse_args()

    log_path = Path(args.log_path).resolve()
    log_path.mkdir(parents=True, exist_ok=True)
    adapter = BeamComputeAdapter(
        base_model=args.model,
        **compute_options_from_args(args),
    )
    with adapter:
        print(
            "Running the upstream Tinker Cookbook supervised loop\n"
            "dataset=HuggingFaceH4/no_robots\n"
            f"model={args.model}\n"
            f"gpu={adapter.gpu}\n"
            f"steps={args.steps}\n"
            f"batch_size={args.batch_size}\n"
            f"checkpoint_volume=beam://{args.volume_name}/checkpoints",
            flush=True,
        )
        cookbook_main(
            Config(
                model_name=args.model,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                lora_rank=8,
                save_every=0,
                max_steps=args.steps,
                log_path=str(log_path),
            )
        )
    print(f"Run metadata: {log_path}", flush=True)
    print(
        f"Inspect checkpoints: beam ls {args.volume_name}/checkpoints",
        flush=True,
    )


if __name__ == "__main__":
    main()
