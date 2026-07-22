# Examples

## Real supervised fine-tuning

`cookbook_sl_loop.py` runs the official Tinker Cookbook supervised loop on the
real `HuggingFaceH4/no_robots` dataset. This is the practical example and the
production acceptance test for OpenTinker.

```bash
uv run python examples/cookbook_sl_loop.py \
  --profile default \
  --steps 20 \
  --batch-size 4 \
  --max-length 1024 \
  --log-path ./runs/no-robots
```

The recipe logs its normal NLL/BPB metrics and writes final state plus sampler
checkpoints to `beam://tinker-checkpoints/checkpoints/...`. It uses A10G
serverless by default.

Open Beam's native picker to choose from all current on-demand offers:

```bash
uv run python examples/cookbook_sl_loop.py \
  --profile default --on-demand --steps 20
```

Use `--on-demand --gpu H100` to filter the picker. The selected machine is
attached automatically and released when training exits.

Evaluate the returned state checkpoint against held-out No Robots examples:

```bash
uv run python examples/evaluate_checkpoint.py \
  beam://tinker-checkpoints/checkpoints/<model-id>/weights/final \
  --profile default --gpu A10G --examples 8
```

The command measures the same fresh base model before loading the adapter,
loads the checkpoint through the ordinary Tinker `TrainingClient`, and fails
unless held-out mean NLL improves.

## Fast smoke test

`basic_finetune.py` is intentionally tiny. It is useful for checking account,
image, endpoint, GPU, and Tinker protocol setup, but it is not presented as a
meaningful model fine-tune.

```bash
uv run python examples/basic_finetune.py \
  --profile default --gpu A10G --steps 4
```

For an on-demand machine, add `--on-demand` to browse all offers or combine it
with `--gpu` to filter them. The adapter releases reservations and Pods on
context exit.
