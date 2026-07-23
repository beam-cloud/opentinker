# Examples

For an architectural overview and annotated fine-tuning/distillation flows,
start with [`docs/system-diagrams.md`](../docs/system-diagrams.md).

Every example has the same hardware contract:

```text
no flags              -> Beam serverless A10G
--on-demand           -> reserve, use, and release marketplace hardware
--pool my-gpus        -> use your attached or pre-reserved hardware
```

`--pool` auto-detects the connected GPU type. See
[Bring your own hardware](../docs/bring-your-own-hardware.md) to attach a GPU
host with `beam pool join`, print an installer for a remote host, or use a
self-hosted Beta9 pool.

Every run prints a clickable Beam/Beta9 dashboard URL immediately after Pod
creation, before the model is ready. It also prints the container ID and the
exact `container attach` command for terminal monitoring. Ctrl+C cleanly stops
the local Tinker loop, flushes completed checkpoints, terminates the Pod, and
releases adapter-owned on-demand hardware; user-owned `--pool` capacity stays
attached.

## Fine-tune your own JSONL

`finetune_jsonl.py` is the shortest practical path from user data to a trained
adapter. Each line contains a `messages` array using familiar `system`, `user`,
and `assistant` roles:

```bash
uv run python examples/finetune_jsonl.py examples/data/sft.jsonl \
  --profile default --on-demand --gpu A16
```

Replace the sample path with your own JSONL. The command validates records,
uses the model renderer to build Tinker datums and assistant-only loss masks,
runs LoRA forward/backward plus AdamW, verifies that NLL improved, and returns
both resumable state and sampler handles.

Add `--eval-data ./held-out.jsonl` to compute the before/after NLL on a separate
file with the same message schema. Without it, the small example evaluates up
to eight training records as a quick wiring check.

For a nonstandard input schema, load or map records in Python and call
`opentinker.data.conversations_to_datums(...)`. See
[`docs/data-preparation.md`](../docs/data-preparation.md) for SFT, multi-turn,
and distillation examples.

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
checkpoints as `tinker://...` handles backed by
`beam://tinker-checkpoints/checkpoints/...`. It uses A10G serverless by default.

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
  tinker://<model-id>/weights/final \
  --profile default --gpu A10G --examples 8
```

The command measures the same fresh base model before loading the adapter,
loads the checkpoint through the ordinary Tinker `TrainingClient`, and fails
unless held-out mean NLL improves.

## Teacher-to-student distillation

`distill_tool_planner.py` distills a strict six-call analytics planning skill
from `Qwen/Qwen3-4B-Instruct-2507` into `Qwen/Qwen3-0.6B`:

```bash
uv run python examples/distill_tool_planner.py \
  --profile default --on-demand --gpu A16
```

The 4B teacher generates 18 demonstrations. An exact verifier rejects invalid
JSON, wrong arguments, missing calls, and broken dependencies before the
output becomes a Tinker training datum. The example then trains the student,
saves state and sampler checkpoints, reloads the sampler checkpoint through
Tinker's ordinary `SamplingClient`, and evaluates six unseen requests. It
writes auditable `verified_teacher_data.jsonl` and `results.json` artifacts to
`runs/tool-planner-distillation/`.

In the verified prod3 A16 run, the base student scored 0/6, the teacher 6/6,
and the distilled student 6/6 exact after checkpoint reload. A separate A16
reservation then loaded the persisted handle and independently scored 6/6.

Use `--checkpoint tinker://<model-id>/sampler_weights/<name>` with the same
command to skip training and verify a saved student on a newly reserved GPU.

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
