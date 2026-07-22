# OpenTinker

OpenTinker is a Beam/Beta9 compute backend for the upstream
[`tinker`](https://pypi.org/project/tinker/) SDK. Your training code keeps using
Tinker's `ServiceClient`, `TrainingClient`, datums, futures, renderers, metrics,
and Cookbook recipes. Model forward/backward, AdamW, LoRA weights, sampling,
and checkpoint I/O execute in a PyTorch container on your Beam GPU.

This is not a launcher that sends the expensive work back to Tinker's managed
service. The adapter starts a Tinker-compatible endpoint in a Beam Pod and
points the ordinary Tinker client at it.

## What you need

- Python 3.11 or newer
- a [Beam](https://beam.cloud) account, or a self-hosted Beta9 cluster
- a configured CLI profile (`beam configure`)
- Hugging Face access for the model and dataset you choose; add an `HF_TOKEN`
  Beam secret for gated models or higher download limits

Clone this repository, then install the package and the official Tinker
Cookbook:

```bash
uv sync --extra beam --extra examples
```

Plain pip also works:

```bash
python -m pip install -e '.[beam,examples]'
```

## Fine-tune a real model on a real dataset

This command runs the upstream
[`tinker_cookbook.recipes.sl_loop`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/sl_loop.py)
on `HuggingFaceH4/no_robots`. The recipe downloads the dataset, renders real
conversations into Tinker datums, performs LoRA forward/backward and AdamW
steps on an A10G, logs per-step NLL/BPB metrics, and saves its final state and
sampler checkpoints to a Beam Volume.

```bash
uv run python examples/cookbook_sl_loop.py \
  --profile default \
  --steps 20 \
  --batch-size 4 \
  --max-length 1024 \
  --log-path ./runs/no-robots
```

With no compute flags, OpenTinker requests an A10G from Beam's serverless pool.

## Choose any on-demand GPU interactively

Add `--on-demand` without a GPU to open Beam's native hardware picker inside
the training command:

```bash
uv run python examples/cookbook_sl_loop.py \
  --profile default --on-demand --steps 20
```

The picker shows every currently available offer with its GPU count, region,
hourly price, and provider. After you confirm an offer, OpenTinker discovers
the selected GPU, binds the training Pod to the new pool, and releases the
machine when the adapter exits. The image is built before the picker opens, so
paid hardware is not held during image construction.

Filter the picker when you already know the GPU family you want:

```bash
uv run python examples/cookbook_sl_loop.py \
  --profile default --on-demand --gpu H100 --steps 20
```

Reservations have a one-hour safety TTL by default; change it with
`--machine-ttl 6h`. In a headless process the Beam CLI selects the cheapest
matching offer without prompting, so pass `--gpu` for deterministic automation.
You can also use an existing private pool with `--pool your-existing-pool`
without `--on-demand`.

## Use the Cookbook recipe in your own code

The wrapper is deliberately small. The training loop below is the official
Cookbook implementation, not a parallel OpenTinker trainer:

```python
from tinker_cookbook.recipes.sl_loop import Config
from tinker_cookbook.recipes.sl_loop import main as supervised_fine_tune

from opentinker import BeamComputeAdapter

with BeamComputeAdapter(
    base_model="Qwen/Qwen3-4B-Instruct-2507",
    gpu="A10G",
    profile="default",
    volume_name="tinker-checkpoints",
    max_length=1024,
):
    supervised_fine_tune(
        Config(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            batch_size=4,
            max_length=1024,
            max_steps=20,
            log_path="./runs/no-robots",
        )
    )
```

The recipe creates its own `tinker.ServiceClient()`. While the adapter context
is active, that client is transparently routed to the Beam backend. No recipe
fork is required.

The Python API has the same interactive path. Leaving `gpu` unset while using
`on_demand=True` opens the unfiltered Beam picker:

```python
with BeamComputeAdapter(
    base_model="Qwen/Qwen3-4B-Instruct-2507",
    profile="default",
    on_demand=True,
) as service_client:
    ...
```

## Checkpoints are Beam Volume paths

OpenTinker creates or reuses the named `tinker-checkpoints` Volume. Tinker's
`save_state()` and `save_weights_for_sampler()` return directly usable paths:

```text
beam://tinker-checkpoints/checkpoints/<model-id>/weights/final
beam://tinker-checkpoints/checkpoints/<model-id>/sampler_weights/final
```

Inspect or download them with the Beam CLI:

```bash
beam ls tinker-checkpoints/checkpoints
beam cp \
  beam://tinker-checkpoints/checkpoints/<model-id>/weights/final \
  ./checkpoints/final
```

Each checkpoint directory contains PEFT adapter weights, adapter config,
`opentinker.json` metadata, and (for state checkpoints) `optimizer.pt`. Rerun
the Cookbook command with the same `--log-path` to use its normal
`checkpoints.jsonl` resume flow, including optimizer state.

You can also evaluate a saved adapter against held-out No Robots conversations:

```bash
uv run python examples/evaluate_checkpoint.py \
  beam://tinker-checkpoints/checkpoints/<model-id>/weights/final \
  --profile default --gpu A10G --examples 8
```

This evaluates a fresh base model, loads the checkpoint through Tinker's normal
`TrainingClient.load_state()`, evaluates the same held-out batch again, and
fails unless mean NLL improves.

To exercise Tinker's complete state-and-optimizer resume path explicitly:

```bash
uv run python examples/evaluate_checkpoint.py \
  beam://tinker-checkpoints/checkpoints/<model-id>/weights/final \
  --profile default --gpu A10G --resume-with-optimizer
```

## Ordinary Tinker code also works

```python
import opentinker as tinker

adapter = tinker.BeamComputeAdapter(
    base_model="Qwen/Qwen3-0.6B",
    gpu="A10G",
    profile="default",
)

with adapter as service_client:
    training_client = service_client.create_lora_training_client(
        base_model="Qwen/Qwen3-0.6B",
        rank=8,
    )
    # Tinker Datum -> forward_backward -> optim_step -> save_state
```

`import opentinker as tinker` delegates Tinker's public API, so the rest of a
normal Tinker program does not need a second set of types or helpers.

## Compatibility

The current single-node backend supports:

- upstream Tinker `ServiceClient`, `TrainingClient`, and `SamplingClient`
- PEFT LoRA causal-language-model training
- token-input cross-entropy and importance-sampling losses
- `forward`, `forward_backward`, and AdamW `optim_step`
- state and sampler checkpoints in a named Beam Volume
- checkpoint weight loading and optimizer-state resume
- the Cookbook supervised `sl_loop`, including its No Robots dataset,
  renderers, metrics, final checkpoints, and local resume record

It does not yet implement distributed/multi-node training, multimodal inputs,
arbitrary custom loss functions, or the full Tinker account-management REST
API. The adapter's temporary `tinker.ServiceClient` override is process-global;
do not run two adapter contexts concurrently in one Python process.

## Why there is an endpoint

The upstream Tinker SDK is an HTTP client. Preserving its unmodified clients
and futures requires a compatibility endpoint between the local workflow and
the remote GPU. OpenTinker runs that endpoint inside the Beam Pod. Future
retrieval is long-polled so model downloads and GPU steps do not generate a
tight `try_again` request loop.

See [`examples/`](examples/) for the real Cookbook run and a tiny smoke test.
