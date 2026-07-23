# OpenTinker

Run existing [Tinker](https://github.com/thinking-machines-lab/tinker) training
code on GPUs managed by [Beam](https://beam.cloud) or a self-hosted Beta9
cluster.

Your `tinker` imports, `ServiceClient` calls, training loop, data, losses,
renderers, futures, and `tinker://...` checkpoint handles stay the same.
OpenTinker adds the compute backend around that code.

You need Beam credentials (or a configured self-hosted Beta9 cluster), not a
Tinker account or `TINKER_API_KEY`. Model forward/backward, optimization, and
sampling run on the GPU you select; default `tinker.ServiceClient()` calls do
not send model requests to Tinker's hosted service.

| Existing Tinker workflow | OpenTinker change |
| --- | --- |
| Imports, clients, recipes, data, losses, and configs | No change |
| Python script | Wrap the Tinker workflow in `BeamComputeAdapter` |
| Marimo notebook | Remove the hosted-Tinker credential cells; add click-gated `start()` and `finish()` cells |

## Existing Tinker project

For the Beam path below, you need Python 3.11 or newer, Git,
[uv](https://docs.astral.sh/uv/getting-started/installation/), and a Beam
account with credits. You do not need a local GPU, CUDA, or Docker.

From your existing project directory, install OpenTinker and log in to Beam:

```bash
uv add "opentinker[beam] @ git+https://github.com/beam-cloud/opentinker.git"
uv run beam login
uv run beam config list
```

If the project does not use uv, the equivalent virtual-environment commands
are:

```bash
python -m pip install "opentinker[beam] @ git+https://github.com/beam-cloud/opentinker.git"
beam login
beam config list
```

Use a context printed by the last command as `profile=...`. Then wrap the part
of your program that uses Tinker:

```python
from opentinker import BeamComputeAdapter

async def main():
    with BeamComputeAdapter(
        base_model=config.model_name,
        profile="default",
        gpu="A10G",
    ):
        await train.main(config)  # Your existing Tinker workflow.
```

That is the complete integration. Code inside the context may continue to
create `tinker.ServiceClient()` itself, as Cookbook recipes do. Start the
context before the first Tinker client and keep it open through the final
Tinker request.

If your code accepts a client explicitly, use the ordinary Tinker
`ServiceClient` returned by the context:

```python
with BeamComputeAdapter(
    base_model="Qwen/Qwen3.5-4B",
    profile="default",
    gpu="A10G",
) as service_client:
    run_existing_tinker_workflow(service_client)
```

Use module-style `tinker.ServiceClient()` inside the context. An alias created
earlier with `from tinker import ServiceClient` cannot be redirected. Leave
`base_url` unset as ordinary Tinker code does; an explicitly supplied
`base_url` intentionally bypasses adapter routing.

Use `BeamComputeAdapter` for ordinary Python programs. Use the notebook
`start()` and `finish()` helpers below only when one workflow spans multiple
notebook cells. Do not mix the two lifecycle styles.

The public Qwen model below needs no Hugging Face credential. For a private or
gated model, authenticate the local process (`hf auth login` or a local
`HF_TOKEN`) because Cookbook code loads tokenizers locally. Also create an
`HF_TOKEN` secret in Beam/Beta9 and pass `secrets=("HF_TOKEN",)` so the remote
Pod can load the model.

For self-hosted Beta9, install
`opentinker[beta9]`, then pass `provider="beta9"` and your Beta9 profile.

## Upstream Tinker Cookbook notebook

This is the complete path for
[`303_sft_with_config.py`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/3e04119ce293a2b6ba5284e35267c9ba6d27c5da/tutorials/303_sft_with_config.py).
It preserves the upstream dataset, config, and training code while replacing
the hosted-Tinker credential and compute lifecycle.

### 1. Clone, copy, and open the tested notebook

Run these commands in a terminal:

```bash
mkdir -p ~/opentinker-workspace
cd ~/opentinker-workspace

git clone https://github.com/beam-cloud/opentinker.git
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
git -C tinker-cookbook checkout 3e04119ce293a2b6ba5284e35267c9ba6d27c5da

mkdir -p notebooks
cp -n tinker-cookbook/tutorials/303_sft_with_config.py \
  notebooks/303_sft_with_config_beam.py

cd opentinker
uv sync --locked --extra beam --extra notebooks
uv run beam login
uv run beam config list
uv run marimo edit ../notebooks/303_sft_with_config_beam.py
```

`cp -n` keeps the upstream checkout clean and will not overwrite an existing
edited copy. Skip a `git clone` line only when that repository already exists
at the exact path shown above. Run every later `uv` or `beam` command from
`~/opentinker-workspace/opentinker`.

### 2. Replace the two Tinker API-key cells

In Marimo, go to **Step 3 — Run training**. Delete both upstream cells:

- the **Paste your Tinker API key** password widget; and
- the next cell containing `TINKER_API_KEY`, `api_key`, and
  `await train.main(config)`.

Do not paste a Tinker key and do not leave those cells in the notebook.
Replace them with these two cells.

Button:

```python
run_on_beam = mo.ui.run_button(label="Start Beam and run training")
run_on_beam
```

Beam setup and the existing training call:

```python
mo.stop(not run_on_beam.value)

from opentinker.notebook import start

adapter = start(
    base_model=config.model_name,
    profile="default",
    gpu="A10G",
)

await train.main(config)
```

Replace `"default"` with the Beam context you selected during installation.
Click **Start Beam and run training**. OpenTinker prints the Pod ID, dashboard
URL, attach command, and then `OpenTinker backend ready`. Training starts only
after that message.

Keep `start(...)` and `await train.main(config)` in the same cell and in that
order. Marimo schedules cells from variable dependencies, not their visual
position. The button is also intentional: reopening the notebook will not
allocate a paid GPU until you click it.

If the notebook still displays **Paste your Tinker API key**, the original
credential cells have not been removed.

### 3. Finish the Beam task

Add these two cells after the training and evaluation cells.

Button:

```python
finish_button = mo.ui.run_button(label="Finish OpenTinker task")
finish_button
```

Action:

```python
adapter
mo.stop(not finish_button.value)

from opentinker.notebook import finish

finish()
```

Click **Finish OpenTinker task** and wait for `True` before closing Marimo.
That flushes and verifies saved checkpoints, records the Beam task as
`COMPLETE`, and releases adapter-owned hardware.

The command boundary is:

| Where | What runs there |
| --- | --- |
| Terminal | `git clone`, `uv sync`, `beam login`, `beam config list`, `marimo edit` |
| Notebook | `start(...)`, the existing Tinker workflow, `finish()` |
| Separate process | `uv run python examples/...` |

Running an example command does not configure an already-open notebook; it
starts a separate workflow. See
[Cookbook notebooks on Beam](docs/cookbook-notebooks.md) for other tutorials,
on-demand hardware, private pools, cancellation, and the tested compatibility
matrix.

## Choose hardware

The same options work in `BeamComputeAdapter(...)` and notebook `start(...)`:

| Arguments | Result |
| --- | --- |
| `gpu="A10G"` | Beam serverless A10G |
| `on_demand=True` | Open Beam's interactive machine picker |
| `on_demand=True, gpu="H100", gpu_count=4` | Reserve a matching 4-GPU machine |
| `pool="my-gpus"` | Use hardware you reserved, attached, or self-host |

Add `interconnect="nvlink"` when every requested GPU must have NVLink or
NVSwitch connectivity. OpenTinker uses single-node PyTorch DDP for
`gpu_count > 1`. See
[Bring your own hardware](docs/bring-your-own-hardware.md) for private pools
and self-hosted Beta9.

With `on_demand=True`, Beam's picker appears in the terminal that launched
Marimo. Supplying `gpu="H100"` filters that picker; it does not move the picker
into the notebook.

## Checkpoints and task lifetime

- Normal completion flushes checkpoints, waits for task status `COMPLETE`, and
  releases only hardware OpenTinker reserved.
- In a Python `BeamComputeAdapter` context, Ctrl+C and exceptions trigger
  cleanup. In a notebook, use the click-gated `finish()` above or the
  click-gated `stop()` pattern in the notebook guide; interrupting an
  unrelated Marimo cell is not a graceful finish.
- State and sampler handles remain ordinary Tinker paths:

```text
tinker://<model-id>/weights/<name>
tinker://<model-id>/sampler_weights/<name>
```

They are stored on the provider's persistent `tinker-checkpoints` Volume and
can be loaded by a fresh Pod configured with the same `base_model` and
`volume_name`. OpenTinker verifies staged Volume writes inside the Pod without
downloading every published object.

## Working examples

From the cloned OpenTinker repository, fine-tune on the real No Robots
dataset:

```bash
uv run python examples/cookbook_sl_loop.py \
  --profile default --steps 20 --batch-size 4 --max-length 1024
```

Distill a 16-intent Banking77 support router from Qwen3-14B into Qwen3-0.6B:

```bash
uv run python examples/distill_support_router.py \
  --profile default --on-demand --gpu L40S
```

The verified run improved the held-out student from `0/32` to `23/32` exact
and reproduced `23/32` after loading its saved checkpoint in a fresh A10G
Pod. See [Examples](examples/) and
[Practical distillation](docs/distillation.md).

## Compatibility

OpenTinker currently supports causal-language-model sampling, PEFT LoRA,
AdamW, cross-entropy and importance-sampling losses, state/optimizer resume,
sampler checkpoints, sequence-level distillation, the Cookbook supervised
loop, and single-node multi-GPU DDP.

Multi-node training, parameter sharding, multimodal inputs, arbitrary custom
loss functions, logit-level distillation, and the full Tinker
account-management API are not implemented. Adapter routing is process-global,
so adapter contexts must not overlap.

More detail:

- [Fine-tuning and distillation: the ML view](docs/ml-training.md)
- [System diagrams](docs/system-diagrams.md)
- [Data preparation](docs/data-preparation.md)
- [Cookbook notebook compatibility](docs/cookbook-notebooks.md#tutorial-compatibility)
