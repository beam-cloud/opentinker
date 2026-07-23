# Run Tinker Cookbook notebooks on Beam

The official Tinker tutorials are marimo notebooks stored as Python files in a
separate repository. OpenTinker supplies their model compute, but it does not
copy those notebook files into the OpenTinker checkout.

This guide uses the following layout:

```text
~/opentinker-workspace/
├── opentinker/
├── tinker-cookbook/
└── notebooks/
```

The first two directories are clean Git checkouts. `notebooks/` contains your
editable copies of individual tutorials.

The commands below were tested against Tinker Cookbook commit
`3e04119ce293a2b6ba5284e35267c9ba6d27c5da`. Pinning the checkout matters:
the notebook source and the installed `tinker-cookbook` Python package must
agree.

## 1. Check the prerequisites

You need Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
Python 3.11 or newer, a Beam account with credits, and Hugging Face access to
the model you choose. No local GPU, CUDA toolkit, or Docker daemon is needed.

Confirm Git and uv in a terminal:

```bash
git --version
uv --version
```

## 2. Clone and install

Run these commands in a terminal, not in a notebook cell:

```bash
mkdir -p ~/opentinker-workspace
cd ~/opentinker-workspace

git clone https://github.com/beam-cloud/opentinker.git
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
git -C tinker-cookbook checkout 3e04119ce293a2b6ba5284e35267c9ba6d27c5da
mkdir -p notebooks

cd ~/opentinker-workspace/opentinker
uv sync --locked --extra beam --extra notebooks
uv run beam login
uv run beam config list
```

`beam login` creates the default Beam profile. If you already configured Beam,
skip it and confirm the profile you want appears in `beam config list`.
Copy that context name exactly into the notebook's `profile=...` argument.

These are one-time clone commands. If a checkout already exists, do not clone
over it. Update OpenTinker with
`git -C ~/opentinker-workspace/opentinker pull`. Keep the Cookbook checkout at
the tested commit above; a later OpenTinker release may name a newer tested
revision.

For a self-hosted Beta9 cluster, install the Beta9 provider instead:

```bash
cd ~/opentinker-workspace/opentinker
uv sync --locked --extra beta9 --extra notebooks
uv run beta9 config list
```

Its setup cell must also select the Beta9 provider:

```python
from opentinker.notebook import start

adapter = start(
    base_model="Qwen/Qwen3.5-4B",
    provider="beta9",
    profile="my-beta9-profile",
    pool="my-gpu-pool",
)
```

The `notebooks` extra installs the current Tinker Cookbook package and tutorial
dependencies, including marimo. It does not install the repository-level
`tutorials/` files, so the separate clone is still required.

## 3. Open the first training notebook

Tutorial 303 is the best first end-to-end notebook because it runs one
single-model SFT workflow. Copy it before editing, then launch marimo from the
OpenTinker environment:

```bash
cd ~/opentinker-workspace
cp -n tinker-cookbook/tutorials/303_sft_with_config.py \
  notebooks/303_sft_with_config_beam.py

cd ~/opentinker-workspace/opentinker
uv run marimo edit ../notebooks/303_sft_with_config_beam.py
```

This keeps your changes in `notebooks/303_sft_with_config_beam.py`; the
Cookbook checkout remains clean and pinned to the tested revision. `cp -n`
refuses to overwrite an existing edited copy. Choose another destination
filename when you want a second experiment.

## 4. Add one Beam setup cell

In the Marimo editor, add a code cell at the very top, before the tutorial's
Tinker API-key and `ServiceClient` cells. Replace `"default"` with the exact
context shown by `uv run beam config list`:

```python
from opentinker.notebook import start

adapter = start(
    base_model="Qwen/Qwen3.5-4B",
    profile="default",
    gpu="A10G",
)
```

Use the exact model configured later in the notebook. Tutorial 303 currently
uses `Qwen/Qwen3.5-4B`. One OpenTinker backend serves one base model.

`start()` does three notebook-specific things:

1. starts the Beam Pod and prints its dashboard and attach command;
2. routes later, unchanged `tinker.ServiceClient()` calls to that Pod; and
3. satisfies the upstream tutorial's API-key gate without requiring a Tinker
   account key.

Run only this new cell first. Wait until it prints
`OpenTinker backend ready: ...`; a cold first run can spend several minutes
building the image, downloading the model, and starting the GPU. Leave any
Tinker API-key field empty, then run the existing Tutorial 303 cells normally.

Do not paste an OpenTinker example CLI command into a shell cell and expect
later notebook cells to use it: that command runs a separate Python process
and a separate training workflow. Run `beam login`, `beam config list`, and
monitoring commands in a terminal. Put hardware selection in `start(...)`.

The setup cell is safe to rerun. It returns the active adapter instead of
starting a second Pod with the same configuration.

Marimo executes saved cells when a notebook opens. Reopening this edited
notebook therefore starts the setup cell and can allocate paid GPU capacity.
The default idle grace is one hour. If it expires, an exact setup-cell rerun
reports that the old backend is unreachable; call `stop()`, rerun the setup
cell, and explicitly resume from a saved checkpoint. OpenTinker will not
silently replace lost training state. Explicit completion is still the safest
and cheapest path.

## 5. Complete the remote task

A saved bare `finish()` cell would execute when Marimo reopens and could
immediately terminate a new Pod. After the last training, inference, or
evaluation cell, add these two click-gated cells instead. Cookbook notebooks
already define `mo`.

Button cell:

```python
finish_button = mo.ui.run_button(label="Finish OpenTinker task")
finish_button
```

Action cell:

```python
mo.stop(not finish_button.value)

from opentinker.notebook import finish

finish()
```

Click the button and keep Marimo open until the action cell returns `True`.
`finish()` flushes and verifies completed checkpoints on the Beam Volume,
asks the remote process to exit with status zero, waits for the Beam task to
become `COMPLETE`, and releases only hardware that OpenTinker reserved. Staged
Volume writes can take several minutes to upload and verify.

To abandon a run instead of completing it:

Button cell:

```python
stop_button = mo.ui.run_button(label="Cancel OpenTinker task")
stop_button
```

Action cell:

```python
mo.stop(not stop_button.value)

from opentinker.notebook import stop

stop()
```

`stop()` preserves completed Volume checkpoints and terminates the Pod. The
setup cell prints the live dashboard URL and attach command, so you can still
inspect a long-running cell from another terminal.

If the notebook kernel exits without either call, OpenTinker performs a
best-effort `stop()` so the Pod is not orphaned. That is a cancellation path;
use `finish()` when you want the Beam task recorded as `COMPLETE`.

Checkpoint handles such as
`tinker://<model-id>/sampler_weights/<name>` point into the persistent
`tinker-checkpoints` Beam Volume. They are not local files under
`~/opentinker-workspace/notebooks`.

## Translate example CLI flags into a notebook cell

The runnable scripts and notebook API use the same option names. CLI spelling
uses hyphens; Python spelling uses underscores:

| Example command flag | Notebook argument |
| --- | --- |
| `--model Qwen/Qwen3.5-4B` | `base_model="Qwen/Qwen3.5-4B"` |
| `--provider beta9` | `provider="beta9"` |
| `--profile prod3` | `profile="prod3"` |
| `--gpu L40S` | `gpu="L40S"` |
| `--gpu-count 4` | `gpu_count=4` |
| `--interconnect nvlink` | `interconnect="nvlink"` |
| `--on-demand` | `on_demand=True` |
| `--pool my-gpus` | `pool="my-gpus"` |
| `--cpu 16` | `cpu=16` |
| `--memory 64Gi` | `memory="64Gi"` |
| `--machine-ttl 2h` | `machine_ttl="2h"` |
| `--volume-name checkpoints` | `volume_name="checkpoints"` |
| `--max-length 1024` | `max_length=1024`, plus the same preprocessing/config limit in the notebook |

For example, this script command:

```bash
uv run python examples/cookbook_sl_loop.py \
  --profile default --on-demand --gpu L40S \
  --gpu-count 4 --interconnect nvlink
```

becomes this notebook setup cell:

```python
from opentinker.notebook import start

adapter = start(
    base_model="Qwen/Qwen3.5-4B",
    profile="default",
    on_demand=True,
    gpu="L40S",
    gpu_count=4,
    interconnect="nvlink",
)
```

`on_demand=True` without `gpu` opens Beam's interactive machine picker.
`pool="my-gpus"` uses hardware you already reserved or attached and never
releases that user-owned pool.

Flags such as `--steps`, `--batch-size`, `--learning-rate`, `--renderer`, and
data or output paths configure the training workflow. Change the corresponding
Cookbook variables or `train.Config` fields; they are not adapter arguments.

There is no shell command that changes the Python process of an already
running notebook. Run `beam login`, `beam config list`, and monitoring
commands in a terminal. Put compute configuration in `start(...)`, and keep
training configuration in the notebook.

## Tutorial compatibility

OpenTinker implements the Tinker client surface used for causal-language-model
sampling, PEFT LoRA training, AdamW, cross-entropy and importance-sampling
losses, checkpoint resume, and sequence-level distillation. It does not
pretend to support every current Cookbook tutorial.

“Verified” means a notebook from commit `3e04119c` was run on a production
Beam A10G with only the OpenTinker lifecycle cells added. A “verified variant”
also changes the model, checkpoint, or prompt named in the row. “Audited”
means its API path was checked against the backend and local tests, but the
complete notebook was not part of this production acceptance run.

| Tutorial | Verification | Status on OpenTinker |
| --- | --- | --- |
| `101_hello_tinker.py` | Verified checkpoint-reload variant | The notebook's base-model sampling cell was pointed at Tutorial 303's 1.56 GB sampler checkpoint and a translation prompt. A fresh A10G Pod loaded it through `model_path`, resolved `get_tokenizer()`, generated “Bonjour,” finished as `COMPLETE`, and left no replacement container. |
| `102_first_sft.py` | Audited | The Qwen3.5 SFT and post-training sampling sections work. Stop before the later Kimi K2.6 section: one OpenTinker backend cannot serve a second base model. |
| `103_async_patterns.py` | Audited | Sampling works, but concurrent requests are queued by one engine; use multi-completion sampling to spread work across GPUs. |
| `201_rendering.py` | Audited | Local rendering sections work; multimodal model compute is not supported. |
| `202_loss_functions.py` | Audited | Cross-entropy and importance-sampling sections only; PPO, CISPO, and arbitrary custom loss backward passes are unsupported. |
| `203_completers.py` | Audited | Token and message completers work when the adapter model matches. |
| `204_weights.py` | Audited | State and sampler save/load work on the Beam Volume. TTL, listing, publication, archive, and download sections are not supported. |
| `205_evaluations.py` | Audited | Cross-entropy evaluation and checkpoint sampling work. |
| `303_sft_with_config.py` | Verified end to end | Recommended first training notebook. The upstream dataset/config loop trained Qwen3.5-4B, saved state and sampler checkpoints, verified their remote Volume objects, and completed cleanly. |
| `406_prompt_distillation.py` | Audited | Its single-model sequence-level prompt-distillation workflow is supported. |
| `501`–`503` deployment tutorials | Unsupported | Hugging Face merge, archive export, and Hub publication are not implemented by the OpenTinker backend. |

RL, multimodal, multi-model, and custom-loss tutorials should currently be
treated as upstream reference material, not drop-in OpenTinker workflows. See
[Compatibility](../README.md#compatibility) for the backend boundary.

## Open another notebook

The launch command always has the same shape:

```bash
cd ~/opentinker-workspace
cp -n tinker-cookbook/tutorials/101_hello_tinker.py \
  notebooks/101_hello_tinker_beam.py
cd ~/opentinker-workspace/opentinker
uv run marimo edit ../notebooks/101_hello_tinker_beam.py
```

Change only the tutorial path and the `base_model` in the setup cell. Run
`finish()` before moving to another notebook so the first Pod and any
adapter-owned reservation complete cleanly.
