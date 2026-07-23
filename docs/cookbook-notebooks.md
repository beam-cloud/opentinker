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
Python 3.11 or newer, and Hugging Face access to the model you choose. Beam
runs require a Beam account with credits; self-hosted runs require a
configured Beta9 profile and GPU cluster. No local GPU, CUDA toolkit, or Docker
daemon is needed.

Public models need no Hugging Face login. For a private or gated model,
authenticate both sides: use `hf auth login` (or a local `HF_TOKEN`) for
Cookbook tokenizer loading, then create an `HF_TOKEN` provider secret and pass
`secrets=("HF_TOKEN",)` to `start(...)` for the remote Pod.

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

These are one-time clone commands. Skip one only when that checkout already
exists at the exact path shown above. Update OpenTinker with
`git -C ~/opentinker-workspace/opentinker pull`. Keep the Cookbook checkout at
the tested commit above; a later OpenTinker release may name a newer tested
revision.

For a self-hosted Beta9 cluster, install the Beta9 provider instead:

```bash
cd ~/opentinker-workspace/opentinker
uv sync --locked --extra beta9 --extra notebooks
uv run beta9 config list
```

When you replace the credential cells in step 4, use the Beta9 provider in
that same click-gated setup-and-training cell:

```python
mo.stop(not run_on_beam.value)

from opentinker.notebook import start

adapter = start(
    base_model=config.model_name,
    provider="beta9",
    profile="my-beta9-profile",
    pool="my-gpu-pool",
)

await train.main(config)
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

## 4. Replace the hosted-Tinker credential cells

You do not need a Tinker account or Tinker API key. In Tutorial 303, go to
**Step 3 — Run training** and delete both upstream cells:

1. the **Paste your Tinker API key** password widget; and
2. the next cell containing `TINKER_API_KEY`, `api_key`, and
   `await train.main(config)`.

Do not add an unrelated setup cell at the top and leave the credential cells
in place. Marimo schedules cells from variable dependencies, not their visual
position, so that arrangement can run the API-key check before Beam is ready.

Replace the two deleted cells with these two cells.

Button:

```python
run_on_beam = mo.ui.run_button(label="Start Beam and run training")
run_on_beam
```

Beam setup and training:

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

Replace `"default"` with the exact context shown by
`uv run beam config list`. Keep `start(...)` and the existing training call in
the same cell and in that order. Tutorial 303's `config.model_name` is
`Qwen/Qwen3.5-4B`; one OpenTinker backend trains one base model. Sampling
clients may use another base model for inference, which is how
teacher-to-student distillation works.

Click **Start Beam and run training**. OpenTinker prints the Pod ID, dashboard
URL, attach command, and then `OpenTinker backend ready: ...`. A cold first run
can spend several minutes building the image, downloading the model, and
starting the GPU. Training begins only after the backend is ready.

The click gate prevents Marimo from allocating a paid GPU or restarting
training merely because the notebook was opened. An exact `start()` rerun
reuses the active Pod. The default idle grace is one hour; if the task has
ended, call `stop()`, start again, and explicitly resume from a saved
checkpoint instead of silently losing training state.

Do not paste an OpenTinker example CLI command into a shell cell and expect
later notebook cells to use it: that command runs a separate Python process
and a separate workflow. Run `beam login`, `beam config list`, and monitoring
commands in a terminal. Put hardware selection in `start(...)`.

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
adapter
mo.stop(not finish_button.value)

from opentinker.notebook import finish

finish()
```

Click the button and keep Marimo open until the action cell returns `True`.
`finish()` flushes and verifies completed checkpoints on the provider Volume,
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

`stop()` preserves completed Volume checkpoints and terminates the Pod. It
becomes effective after `OpenTinker backend ready` is printed. If you need to
abort during cold startup, interrupt the setup-and-training cell; `start()`
cleans up a partially created Pod. The setup cell prints the live dashboard
URL and attach command, so you can still inspect a long-running cell from
another terminal.

If the notebook kernel exits without either call, OpenTinker performs a
best-effort `stop()` so the Pod is not orphaned. Interrupting a later Marimo
cell does not itself finish or stop the notebook backend. Use the cancel
button explicitly, or use `finish()` when you want the Beam task recorded as
`COMPLETE`.

Checkpoint handles such as
`tinker://<model-id>/sampler_weights/<name>` point into the provider's
persistent `tinker-checkpoints` Volume. They are not local files under
`~/opentinker-workspace/notebooks`. A fresh Pod loading one must use the same
`base_model` and `volume_name`; those values are not encoded in a
`tinker://...` handle.

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

becomes these arguments in the same click-gated setup-and-training cell:

```python
mo.stop(not run_on_beam.value)

from opentinker.notebook import start

adapter = start(
    base_model=config.model_name,
    profile="default",
    on_demand=True,
    gpu="L40S",
    gpu_count=4,
    interconnect="nvlink",
)

await train.main(config)
```

On an interactive run, `on_demand=True` opens Beam's machine picker in the
terminal that launched Marimo. Adding `gpu="L40S"` filters the picker to that
GPU type. `pool="my-gpus"` uses hardware you already reserved or attached and
never releases that user-owned pool.

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

“Verified” means the notebook's upstream model, data, config, and training
path from commit `3e04119c` ran on a production Beam A10G. That acceptance run
manually started Beam before training; the deterministic click-gated cell
layout above was subsequently checked with Marimo's strict validator and
Python compilation. A “verified variant” also changes the model, checkpoint,
or prompt named in the row. “Audited” means its API path was checked against
the backend and local tests, but the complete notebook was not part of this
production acceptance run.

| Tutorial | Verification | Status on OpenTinker |
| --- | --- | --- |
| `101_hello_tinker.py` | Verified checkpoint-reload variant | The notebook's base-model sampling cell was pointed at Tutorial 303's 1.56 GB sampler checkpoint and a translation prompt. A fresh A10G Pod loaded it through `model_path`, resolved `get_tokenizer()`, generated “Bonjour,” finished as `COMPLETE`, and left no replacement container. |
| `102_first_sft.py` | Audited | The Qwen3.5 SFT and post-training sampling sections work. Before the later Kimi K2.6 training section, finish the first backend and start another with Kimi as its trainable `base_model`. |
| `103_async_patterns.py` | Audited | Sampling works, but concurrent requests are queued by one engine; use multi-completion sampling to spread work across GPUs. |
| `201_rendering.py` | Audited | Local rendering sections work; multimodal model compute is not supported. |
| `202_loss_functions.py` | Audited | Cross-entropy and importance-sampling sections only; PPO, CISPO, and arbitrary custom loss backward passes are unsupported. |
| `203_completers.py` | Audited | Token and message completers work when the adapter model matches. |
| `204_weights.py` | Audited | State and sampler save/load work on the provider Volume. TTL, listing, publication, archive, and download sections are not supported. |
| `205_evaluations.py` | Audited | Cross-entropy evaluation and checkpoint sampling work. |
| `303_sft_with_config.py` | Verified end to end | Recommended first training notebook. The upstream dataset/config loop trained Qwen3.5-4B, saved state and sampler checkpoints, verified their remote Volume objects, and completed cleanly. |
| `406_prompt_distillation.py` | Audited | Its single-model sequence-level prompt-distillation workflow is supported. |
| `501`–`503` deployment tutorials | Unsupported | Hugging Face merge, archive export, and Hub publication are not implemented by the OpenTinker backend. |

RL, multimodal, multiple-trainable-model, and custom-loss tutorials should
currently be treated as upstream reference material, not drop-in OpenTinker
workflows. See [Compatibility](../README.md#compatibility) for the backend
boundary.

## Open another notebook

The launch command always has the same shape:

```bash
cd ~/opentinker-workspace
cp -n tinker-cookbook/tutorials/101_hello_tinker.py \
  notebooks/101_hello_tinker_beam.py
cd ~/opentinker-workspace/opentinker
uv run marimo edit ../notebooks/101_hello_tinker_beam.py
```

Every upstream notebook must have its hosted-Tinker password widget and API-key
guard removed. For Tutorial 101, replace the password widget with:

```python
run_on_beam = mo.ui.run_button(label="Start Beam and run tutorial")
run_on_beam
```

Replace the entire following cell—the one that imports `os`, checks
`TINKER_API_KEY`, and creates `service_client`—with these two cells.

Setup:

```python
mo.stop(not run_on_beam.value)

from opentinker.notebook import start

adapter = start(
    base_model="Qwen/Qwen3.5-9B-Base",
    profile="default",
    gpu="A10G",
)
```

Service client and the original capabilities check:

```python
adapter
service_client = tinker.ServiceClient()

capabilities = await service_client.get_server_capabilities_async()
print("Available models:")
for model in capabilities.supported_models:
    print(f"  - {model.model_name}")
```

Keep the model passed to `start()` identical to the model requested later in
the notebook. The `adapter` line is required: it makes the Tinker cell depend
on setup in Marimo. Run `finish()` before moving to another notebook so the
first Pod and any adapter-owned reservation complete cleanly.
