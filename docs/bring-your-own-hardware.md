# Bring your own hardware

OpenTinker separates the training API from the equipment that executes it.

Run the same Tinker program on Beam's serverless fleet, a temporary marketplace
reservation, or a GPU attached to a private Beam/Beta9 pool:

```text
your Tinker workflow -> BeamComputeAdapter -> named compute pool -> your GPU
```

OpenTinker builds the image, starts the compatible endpoint on the selected
pool, and writes checkpoints to the configured Beam Volume. Only the adapter's
hardware options change between sources.

## The three hardware paths

| Hardware source | Command-line option | Ownership and cleanup |
| --- | --- | --- |
| Beam serverless A10G | no hardware flags | Beam schedules the Pod; OpenTinker terminates it |
| On-demand marketplace machine | `--on-demand` and optionally `--gpu H100` | OpenTinker reserves a temporary pool and releases it on exit |
| Your machine or existing pool | `--pool my-gpus` | You own the pool lifecycle; OpenTinker only starts and terminates its Pod |

For a single-GPU-type pool, `--pool` is enough. OpenTinker discovers the GPU
type from the connected machines, so there is no duplicated `--gpu` setting to
keep synchronized. Pass `--gpu` explicitly only when a pool advertises more
than one GPU type or you want an additional constraint.

## 1. Use serverless capacity

With no hardware flags, examples request an A10G from Beam's serverless pool:

```bash
uv run python examples/finetune_jsonl.py examples/data/sft.jsonl \
  --profile default
```

## 2. Reserve arbitrary on-demand hardware

Let OpenTinker open Beam's native offer picker as part of the training command:

```bash
uv run python examples/finetune_jsonl.py examples/data/sft.jsonl \
  --profile default --on-demand
```

Add `--gpu H100` to filter the picker. OpenTinker builds the image first,
invokes `beam machine reserve`, routes the Pod to the resulting pool, and runs
`beam machine release` during cleanup.

You can also reserve hardware yourself when several jobs should share its
lifetime:

```bash
beam machine reserve --gpu H100 --ttl 6h --name training

uv run python examples/finetune_jsonl.py examples/data/sft.jsonl \
  --profile default --pool training

beam machine release --pool training --yes
```

Because the pool existed before the adapter, OpenTinker does not release it.
That makes job lifetime and machine lifetime separate and predictable.

## 3. Attach your own GPU machine

The shortest path is to run this on the GPU host:

```bash
beam pool join opentinker-gpus \
  --gpu H100 \
  --max-gpus 1 \
  --background
```

`pool join` creates or reuses the private pool, runs the agent preflight, and
installs the machine agent. NVIDIA drivers must already be installed on an
NVIDIA host.

If you administer the pool from a different machine, create it first and print
a short-lived installer to run on the GPU host:

```bash
# On your workstation
beam pool create opentinker-gpus --gpu H100
beam pool join-command opentinker-gpus --ttl 30m

# Copy the printed command to the GPU host and run it there.
```

Confirm that the machine is connected:

```bash
beam pool machines opentinker-gpus
```

Then run any OpenTinker example with one hardware flag:

```bash
uv run python examples/finetune_jsonl.py ./my-data.jsonl \
  --profile default --pool opentinker-gpus
```

Python uses the same `pool` option:

```python
import opentinker as tinker

with tinker.BeamComputeAdapter(
    base_model="Qwen/Qwen3-0.6B",
    profile="default",
    pool="opentinker-gpus",  # GPU type is discovered from the pool
) as service_client:
    training_client = service_client.create_lora_training_client(
        base_model="Qwen/Qwen3-0.6B",
        rank=8,
    )
    # Ordinary Tinker workflow continues here.
```

## Self-hosted Beta9

The pool model is shared by Beam and Beta9. Against a self-hosted Beta9
profile, use the `beta9` CLI to join hardware and select that provider in
OpenTinker:

```bash
uv sync --extra beta9 --extra examples

beta9 pool join opentinker-gpus --gpu H100 --max-gpus 1 --background

uv run python examples/finetune_jsonl.py ./my-data.jsonl \
  --provider beta9 --profile my-cluster --pool opentinker-gpus
```

See the [Beta9 project](https://github.com/beam-cloud/beta9) and its current
[`pool create`, `pool join`, and installer implementation](https://github.com/beam-cloud/beta9/blob/main/sdk/src/beta9/cli/pool.py)
for cluster-side setup and advanced agent controls.

## Resource ownership

- `--on-demand` means OpenTinker owns the reservation and releases it.
- `--pool` means you own the pool; OpenTinker never deletes or releases it.
- No hardware flags mean A10G serverless.
- A private pool must have at least one connected GPU machine before startup.
- A homogeneous pool needs only `--pool`; a heterogeneous pool also needs
  `--gpu`.
- Checkpoints use the same Beam Volume regardless of where the GPU lives.

Tinker supplies the workflow API, Beam or Beta9 schedules the job, and the
selected pool determines who owns the physical GPU.

## Monitoring and interruption

OpenTinker prints the provider-authored Pod management URL before waiting for
the model to load, alongside the container ID, app name, and `container attach`
command. Older Beam gateways that omit the deep link fall back to the live
`https://platform.beam.cloud/containers` page; self-hosted Beta9 prints its
profile-aware CLI lookup command. The link is also available from
`adapter.dashboard_url`; `adapter.container_id` supports custom logging or run
tracking.

Ctrl+C stops the local workflow because that process is what issues Tinker's
forward, backward, optimizer, and sampling requests. During shutdown,
OpenTinker flushes checkpoints that have already been saved, terminates the
Pod, and releases only an adapter-owned `--on-demand` reservation. A machine
attached through `--pool` remains yours. If any cleanup operation fails, the
terminal output repeats the dashboard link and prints exact manual stop and
release commands.
