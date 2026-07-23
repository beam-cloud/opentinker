# Examples

Looking for the official interactive Tinker notebooks instead of these
runnable scripts? Clone `tinker-cookbook` beside this repository and follow
[Cookbook notebooks on Beam](../docs/cookbook-notebooks.md). That guide shows
which commands run in the terminal, which code belongs in the first notebook
cell, and how the flags below map to `BeamComputeAdapter` arguments.

All examples use the same compute flags:

```text
no flags              -> Beam serverless A10G
--on-demand           -> Beam's interactive hardware picker
--on-demand --gpu X   -> reserve matching marketplace capacity
--pool my-gpus        -> use your attached or pre-reserved hardware
--gpu-count N         -> DDP across N GPUs on one machine
--interconnect nvlink -> require all-pairs NVLink/NVSwitch connectivity
```

Every run prints its Beam dashboard URL and attach command as soon as the Pod
exists. Successful runs exit naturally and are recorded as `COMPLETE`. Ctrl+C
stops the Pod, preserves completed Volume checkpoints, and releases only
capacity the adapter reserved.

## Multi-GPU acceptance run

[`multigpu_finetune.py`](multigpu_finetune.py) trains a 4B model on 128 real
No Robots conversations, evaluates on a disjoint set, writes state and sampler
checkpoints, reloads the state into a new Tinker training client, and requires
the reloaded NLL to match:

```bash
uv run python examples/multigpu_finetune.py \
  --profile default --pool four-l40s \
  --gpu-count 4 --batch-size 16 --cpu 16 --memory 64Gi
```

Use `--on-demand` instead of `--pool` to select the 4x machine in Beam's
picker. Add `--interconnect nvlink` only for a linked machine; the default
records either `nvlink` or `pcie` in the result without rejecting it.

## Real-world teacher-to-student distillation

[`distill_support_router.py`](distill_support_router.py) distills a banking
support-routing capability from Qwen3-14B into Qwen3-0.6B using the real
Banking77 train/test splits:

```bash
uv run python examples/distill_support_router.py \
  --profile default --on-demand --gpu L40S
```

The teacher gets a 16-intent policy and generates structured routes. Exact
verification rejects wrong or malformed answers before they become training
data. The example then compares the untouched student, teacher, and saved
student checkpoint on held-out test tickets, failing unless distillation
materially improves exact accuracy.

The verified `prod3` run scored `0/32` for the untouched student, `31/32` for
the teacher, and `23/32` for the distilled student. A separate A10G pod loaded
only the persisted sampler handle and reproduced `23/32`.

Artifacts:

- `teacher_audit.jsonl`: accepted and rejected teacher attempts
- `verified_teacher_data.jsonl`: the actual student training set
- `results.json`: per-case base, teacher, and distilled predictions
- `tinker://...` state and sampler handles stored on the Beam Volume

Use `--checkpoint tinker://...` in a second run to prove the saved student
works in a fresh Pod. Read
[`docs/distillation.md`](../docs/distillation.md) for the complete data and
evaluation design.

## Supervised fine-tuning

Run the official Tinker Cookbook loop on `HuggingFaceH4/no_robots`:

```bash
uv run python examples/cookbook_sl_loop.py \
  --profile default --steps 20 --batch-size 4 --max-length 1024 \
  --log-path ./runs/no-robots
```

Fine-tune your own OpenAI-style conversation JSONL:

```bash
uv run python examples/finetune_jsonl.py ./train.jsonl \
  --eval-data ./held-out.jsonl --profile default
```

Evaluate a saved state checkpoint against held-out No Robots examples:

```bash
uv run python examples/evaluate_checkpoint.py \
  tinker://<model-id>/weights/<name> \
  --profile default --gpu A10G --examples 8
```

Pass a `tinker://.../sampler_weights/...` handle to the same command to load
the persisted adapter in a fresh Pod and run generation instead.

See [`docs/data-preparation.md`](../docs/data-preparation.md) for schemas,
rendering, loss masks, and preprocessing helpers.

## Other examples

- [`distill_tool_planner.py`](distill_tool_planner.py) is a compact synthetic
  example with an executable JSON-plan verifier.
- [`basic_finetune.py`](basic_finetune.py) is a fast account, image, endpoint,
  GPU, and Tinker-protocol smoke test.

See [Bring your own hardware](../docs/bring-your-own-hardware.md) to connect a
GPU host or self-hosted Beta9 pool, and
[System diagrams](../docs/system-diagrams.md) for the full architecture.
