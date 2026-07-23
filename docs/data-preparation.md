# Data preparation

OpenTinker does not invent a second model-input format. You provide readable
conversation or teacher-output records; the official Tinker Cookbook renderer
turns them into tokenized `tinker.Datum` objects with target tokens and loss
weights.

For the runtime architecture and annotated training flows, see
[System diagrams](system-diagrams.md).

## Supervised fine-tuning records

Store one JSON object per line. The default field is `messages`:

```json
{"messages":[{"role":"system","content":"Classify support requests as compact JSON."},{"role":"user","content":"Checkout is returning 500 errors."},{"role":"assistant","content":"{\"priority\":\"P0\",\"team\":\"payments\"}"}],"source":"support-export"}
```

Top-level metadata such as `source` is allowed and ignored by preprocessing.
Every message needs string `role` and `content` fields. Optional message fields
such as `name`, `tool_call_id`, and `trainable` are preserved for renderers that
use them.

Run a file directly:

```bash
uv run python examples/finetune_jsonl.py ./my-data.jsonl \
  --eval-data ./my-held-out-data.jsonl \
  --profile default --on-demand --gpu A16 \
  --model Qwen/Qwen3-0.6B --renderer qwen3 \
  --epochs 4 --batch-size 2 --max-length 512
```

The command performs the complete workflow:

1. Parse and validate each JSONL row.
2. Use the selected model renderer to apply its chat template and tokenizer.
3. Create shifted target tokens and per-token loss weights.
4. Train a rank-8 LoRA adapter through Tinker's `forward_backward` and
   `optim_step` APIs.
5. Measure NLL before and after training.
6. Save resumable state and sampler checkpoints to the Beam Volume.

Keep training and evaluation files separate. `--eval-data` uses the same JSONL
schema and reports NLL before and after fine-tuning on up to eight held-out
records. When it is omitted, the example reuses training rows only as a quick
end-to-end wiring check.

The included [`examples/data/sft.jsonl`](../examples/data/sft.jsonl) teaches a
small support-routing JSON skill and is intended to be copied and replaced.

## Building datums in your own program

Use `read_jsonl` and `conversations_to_datums` when the input already follows
the message schema:

```python
import tinker

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opentinker.data import conversations_to_datums, read_jsonl

model = "Qwen/Qwen3-0.6B"
tokenizer = get_tokenizer(model)
renderer = renderers.get_renderer("qwen3", tokenizer)

records = read_jsonl("my-data.jsonl")
datums = conversations_to_datums(
    records,
    renderer=renderer,
    max_length=2048,
    train_on="last_assistant_message",
)
```

For CSV rows, database objects, or another schema, map them to messages first:

```python
records = [
    {
        "messages": [
            {"role": "system", "content": "Return a normalized category."},
            {"role": "user", "content": row["raw_request"]},
            {"role": "assistant", "content": row["approved_category"]},
        ]
    }
    for row in source_rows
]

datums = conversations_to_datums(
    records,
    renderer=renderer,
    max_length=2048,
    train_on="last_assistant_message",
)
```

The returned values are ordinary Tinker datums and work without an
OpenTinker-specific training loop:

```python
forward_backward = training_client.forward_backward(
    datums[:batch_size],
    loss_fn="cross_entropy",
)
optimizer = training_client.optim_step(
    tinker.AdamParams(learning_rate=1e-3)
)
forward_backward.result()
optimizer.result()
```

## Choosing which tokens receive loss

`train_on` controls the renderer's loss mask:

- `last_assistant_message`: train only the final answer. This is the safest
  default for instruction tuning and distillation.
- `last_assistant_turn`: include the last assistant/tool interaction.
- `all_assistant_messages`: supervise every assistant answer in a multi-turn
  conversation.
- `all_messages`: include user and system message content as targets too.
- `all_tokens`: train on all rendered tokens, including template tokens.
- `all_user_and_system_messages`: supervise only user/system content.
- `customized`: use renderer-supported per-message `trainable` flags.

For `customized`, every message must explicitly include `"trainable": true`
or `"trainable": false`. For all other modes, omit that field. Individual
renderers may support only a subset of these modes and will report unsupported
choices before training starts.

Use `reduction="mean"` (the default) to normalize loss weights per example.
Use `reduction="none"` only when intentionally weighting longer responses more
heavily.

Always inspect truncation behavior. A small `max_length` can remove the final
assistant answer and leave little or no useful loss. Bucket or filter unusually
long examples before training when appropriate.

## Distillation records

Sequence-level distillation has two distinct phases: generate and verify a
teacher answer, then train the student on that answer. Persist the boundary as
JSONL so the exact teacher dataset is auditable:

```json
{"prompt":"Read orders.csv and produce a six-step tool plan.","teacher_response":"{\"calls\":[...]}","verified":true,"teacher":"Qwen/Qwen3-4B-Instruct-2507"}
```

Convert verified rows into student datums:

```python
from opentinker.data import distillation_records_to_datums, read_jsonl

teacher_rows = read_jsonl("verified-teacher-data.jsonl")
student_datums = distillation_records_to_datums(
    teacher_rows,
    renderer=student_renderer,
    max_length=2048,
    instruction="Compile this request into JSON. Return JSON only.",
    require_verified=True,
)
```

The helper constructs this student conversation for every row:

```json
{"messages":[{"role":"user","content":"<instruction>\n\n<prompt>"},{"role":"assistant","content":"<teacher_response>"}]}
```

Only the assistant teacher response receives loss. `require_verified=True`
fails before training if any row is not explicitly marked `verified: true`.

The full [`distill_tool_planner.py`](../examples/distill_tool_planner.py)
example demonstrates the important production details:

- the teacher receives a detailed schema/tool contract;
- the smaller student receives only the short instruction and user request;
- an exact semantic verifier rejects malformed or incorrect teacher outputs;
- verified teacher rows are written with `write_jsonl` before training;
- held-out scenarios never enter the teacher training file;
- the saved student checkpoint is reloaded and evaluated independently.

For probabilistic tasks, replace exact equality with a task-specific verifier,
reward threshold, unit test, schema validator, or human approval step. Do not
blindly train on every teacher sample.

## Helper reference

The public helpers live in `opentinker.data`:

- `read_jsonl` / `write_jsonl`
- `validate_messages` / `conversation_record`
- `conversation_to_datum`
- `conversations_to_datums`
- `prompt_completion_messages`
- `distillation_records_to_datums`

They require the optional Cookbook dependencies installed by
`uv sync --extra beam --extra examples` or
`pip install -e '.[beam,examples]'`.
