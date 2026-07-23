"""Data preparation for Tinker supervised and distillation workflows.

The helpers accept ordinary dictionaries from JSONL, Hugging Face Datasets,
databases, or in-memory generators. The official Tinker Cookbook renderer
handles tokenization and loss-mask construction.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

TrainOn = Literal[
    "last_assistant_message",
    "last_assistant_turn",
    "all_assistant_messages",
    "all_messages",
    "all_tokens",
    "all_user_and_system_messages",
    "customized",
]
Reduction = Literal["none", "mean"]

TRAIN_ON_CHOICES: tuple[TrainOn, ...] = (
    "last_assistant_message",
    "last_assistant_turn",
    "all_assistant_messages",
    "all_messages",
    "all_tokens",
    "all_user_and_system_messages",
    "customized",
)


class TextMessage(TypedDict):
    """OpenAI-style text message accepted by Cookbook text renderers."""

    role: str
    content: str
    trainable: NotRequired[bool]
    name: NotRequired[str]
    tool_call_id: NotRequired[str]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON objects from a JSONL file with useful filename/line errors."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number}: each JSONL row must be an object")
            records.append(cast(dict[str, Any], record))
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write mappings as newline-delimited JSON and return the resolved path."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    return destination


def validate_messages(messages: object, *, label: str = "messages") -> list[TextMessage]:
    """Validate and copy a text conversation without discarding optional fields."""

    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ValueError(f"{label} must be a sequence of message objects")
    normalized: list[TextMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"{label}[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{label}[{index}].role must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError(f"{label}[{index}].content must be a string")
        if "trainable" in message and not isinstance(message["trainable"], bool):
            raise ValueError(f"{label}[{index}].trainable must be a boolean")
        normalized.append(cast(TextMessage, cast(object, dict(message))))
    if not normalized:
        raise ValueError(f"{label} must contain at least one message")
    if not any(message["role"] == "assistant" for message in normalized):
        raise ValueError(f"{label} must contain at least one assistant message")
    return normalized


def conversation_record(
    record: Mapping[str, Any],
    *,
    messages_key: str = "messages",
    label: str = "record",
) -> list[TextMessage]:
    """Extract one validated conversation from a dataset record."""

    if messages_key not in record:
        raise ValueError(f"{label} has no {messages_key!r} field")
    return validate_messages(record[messages_key], label=f"{label}.{messages_key}")


def prompt_completion_messages(
    prompt: str,
    completion: str,
    *,
    instruction: str | None = None,
    system: str | None = None,
) -> list[TextMessage]:
    """Create a train-on-the-answer conversation for SFT or distillation."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("completion must be a non-empty string")
    user_content = f"{instruction.strip()}\n\n{prompt}" if instruction else prompt
    messages: list[TextMessage] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": completion},
        ]
    )
    return messages


def conversation_to_datum(
    messages: object,
    *,
    renderer: Any,
    max_length: int | None,
    train_on: TrainOn = "all_assistant_messages",
    reduction: Reduction = "mean",
) -> Any:
    """Render one conversation into the ordinary Tinker ``Datum`` format."""

    if max_length is not None and max_length <= 1:
        raise ValueError("max_length must be greater than one or None")
    try:
        from tinker_cookbook import renderers
        from tinker_cookbook.supervised.data import conversation_to_datum as convert
    except ImportError as exc:
        raise ImportError(
            "data preprocessing requires the Cookbook; install `opentinker[examples]`"
        ) from exc
    normalized = validate_messages(messages)
    if train_on == "customized":
        missing = [index for index, message in enumerate(normalized) if "trainable" not in message]
        if missing:
            raise ValueError(
                "train_on='customized' requires a boolean trainable field on every message; "
                f"missing at indexes {missing}"
            )
    elif any("trainable" in message for message in normalized):
        raise ValueError(
            "message trainable fields require train_on='customized'; remove them or change "
            "the loss-mask mode"
        )
    return convert(
        cast(Any, normalized),
        renderer,
        max_length,
        renderers.TrainOnWhat(train_on),
        reduction=reduction,
    )


def conversations_to_datums(
    records: Iterable[Mapping[str, Any]],
    *,
    renderer: Any,
    max_length: int | None,
    messages_key: str = "messages",
    train_on: TrainOn = "all_assistant_messages",
    reduction: Reduction = "mean",
) -> list[Any]:
    """Convert OpenAI-style conversation records into Tinker datums."""

    datums: list[Any] = []
    for index, record in enumerate(records):
        messages = conversation_record(record, messages_key=messages_key, label=f"record[{index}]")
        datums.append(
            conversation_to_datum(
                messages,
                renderer=renderer,
                max_length=max_length,
                train_on=train_on,
                reduction=reduction,
            )
        )
    return datums


def distillation_records_to_datums(
    records: Iterable[Mapping[str, Any]],
    *,
    renderer: Any,
    max_length: int | None,
    prompt_key: str = "prompt",
    response_key: str = "teacher_response",
    instruction: str | None = None,
    system: str | None = None,
    require_verified: bool = False,
    reduction: Reduction = "mean",
) -> list[Any]:
    """Convert verified teacher prompt/response records into student SFT datums."""

    datums: list[Any] = []
    for index, record in enumerate(records):
        if require_verified and record.get("verified") is not True:
            raise ValueError(f"record[{index}] is not marked verified")
        prompt = record.get(prompt_key)
        response = record.get(response_key)
        if not isinstance(prompt, str):
            raise ValueError(f"record[{index}].{prompt_key} must be a string")
        if not isinstance(response, str):
            raise ValueError(f"record[{index}].{response_key} must be a string")
        messages = prompt_completion_messages(
            prompt,
            response,
            instruction=instruction,
            system=system,
        )
        datums.append(
            conversation_to_datum(
                messages,
                renderer=renderer,
                max_length=max_length,
                train_on="last_assistant_message",
                reduction=reduction,
            )
        )
    return datums


__all__ = [
    "TRAIN_ON_CHOICES",
    "Reduction",
    "TextMessage",
    "TrainOn",
    "conversation_record",
    "conversation_to_datum",
    "conversations_to_datums",
    "distillation_records_to_datums",
    "prompt_completion_messages",
    "read_jsonl",
    "validate_messages",
    "write_jsonl",
]
