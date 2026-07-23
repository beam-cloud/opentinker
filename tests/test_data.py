from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from opentinker.data import (
    conversation_record,
    conversation_to_datum,
    conversations_to_datums,
    distillation_records_to_datums,
    prompt_completion_messages,
    read_jsonl,
    validate_messages,
    write_jsonl,
)


def test_reads_and_writes_jsonl_with_line_context(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "nested" / "records.jsonl",
        [{"prompt": "héllo", "value": 1}, {"prompt": "world", "value": 2}],
    )

    assert path == (tmp_path / "nested" / "records.jsonl").resolve()
    assert read_jsonl(path) == [
        {"prompt": "héllo", "value": 1},
        {"prompt": "world", "value": 2},
    ]

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"ok": true}\nnot json\n')
    with pytest.raises(ValueError, match=r"invalid\.jsonl:2: invalid JSON"):
        read_jsonl(invalid)


def test_validates_conversations_and_preserves_optional_fields() -> None:
    messages = validate_messages(
        [
            {"role": "user", "content": "Question", "name": "customer"},
            {"role": "assistant", "content": "Answer", "trainable": True},
        ]
    )

    assert messages[0].get("name") == "customer"
    assert messages[1].get("trainable") is True
    assert conversation_record({"chat": messages}, messages_key="chat") == messages

    with pytest.raises(ValueError, match="assistant message"):
        validate_messages([{"role": "user", "content": "No answer"}])
    with pytest.raises(ValueError, match=r"messages\[0\]\.content"):
        validate_messages([{"role": "assistant", "content": 123}])


def test_prompt_completion_messages_keeps_teacher_contract_out_of_student_prompt() -> None:
    messages = prompt_completion_messages(
        "Classify this request",
        '{"priority":"P1"}',
        instruction="Return JSON only.",
        system="You are concise.",
    )

    assert messages == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Return JSON only.\n\nClassify this request"},
        {"role": "assistant", "content": '{"priority":"P1"}'},
    ]


def test_conversations_to_datums_delegates_rendering_and_loss_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def convert(
        messages: list[dict[str, Any]],
        renderer: Any,
        max_length: int,
        train_on_what: Any,
        *,
        reduction: str,
    ) -> str:
        calls.append(
            {
                "messages": messages,
                "renderer": renderer,
                "max_length": max_length,
                "train_on": train_on_what.value,
                "reduction": reduction,
            }
        )
        return f"datum-{len(calls)}"

    monkeypatch.setattr(
        "tinker_cookbook.supervised.data.conversation_to_datum",
        convert,
    )
    renderer = object()
    datums = conversations_to_datums(
        [
            {
                "messages": [
                    {"role": "user", "content": "One"},
                    {"role": "assistant", "content": "First"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Two"},
                    {"role": "assistant", "content": "Second"},
                ]
            },
        ],
        renderer=renderer,
        max_length=512,
        train_on="last_assistant_message",
    )

    assert datums == ["datum-1", "datum-2"]
    assert calls[0] == {
        "messages": [
            {"role": "user", "content": "One"},
            {"role": "assistant", "content": "First"},
        ],
        "renderer": renderer,
        "max_length": 512,
        "train_on": "last_assistant_message",
        "reduction": "mean",
    }


def test_custom_loss_mask_requires_explicit_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tinker_cookbook.supervised.data.conversation_to_datum",
        lambda *_args, **_kwargs: "datum",
    )
    with pytest.raises(ValueError, match="requires a boolean trainable field"):
        conversation_to_datum(
            [
                {"role": "user", "content": "Question", "trainable": False},
                {"role": "assistant", "content": "Answer"},
            ],
            renderer=object(),
            max_length=512,
            train_on="customized",
        )
    with pytest.raises(ValueError, match="require train_on='customized'"):
        conversation_to_datum(
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer", "trainable": True},
            ],
            renderer=object(),
            max_length=512,
            train_on="last_assistant_message",
        )


def test_distillation_records_require_verification_and_train_on_teacher_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def convert(
        messages: list[dict[str, Any]],
        _renderer: Any,
        _max_length: int,
        train_on_what: Any,
        *,
        reduction: str,
    ) -> str:
        calls.append(
            {
                "messages": messages,
                "train_on": train_on_what.value,
                "reduction": reduction,
            }
        )
        return "datum"

    monkeypatch.setattr(
        "tinker_cookbook.supervised.data.conversation_to_datum",
        convert,
    )
    datums = distillation_records_to_datums(
        [
            {
                "prompt": "Plan this task",
                "teacher_response": '{"calls":[]}',
                "verified": True,
            }
        ],
        renderer=object(),
        max_length=1024,
        instruction="Return JSON only.",
        require_verified=True,
    )

    assert datums == ["datum"]
    assert calls == [
        {
            "messages": [
                {"role": "user", "content": "Return JSON only.\n\nPlan this task"},
                {"role": "assistant", "content": '{"calls":[]}'},
            ],
            "train_on": "last_assistant_message",
            "reduction": "mean",
        }
    ]

    with pytest.raises(ValueError, match="not marked verified"):
        distillation_records_to_datums(
            [{"prompt": "bad", "teacher_response": "bad", "verified": False}],
            renderer=object(),
            max_length=1024,
            require_verified=True,
        )
