from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from examples.distill_support_router import (
    POLICIES,
    Ticket,
    assert_improved,
    parse_routing,
    select_tickets,
)
from opentinker._distillation import extract_json_object, train_lora


def test_banking_routes_have_a_strict_deployment_contract() -> None:
    ticket = Ticket(
        text="There is a cash withdrawal I do not recognize.",
        intent="cash_withdrawal_not_recognised",
    )

    assert ticket.expected == {
        "intent": "cash_withdrawal_not_recognised",
        "queue": "fraud_review",
        "priority": "P0",
    }
    assert set(POLICIES) == {
        "card_arrival",
        "card_delivery_estimate",
        "card_not_working",
        "declined_card_payment",
        "pending_card_payment",
        "card_payment_not_recognised",
        "cash_withdrawal_not_recognised",
        "compromised_card",
        "lost_or_stolen_card",
        "lost_or_stolen_phone",
        "cash_withdrawal_charge",
        "card_payment_fee_charged",
        "failed_transfer",
        "pending_transfer",
        "transfer_not_received_by_recipient",
        "unable_to_verify_identity",
    }


def test_select_tickets_is_balanced_and_deterministic() -> None:
    rows = [
        {"text": f"{intent}-{index}", "label_text": intent}
        for intent in POLICIES
        for index in range(4)
    ]

    first = select_tickets(rows, per_intent=2, seed=7)
    second = select_tickets(reversed(rows), per_intent=2, seed=7)

    assert len(first) == len(POLICIES) * 2
    assert {intent: sum(ticket.intent == intent for ticket in first) for intent in POLICIES} == {
        intent: 2 for intent in POLICIES
    }
    assert [ticket.text for ticket in first] != [ticket.text for ticket in second]
    assert [ticket.text for ticket in first] == [
        ticket.text for ticket in select_tickets(rows, per_intent=2, seed=7)
    ]


def test_routing_parser_rejects_extra_prose_and_wrong_keys() -> None:
    valid = '{"intent":"pending_transfer","queue":"bank_transfers","priority":"P2"}'

    assert parse_routing(valid) == {
        "intent": "pending_transfer",
        "queue": "bank_transfers",
        "priority": "P2",
    }
    assert parse_routing(f"Result: {valid}") is None
    assert parse_routing('{"intent":"pending_transfer"}') is None
    assert extract_json_object(f"Result: {valid}") == {
        "intent": "pending_transfer",
        "queue": "bank_transfers",
        "priority": "P2",
    }


def test_ab_gate_requires_a_real_held_out_improvement() -> None:
    assert_improved(
        {"exact_accuracy": 0.1},
        {"exact_accuracy": 0.7},
        minimum_accuracy=0.6,
    )
    with pytest.raises(RuntimeError, match="did not beat"):
        assert_improved(
            {"exact_accuracy": 0.7},
            {"exact_accuracy": 0.7},
            minimum_accuracy=0.6,
        )
    with pytest.raises(RuntimeError, match="required"):
        assert_improved(
            {"exact_accuracy": 0.1},
            {"exact_accuracy": 0.5},
            minimum_accuracy=0.6,
        )


@dataclass
class _Result:
    path: str | None = None


class _Future:
    def __init__(self, result: Any = None) -> None:
        self._result = result

    def result(self) -> Any:
        return self._result


class _TrainingClient:
    def __init__(self) -> None:
        self.batches: list[list[Any]] = []
        self.optim_steps = 0

    def forward_backward(self, batch: list[Any], *, loss_fn: str) -> _Future:
        assert loss_fn == "cross_entropy"
        self.batches.append(batch)
        return _Future(object())

    def optim_step(self, _params: Any) -> _Future:
        self.optim_steps += 1
        return _Future()

    def save_state(self, name: str) -> _Future:
        return _Future(_Result(path=f"tinker://model/weights/{name}"))

    def save_weights_for_sampler(self, name: str) -> _Future:
        return _Future(_Result(path=f"tinker://model/sampler_weights/{name}"))


class _ServiceClient:
    def __init__(self) -> None:
        self.training = _TrainingClient()

    def create_lora_training_client(self, **_kwargs: Any) -> _TrainingClient:
        return self.training


def test_shared_training_loop_saves_both_checkpoint_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _ServiceClient()
    monkeypatch.setattr("opentinker._distillation.mean_nll", lambda _result, batch: len(batch))

    result = train_lora(
        service,
        base_model="student",
        datums=["a", "b", "c"],
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_name="router",
    )

    assert service.training.optim_steps == 4
    assert result.losses == [2, 1, 2, 1]
    assert result.state_checkpoint == "tinker://model/weights/router"
    assert result.sampler_checkpoint == "tinker://model/sampler_weights/router"
