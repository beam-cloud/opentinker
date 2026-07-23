"""Distill a banking support router from Qwen3-14B into Qwen3-0.6B."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opentinker import BeamComputeAdapter
from opentinker._distillation import sample_text, train_lora
from opentinker._examples import add_compute_arguments, compute_options_from_args
from opentinker.data import distillation_records_to_datums, write_jsonl

DATASET_ID = "mteb/banking77"
DATASET_REVISION = "18072d2685ea682290f7b8924d94c62acc19c0b2"
OUTPUT_KEYS = {"intent", "queue", "priority"}
STUDENT_INSTRUCTION = (
    "Route this banking support request. Return exactly one JSON object with "
    "the keys intent, queue, and priority. Return no other text."
)


@dataclass(frozen=True)
class IntentPolicy:
    description: str
    queue: str
    priority: str


POLICIES: dict[str, IntentPolicy] = {
    "card_arrival": IntentPolicy(
        "A dispatched physical card is late, missing, or needs tracking.",
        "card_delivery",
        "P2",
    ),
    "card_delivery_estimate": IntentPolicy(
        "The customer asks how long normal card delivery should take.",
        "card_delivery",
        "P2",
    ),
    "card_not_working": IntentPolicy(
        "A physical card fails generally, not one declined purchase.",
        "card_support",
        "P1",
    ),
    "declined_card_payment": IntentPolicy(
        "A card purchase was declined.",
        "card_payments",
        "P1",
    ),
    "pending_card_payment": IntentPolicy(
        "A card purchase is pending or has not cleared.",
        "card_payments",
        "P2",
    ),
    "card_payment_not_recognised": IntentPolicy(
        "The customer disputes an unknown card purchase.",
        "fraud_review",
        "P0",
    ),
    "cash_withdrawal_not_recognised": IntentPolicy(
        "The customer disputes an unknown ATM cash withdrawal.",
        "fraud_review",
        "P0",
    ),
    "compromised_card": IntentPolicy(
        "Card details may be exposed, without a specific disputed transaction.",
        "account_security",
        "P0",
    ),
    "lost_or_stolen_card": IntentPolicy(
        "A physical card is lost or stolen.",
        "account_security",
        "P0",
    ),
    "lost_or_stolen_phone": IntentPolicy(
        "A phone tied to the banking account is lost or stolen.",
        "account_security",
        "P0",
    ),
    "cash_withdrawal_charge": IntentPolicy(
        "The customer was charged a fee for an ATM cash withdrawal.",
        "cash_withdrawals",
        "P2",
    ),
    "card_payment_fee_charged": IntentPolicy(
        "The customer was charged a fee on a card purchase.",
        "card_payments",
        "P2",
    ),
    "failed_transfer": IntentPolicy(
        "A bank transfer failed with an error.",
        "bank_transfers",
        "P1",
    ),
    "pending_transfer": IntentPolicy(
        "A transfer is pending or processing, with no claim that another "
        "person or account is missing the money.",
        "bank_transfers",
        "P2",
    ),
    "transfer_not_received_by_recipient": IntentPolicy(
        "A person, retailer, or destination account has not received a sent "
        "transfer. Prefer this whenever a recipient is waiting or missing "
        "money, even if the sender also says the transfer is pending.",
        "bank_transfers",
        "P1",
    ),
    "unable_to_verify_identity": IntentPolicy(
        "The customer cannot complete identity verification.",
        "identity_verification",
        "P1",
    ),
}


@dataclass(frozen=True)
class Ticket:
    text: str
    intent: str

    @property
    def expected(self) -> dict[str, str]:
        policy = POLICIES[self.intent]
        return {
            "intent": self.intent,
            "queue": policy.queue,
            "priority": policy.priority,
        }


def teacher_system_prompt() -> str:
    policies = "\n".join(
        (
            f"- {intent}: {policy.description} "
            f"Route to queue={policy.queue}, priority={policy.priority}."
        )
        for intent, policy in POLICIES.items()
    )
    return f"""You classify and route banking support requests.

Choose exactly one policy from this list:
{policies}

Return exactly one JSON object:
{{"intent":"<policy name>","queue":"<policy queue>","priority":"<policy priority>"}}

Use the queue and priority attached to the chosen policy. Return no Markdown,
explanation, or extra keys."""


def parse_routing(text: str) -> dict[str, str] | None:
    """Parse a response only when the entire response is the routing object."""

    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != OUTPUT_KEYS:
        return None
    if not all(isinstance(value[key], str) for key in OUTPUT_KEYS):
        return None
    return cast(dict[str, str], value)


def select_tickets(
    rows: Iterable[Mapping[str, Any]],
    *,
    per_intent: int,
    seed: int,
) -> list[Ticket]:
    """Select a deterministic, balanced subset for the configured policies."""

    if per_intent < 1:
        raise ValueError("per_intent must be positive")
    grouped = {intent: [] for intent in POLICIES}
    for row in rows:
        intent = row.get("label_text")
        text = row.get("text")
        if intent in grouped and isinstance(text, str) and text.strip():
            grouped[cast(str, intent)].append(Ticket(text=text.strip(), intent=cast(str, intent)))

    selected: list[Ticket] = []
    for intent, tickets in grouped.items():
        random.Random(f"{seed}:{intent}").shuffle(tickets)
        if len(tickets) < per_intent:
            raise ValueError(f"dataset has {len(tickets)} examples for {intent}; need {per_intent}")
        selected.extend(tickets[:per_intent])
    return selected


def load_banking77(
    *,
    dataset_id: str,
    revision: str,
    teacher_candidates_per_intent: int,
    eval_per_intent: int,
    seed: int,
) -> tuple[list[Ticket], list[Ticket]]:
    """Load balanced train candidates and held-out tickets from separate splits."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("this example requires datasets; install `opentinker[examples]`") from exc

    dataset = load_dataset(dataset_id, revision=revision)
    train = select_tickets(
        cast(Iterable[Mapping[str, Any]], dataset["train"]),
        per_intent=teacher_candidates_per_intent,
        seed=seed,
    )
    held_out = select_tickets(
        cast(Iterable[Mapping[str, Any]], dataset["test"]),
        per_intent=eval_per_intent,
        seed=seed + 1,
    )
    return train, held_out


def student_messages(ticket: Ticket) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": f"{STUDENT_INSTRUCTION}\n\nCustomer message: {ticket.text}",
        }
    ]


def teacher_messages(ticket: Ticket) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": teacher_system_prompt()},
        {"role": "user", "content": ticket.text},
    ]


def evaluate(
    label: str,
    client: Any,
    renderer: Any,
    tokenizer: Any,
    tickets: list[Ticket],
    *,
    max_tokens: int,
    include_teacher_policy: bool = False,
) -> dict[str, Any]:
    """Run inference and report strict contract and intent accuracy."""

    cases: list[dict[str, Any]] = []
    for index, ticket in enumerate(tickets, start=1):
        response = sample_text(
            client,
            renderer,
            tokenizer,
            teacher_messages(ticket) if include_teacher_policy else student_messages(ticket),
            max_tokens=max_tokens,
        )
        parsed = parse_routing(response)
        exact = parsed == ticket.expected
        intent_correct = parsed is not None and parsed["intent"] == ticket.intent
        cases.append(
            {
                "case": index,
                "text": ticket.text,
                "expected": ticket.expected,
                "response": response,
                "parsed": parsed,
                "exact": exact,
                "intent_correct": intent_correct,
            }
        )
        print(
            f"{label} case={index}/{len(tickets)} intent={ticket.intent} exact={exact}",
            flush=True,
        )

    exact_count = sum(case["exact"] for case in cases)
    intent_count = sum(case["intent_correct"] for case in cases)
    total = len(cases)
    return {
        "exact": exact_count,
        "intent_correct": intent_count,
        "total": total,
        "exact_accuracy": exact_count / total,
        "intent_accuracy": intent_count / total,
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student",
        default="Qwen/Qwen3-0.6B",
        help="small model to train and evaluate",
    )
    parser.add_argument(
        "--teacher",
        default="Qwen/Qwen3-14B",
        help="larger model that labels candidate tickets",
    )
    parser.add_argument("--dataset", default=DATASET_ID, help="Hugging Face dataset ID")
    parser.add_argument(
        "--dataset-revision",
        default=DATASET_REVISION,
        help="pinned dataset commit",
    )
    parser.add_argument(
        "--train-per-intent",
        type=int,
        default=6,
        help="verified teacher outputs retained for each intent",
    )
    parser.add_argument(
        "--teacher-candidates-per-intent",
        type=int,
        default=20,
        help="maximum teacher attempts available for each intent quota",
    )
    parser.add_argument(
        "--eval-per-intent",
        type=int,
        default=2,
        help="untouched test-split examples scored for each intent",
    )
    parser.add_argument("--epochs", type=int, default=8, help="student training epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.6,
        help="minimum distilled exact-match accuracy required for success",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="./runs/banking77-distillation",
        help="local directory for teacher data, audits, and scores",
    )
    parser.add_argument(
        "--checkpoint",
        help="evaluate an existing tinker:// sampler checkpoint instead of training",
    )
    add_compute_arguments(parser, machine_ttl="3h")
    return parser.parse_args()


def assert_improved(
    baseline: Mapping[str, Any],
    distilled: Mapping[str, Any],
    *,
    minimum_accuracy: float,
) -> None:
    """Fail the example unless held-out inference proves useful transfer."""

    if distilled["exact_accuracy"] <= baseline["exact_accuracy"]:
        raise RuntimeError("distilled student did not beat the untouched student")
    if distilled["exact_accuracy"] < minimum_accuracy:
        raise RuntimeError(
            f"distilled exact accuracy was {distilled['exact_accuracy']:.1%}; "
            f"required {minimum_accuracy:.1%}"
        )


def main() -> None:
    args = parse_args()
    if args.train_per_intent < 1:
        raise ValueError("--train-per-intent must be positive")
    if args.teacher_candidates_per_intent < args.train_per_intent:
        raise ValueError("--teacher-candidates-per-intent must be at least --train-per-intent")
    if not 0 < args.min_accuracy <= 1:
        raise ValueError("--min-accuracy must be between zero and one")
    if (
        not args.checkpoint
        and args.teacher == "Qwen/Qwen3-14B"
        and args.pool is None
        and (
            (args.gpu is None and not args.on_demand)
            or (args.gpu is not None and args.gpu.upper() == "A10G")
        )
    ):
        raise ValueError(
            "the default 14B teacher needs a GPU with at least 40 GB of memory; "
            "use --on-demand --gpu L40S or pass a suitable --pool"
        )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_candidates, held_out = load_banking77(
        dataset_id=args.dataset,
        revision=args.dataset_revision,
        teacher_candidates_per_intent=args.teacher_candidates_per_intent,
        eval_per_intent=args.eval_per_intent,
        seed=args.seed,
    )

    student_tokenizer = get_tokenizer(args.student)
    student_renderer = renderers.get_renderer(
        "qwen3_disable_thinking",
        student_tokenizer,
    )
    adapter = BeamComputeAdapter(
        base_model=args.student,
        **compute_options_from_args(args),
    )

    if args.checkpoint:
        with adapter as service_client:
            base_client = service_client.create_sampling_client(base_model=args.student)
            baseline = evaluate(
                "base_student",
                base_client,
                student_renderer,
                student_tokenizer,
                held_out,
                max_tokens=args.max_output_tokens,
            )
            checkpoint_client = service_client.create_sampling_client(model_path=args.checkpoint)
            checkpoint_eval = evaluate(
                "checkpoint_student",
                checkpoint_client,
                student_renderer,
                student_tokenizer,
                held_out,
                max_tokens=args.max_output_tokens,
            )
        assert_improved(
            baseline,
            checkpoint_eval,
            minimum_accuracy=args.min_accuracy,
        )
        checkpoint_summary = {
            "task": "Banking77 support message -> intent, queue, and priority",
            "dataset": args.dataset,
            "dataset_revision": args.dataset_revision,
            "student_model": args.student,
            "sampler_checkpoint": args.checkpoint,
            "gpu": adapter.gpu,
            "base_student": baseline,
            "checkpoint_student": checkpoint_eval,
            "container_id": adapter.container_id,
            "dashboard_url": adapter.dashboard_url,
        }
        results_path = output / "checkpoint_evaluation.json"
        results_path.write_text(json.dumps(checkpoint_summary, indent=2) + "\n")
        print(json.dumps(checkpoint_summary, indent=2), flush=True)
        return

    teacher_tokenizer = get_tokenizer(args.teacher)
    teacher_renderer = renderers.get_renderer(
        "qwen3_disable_thinking",
        teacher_tokenizer,
    )
    with adapter as service_client:
        print(
            f"Distilling {args.teacher} -> {args.student} on {adapter.gpu}; "
            f"intents={len(POLICIES)} held_out={len(held_out)}",
            flush=True,
        )
        base_client = service_client.create_sampling_client(base_model=args.student)
        baseline = evaluate(
            "base_student",
            base_client,
            student_renderer,
            student_tokenizer,
            held_out,
            max_tokens=args.max_output_tokens,
        )

        teacher_client = service_client.create_sampling_client(base_model=args.teacher)
        verified_counts = dict.fromkeys(POLICIES, 0)
        teacher_audit: list[dict[str, Any]] = []
        teacher_rows: list[dict[str, Any]] = []
        for index, ticket in enumerate(train_candidates, start=1):
            if verified_counts[ticket.intent] >= args.train_per_intent:
                continue
            response = sample_text(
                teacher_client,
                teacher_renderer,
                teacher_tokenizer,
                teacher_messages(ticket),
                max_tokens=args.max_output_tokens,
            )
            parsed = parse_routing(response)
            accepted = parsed == ticket.expected
            teacher_audit.append(
                {
                    "candidate": index,
                    "text": ticket.text,
                    "reference_intent": ticket.intent,
                    "expected": ticket.expected,
                    "teacher_response": response,
                    "parsed": parsed,
                    "accepted": accepted,
                }
            )
            print(
                f"teacher candidate={index}/{len(train_candidates)} "
                f"intent={ticket.intent} accepted={accepted}",
                flush=True,
            )
            if not accepted:
                continue
            verified_counts[ticket.intent] += 1
            teacher_rows.append(
                {
                    "prompt": f"Customer message: {ticket.text}",
                    "teacher_response": json.dumps(parsed, separators=(",", ":")),
                    "verified": True,
                    "reference_intent": ticket.intent,
                    "teacher_model": args.teacher,
                    "dataset": args.dataset,
                }
            )

        missing = {
            intent: args.train_per_intent - count
            for intent, count in verified_counts.items()
            if count < args.train_per_intent
        }
        audit_path = write_jsonl(output / "teacher_audit.jsonl", teacher_audit)
        if missing:
            raise RuntimeError(f"teacher did not fill the per-intent quota: {missing}")
        teacher_data_path = write_jsonl(
            output / "verified_teacher_data.jsonl",
            teacher_rows,
        )

        teacher_eval = evaluate(
            "teacher",
            teacher_client,
            teacher_renderer,
            teacher_tokenizer,
            held_out,
            max_tokens=args.max_output_tokens,
            include_teacher_policy=True,
        )
        datums = distillation_records_to_datums(
            teacher_rows,
            renderer=student_renderer,
            max_length=args.max_length,
            instruction=STUDENT_INSTRUCTION,
            require_verified=True,
        )
        checkpoint_name = f"banking77-router-{int(time.time())}"
        training = train_lora(
            service_client,
            base_model=args.student,
            datums=datums,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_name=checkpoint_name,
            rank=8,
            seed=args.seed,
        )
        tuned_client = service_client.create_sampling_client(model_path=training.sampler_checkpoint)
        distilled = evaluate(
            "distilled_student",
            tuned_client,
            student_renderer,
            student_tokenizer,
            held_out,
            max_tokens=args.max_output_tokens,
        )

    assert_improved(
        baseline,
        distilled,
        minimum_accuracy=args.min_accuracy,
    )
    summary = {
        "task": "Banking77 support message -> intent, queue, and priority",
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "policies": {intent: asdict(policy) for intent, policy in POLICIES.items()},
        "teacher_model": args.teacher,
        "student_model": args.student,
        "gpu": adapter.gpu,
        "verified_teacher_examples": len(teacher_rows),
        "teacher_acceptance": {
            "accepted": len(teacher_rows),
            "attempted": len(teacher_audit),
            "per_intent": verified_counts,
        },
        "base_student": baseline,
        "teacher": teacher_eval,
        "distilled_student": distilled,
        "training_initial_nll": training.losses[0],
        "training_final_nll": training.losses[-1],
        "state_checkpoint": training.state_checkpoint,
        "sampler_checkpoint": training.sampler_checkpoint,
        "teacher_audit": str(audit_path),
        "teacher_dataset": str(teacher_data_path),
        "container_id": adapter.container_id,
        "dashboard_url": adapter.dashboard_url,
    }
    results_path = output / "results.json"
    results_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "A/B exact accuracy: "
        f"base={baseline['exact_accuracy']:.1%} "
        f"teacher={teacher_eval['exact_accuracy']:.1%} "
        f"distilled={distilled['exact_accuracy']:.1%}",
        flush=True,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
