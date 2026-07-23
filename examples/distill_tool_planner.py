"""Distill structured multi-step tool planning from a 4B teacher into a 0.6B student."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opentinker import BeamComputeAdapter
from opentinker._distillation import extract_json_object, sample_text, train_lora
from opentinker._examples import add_compute_arguments, compute_options_from_args
from opentinker.data import distillation_records_to_datums, write_jsonl

TEACHER_SYSTEM_PROMPT = """You are a workflow compiler. Convert the user's analytics request into
one JSON object with exactly one key, \"calls\". The value is an ordered array of tool calls.

Available tools and exact argument schemas:
- read_csv(path)
- filter_rows(input, column, operator, value), where operator is \"eq\"
- aggregate(input, group_by, metric, operation), where operation is \"sum\" or \"average\"
- sort(input, by, direction), where direction is \"ascending\" or \"descending\"
- write_json(input, path)
- notify(channel, message, artifact)

Every call must have exactly: {\"id\":\"sN\",\"tool\":\"...\",\"args\":{...}}.
Number IDs from s1 in execution order. Refer to prior results as \"$sN\". Always perform all six
steps requested: read, filter, aggregate, sort, write, notify. The notification message must be
\"<operation> <metric> by <group_by> report ready\" using the literal request values. Return JSON
only: no Markdown, prose, or comments.

Example request:
Read /data/orders.csv. Keep rows where status equals paid. Group by region and sum revenue. Sort
revenue descending. Write /reports/revenue.json. Notify #finance when it is ready.

Example response:
{"calls":[{"id":"s1","tool":"read_csv","args":{"path":"/data/orders.csv"}},{"id":"s2","tool":"filter_rows","args":{"input":"$s1","column":"status","operator":"eq","value":"paid"}},{"id":"s3","tool":"aggregate","args":{"input":"$s2","group_by":"region","metric":"revenue","operation":"sum"}},{"id":"s4","tool":"sort","args":{"input":"$s3","by":"revenue","direction":"descending"}},{"id":"s5","tool":"write_json","args":{"input":"$s4","path":"/reports/revenue.json"}},{"id":"s6","tool":"notify","args":{"channel":"#finance","message":"sum revenue by region report ready","artifact":"$s5"}}]}
"""

STUDENT_INSTRUCTION = (
    "Compile this analytics request into the required JSON workflow. Return JSON only."
)


@dataclass(frozen=True)
class Scenario:
    input_path: str
    filter_column: str
    filter_value: str
    group_by: str
    metric: str
    operation: str
    direction: str
    output_path: str
    channel: str

    @property
    def request(self) -> str:
        return (
            f"Read {self.input_path}. Keep rows where {self.filter_column} equals "
            f"{self.filter_value}. Group by {self.group_by} and {self.operation} {self.metric}. "
            f"Sort {self.metric} {self.direction}. Write {self.output_path}. Notify "
            f"{self.channel} when it is ready."
        )

    @property
    def plan(self) -> dict[str, Any]:
        return {
            "calls": [
                {"id": "s1", "tool": "read_csv", "args": {"path": self.input_path}},
                {
                    "id": "s2",
                    "tool": "filter_rows",
                    "args": {
                        "input": "$s1",
                        "column": self.filter_column,
                        "operator": "eq",
                        "value": self.filter_value,
                    },
                },
                {
                    "id": "s3",
                    "tool": "aggregate",
                    "args": {
                        "input": "$s2",
                        "group_by": self.group_by,
                        "metric": self.metric,
                        "operation": self.operation,
                    },
                },
                {
                    "id": "s4",
                    "tool": "sort",
                    "args": {
                        "input": "$s3",
                        "by": self.metric,
                        "direction": self.direction,
                    },
                },
                {
                    "id": "s5",
                    "tool": "write_json",
                    "args": {"input": "$s4", "path": self.output_path},
                },
                {
                    "id": "s6",
                    "tool": "notify",
                    "args": {
                        "channel": self.channel,
                        "message": (
                            f"{self.operation} {self.metric} by {self.group_by} report ready"
                        ),
                        "artifact": "$s5",
                    },
                },
            ]
        }


def scenarios() -> tuple[list[Scenario], list[Scenario]]:
    inputs = [
        ("/data/orders.csv", "status", "paid", "region", "revenue", "sum"),
        ("/data/tickets.csv", "priority", "urgent", "owner", "resolution_hours", "average"),
        ("/data/usage.csv", "plan", "enterprise", "country", "tokens", "sum"),
        ("/data/shipments.csv", "state", "delivered", "carrier", "delay_hours", "average"),
        ("/data/invoices.csv", "currency", "USD", "customer", "amount", "sum"),
        ("/data/incidents.csv", "severity", "critical", "service", "downtime_minutes", "sum"),
        ("/data/leads.csv", "stage", "qualified", "campaign", "deal_value", "average"),
        ("/data/refunds.csv", "reason", "damaged", "warehouse", "refund_amount", "sum"),
        ("/data/sessions.csv", "device", "mobile", "source", "duration_seconds", "average"),
        ("/data/claims.csv", "status", "approved", "adjuster", "payout", "sum"),
        ("/data/alerts.csv", "level", "warning", "cluster", "count", "sum"),
        ("/data/reviews.csv", "verified", "true", "product", "rating", "average"),
        ("/data/costs.csv", "environment", "production", "team", "cost", "sum"),
        ("/data/builds.csv", "result", "failed", "repository", "duration", "average"),
        ("/data/clicks.csv", "campaign", "summer", "browser", "conversions", "sum"),
        ("/data/returns.csv", "condition", "opened", "category", "loss", "average"),
        ("/data/logins.csv", "result", "success", "tenant", "latency_ms", "average"),
        ("/data/payments.csv", "method", "card", "merchant", "fees", "sum"),
        ("/data/exports.csv", "format", "parquet", "workspace", "bytes", "sum"),
        ("/data/messages.csv", "language", "French", "queue", "wait_seconds", "average"),
        ("/data/jobs.csv", "status", "timeout", "worker_pool", "runtime_seconds", "average"),
        ("/data/signups.csv", "source", "partner", "country", "users", "sum"),
        ("/data/scans.csv", "finding", "vulnerable", "project", "risk_score", "average"),
        ("/data/latency.csv", "endpoint", "checkout", "region", "p99_ms", "average"),
    ]
    generated = [
        Scenario(
            input_path=input_path,
            filter_column=filter_column,
            filter_value=filter_value,
            group_by=group_by,
            metric=metric,
            operation=operation,
            direction="descending" if index % 2 == 0 else "ascending",
            output_path=f"/reports/distilled_{index + 1}.json",
            channel=("#analytics", "#operations", "#engineering")[index % 3],
        )
        for index, (
            input_path,
            filter_column,
            filter_value,
            group_by,
            metric,
            operation,
        ) in enumerate(inputs)
    ]
    return generated[:18], generated[18:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--output", default="./runs/tool-planner-distillation")
    parser.add_argument(
        "--checkpoint",
        help="evaluate an existing tinker:// sampler checkpoint instead of training",
    )
    add_compute_arguments(parser, machine_ttl="2h")
    return parser.parse_args()


def extract_plan(text: str) -> dict[str, Any] | None:
    value = extract_json_object(text)
    return value if value is not None and isinstance(value.get("calls"), list) else None


def student_messages(scenario: Scenario) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": f"{STUDENT_INSTRUCTION}\n\n{scenario.request}",
        }
    ]


def evaluate(
    label: str,
    client: Any,
    renderer: Any,
    tokenizer: Any,
    held_out: list[Scenario],
    max_tokens: int,
    *,
    include_teacher_contract: bool = False,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for index, scenario in enumerate(held_out, start=1):
        messages = (
            [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": scenario.request},
            ]
            if include_teacher_contract
            else student_messages(scenario)
        )
        response = sample_text(
            client,
            renderer,
            tokenizer,
            messages,
            max_tokens=max_tokens,
        )
        parsed = extract_plan(response)
        passed = parsed == scenario.plan
        cases.append(
            {
                "case": index,
                "passed": passed,
                "request": scenario.request,
                "response": response,
            }
        )
        print(f"{label} case={index}/{len(held_out)} passed={passed}", flush=True)
    passed_count = sum(case["passed"] for case in cases)
    return {
        "passed": passed_count,
        "total": len(cases),
        "accuracy": passed_count / len(cases),
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_scenarios, held_out = scenarios()

    student_tokenizer = get_tokenizer(args.student)
    student_renderer = renderers.get_renderer("qwen3", student_tokenizer)

    adapter = BeamComputeAdapter(
        base_model=args.student,
        **compute_options_from_args(args),
    )
    if args.checkpoint:
        with adapter as service_client:
            print(
                f"Evaluating {args.checkpoint} on {adapter.gpu}; held_out={len(held_out)}",
                flush=True,
            )
            checkpoint_client = service_client.create_sampling_client(model_path=args.checkpoint)
            checkpoint_eval = evaluate(
                "checkpoint_student",
                checkpoint_client,
                student_renderer,
                student_tokenizer,
                held_out,
                args.max_output_tokens,
            )
        checkpoint_summary = {
            "task": "natural-language analytics request -> six-step JSON tool plan",
            "student": args.student,
            "gpu": adapter.gpu,
            "sampler_checkpoint": args.checkpoint,
            "checkpoint_student": checkpoint_eval,
            "container_id": adapter.container_id,
            "dashboard_url": adapter.dashboard_url,
        }
        results_path = output / "checkpoint_evaluation.json"
        results_path.write_text(json.dumps(checkpoint_summary, indent=2) + "\n")
        print(json.dumps(checkpoint_summary, indent=2), flush=True)
        if checkpoint_eval["accuracy"] < 0.5:
            raise RuntimeError("checkpoint did not demonstrate the held-out tool-planning skill")
        return

    teacher_tokenizer = get_tokenizer(args.teacher)
    teacher_renderer = renderers.get_renderer("qwen3_instruct", teacher_tokenizer)
    with adapter as service_client:
        print(
            f"Distilling {args.teacher} -> {args.student} on {adapter.gpu}; "
            f"train={len(train_scenarios)} held_out={len(held_out)}",
            flush=True,
        )
        base_student = service_client.create_sampling_client(base_model=args.student)
        baseline = evaluate(
            "base_student",
            base_student,
            student_renderer,
            student_tokenizer,
            held_out,
            args.max_output_tokens,
        )

        teacher = service_client.create_sampling_client(base_model=args.teacher)
        dataset_rows: list[dict[str, Any]] = []
        for index, scenario in enumerate(train_scenarios, start=1):
            raw = sample_text(
                teacher,
                teacher_renderer,
                teacher_tokenizer,
                [
                    {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                    {"role": "user", "content": scenario.request},
                ],
                max_tokens=args.max_output_tokens,
            )
            plan = extract_plan(raw)
            accepted = plan == scenario.plan
            print(
                f"teacher train_case={index}/{len(train_scenarios)} accepted={accepted}",
                flush=True,
            )
            if not accepted:
                continue
            canonical_response = json.dumps(plan, separators=(",", ":"))
            dataset_rows.append(
                {
                    "prompt": scenario.request,
                    "teacher_response": canonical_response,
                    "verified": True,
                }
            )

        minimum_examples = max(12, math.ceil(len(train_scenarios) * 0.75))
        if len(dataset_rows) < minimum_examples:
            raise RuntimeError(
                f"teacher produced only {len(dataset_rows)} verified plans; "
                f"need at least {minimum_examples}"
            )
        dataset_path = write_jsonl(output / "verified_teacher_data.jsonl", dataset_rows)

        teacher_eval = evaluate(
            "teacher",
            teacher,
            teacher_renderer,
            teacher_tokenizer,
            held_out,
            args.max_output_tokens,
            include_teacher_contract=True,
        )

        datums = distillation_records_to_datums(
            dataset_rows,
            renderer=student_renderer,
            max_length=args.max_length,
            instruction=STUDENT_INSTRUCTION,
            require_verified=True,
        )
        checkpoint_name = f"tool-planner-{int(time.time())}"
        training = train_lora(
            service_client,
            base_model=args.student,
            datums=datums,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_name=checkpoint_name,
            rank=8,
            seed=0,
        )
        tuned_client = service_client.create_sampling_client(model_path=training.sampler_checkpoint)
        distilled = evaluate(
            "distilled_student",
            tuned_client,
            student_renderer,
            student_tokenizer,
            held_out,
            args.max_output_tokens,
        )

    summary = {
        "task": "natural-language analytics request -> six-step JSON tool plan",
        "teacher_model": args.teacher,
        "student": args.student,
        "gpu": adapter.gpu,
        "verified_teacher_examples": len(dataset_rows),
        "teacher": teacher_eval,
        "base_student": baseline,
        "distilled_student": distilled,
        "training_initial_nll": training.losses[0],
        "training_final_nll": training.losses[-1],
        "state_checkpoint": training.state_checkpoint,
        "sampler_checkpoint": training.sampler_checkpoint,
        "teacher_dataset": str(dataset_path),
        "container_id": adapter.container_id,
        "dashboard_url": adapter.dashboard_url,
    }
    results_path = output / "results.json"
    results_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if distilled["accuracy"] <= baseline["accuracy"]:
        raise RuntimeError("distilled student did not improve over the base student")
    if distilled["accuracy"] < 0.5:
        raise RuntimeError("distilled student did not demonstrate the held-out tool-planning skill")


if __name__ == "__main__":
    main()
