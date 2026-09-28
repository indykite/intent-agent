"""Rendering a finished eval run: the gate verdicts, the per-category breakdown and every
failure with an expected/actual diff - as plain text for the terminal and CI log, Markdown
for the GitHub job summary / PR comment, and JSON for the run artifact.
"""

import difflib
import html
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from evals.scoring import (
    MAX_P95_LATENCY_SECONDS,
    CategorySummary,
    FixtureResult,
    Gate,
    evaluate_gates,
    percent,
    summarize_by_category,
)

_STATUS = {True: "PASS", False: "FAIL"}
_MARK = {True: "✅", False: "❌"}


@dataclass(frozen=True, slots=True)
class Report:
    """Everything the renderers need about one run."""

    generated_at: str
    model: str
    min_correctness: float
    max_p95_latency: float
    results: list[FixtureResult]
    categories: list[CategorySummary]
    gates: list[Gate]

    @property
    def passed(self) -> bool:
        """True only when every gate passed - the run's exit status."""
        return all(gate.passed for gate in self.gates)

    @property
    def failures(self) -> list[FixtureResult]:
        """The results that didn't match, in fixture order."""
        return [result for result in self.results if not result.passed]


def build_report(
    results: list[FixtureResult],
    *,
    model: str,
    min_correctness: float,
    max_p95_latency: float = MAX_P95_LATENCY_SECONDS,
    full_suite: bool = True,
) -> Report:
    """Score a run: categories and gates over `results`. `full_suite` is False for a `--only`
    subset, where a missing adversarial category is the selection, not a gap in the suite.
    """
    return Report(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        model=model,
        min_correctness=min_correctness,
        max_p95_latency=max_p95_latency,
        results=results,
        categories=summarize_by_category(results),
        gates=evaluate_gates(
            results, min_correctness=min_correctness, max_p95_latency=max_p95_latency, full_suite=full_suite
        ),
    )


def _pretty(value: Any) -> str:  # noqa: ANN401 - any JSON value
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def response_diff(result: FixtureResult) -> str:
    """A unified diff from the golden answer to the response, keys sorted so only real
    differences show. Empty for a passing result.
    """
    if result.passed:
        return ""
    lines = difflib.unified_diff(
        _pretty(result.fixture.expected).splitlines(),
        _pretty(result.actual).splitlines(),
        fromfile="expected",
        tofile="actual",
        lineterm="",
    )
    return "\n".join(lines)


def _fence(content: str, info: str = "") -> list[str]:
    """`content` in a Markdown code fence longer than any run of backticks inside it, so a
    hostile fixture prompt or a diff containing ``` can't close the block early.
    """
    longest = max((len(run) for run in re.findall(r"`+", content)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence + info, content, fence]


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """A fixed-width text table."""
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows, strict=True)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    return [fmt.format(*headers), fmt.format(*("-" * width for width in widths)), *(fmt.format(*row) for row in rows)]


def _category_rows(report: Report) -> list[list[str]]:
    return [
        [
            summary.category,
            str(summary.total) if summary.total else "0 (no fixtures)",
            str(summary.passed),
            str(summary.failed),
            str(summary.forced_matches),
            str(summary.exceptions),
            percent(summary.pass_rate),
        ]
        for summary in report.categories
    ]


_CATEGORY_HEADERS = ["category", "fixtures", "passed", "failed", "forced matches", "exceptions", "pass rate"]


def _gate_lines(report: Report) -> list[str]:
    """One line per gate, and its offenders on the next when it has any - a slow fixture that
    still matched its golden answer appears nowhere else in the report.
    """
    lines: list[str] = []
    for gate in report.gates:
        lines.append(f"  [{_STATUS[gate.passed]}] {gate.name}: {gate.observed} (required: {gate.required})")
        if gate.offenders:
            lines.append(f"    offenders: {', '.join(gate.offenders)}")
    return lines


def render_text(report: Report) -> str:
    """The terminal report: gates, categories, then each failure with its diff."""
    total = len(report.results)
    passed = sum(result.passed for result in report.results)
    rerun = sum(1 for result in report.results if result.timeout_retries)
    lines = [
        f"Golden eval suite - model {report.model} - {passed}/{total} fixtures passed - {report.generated_at}",
        *([f"({rerun} fixture(s) rerun after a timeout - see the JSON report's timeout_retries)"] if rerun else []),
        "",
        "Gates:",
        *_gate_lines(report),
        "",
        "Per category:",
        *(f"  {line}" for line in _table(_CATEGORY_HEADERS, _category_rows(report))),
    ]
    failures = report.failures
    if failures:
        lines += ["", f"Failures ({len(failures)}):"]
        for result in failures:
            lines += [
                "",
                f"  {result.fixture.id} [{result.fixture.category}] outcome={result.outcome} - {result.verdict.detail}",
                f"    prompt: {result.fixture.prompt}",
                *(f"    {line}" for line in response_diff(result).splitlines()),
            ]
    lines += ["", f"Result: {_STATUS[report.passed]}"]
    return "\n".join(lines) + "\n"


def _markdown_offenders(report: Report) -> list[str]:
    """A bullet per gate with offenders, `<code>`-wrapped since ids are fixture file stems."""
    bullets = [
        f"- **{gate.name}** offenders: {', '.join(f'`{html.escape(offender)}`' for offender in gate.offenders)}"
        for gate in report.gates
        if gate.offenders
    ]
    return ["", *bullets] if bullets else []


def render_markdown(report: Report) -> str:
    """The GitHub job summary / PR comment: the same content as tables, failures collapsed."""
    total = len(report.results)
    passed = sum(result.passed for result in report.results)
    lines = [
        f"## Golden eval suite: {_MARK[report.passed]} {_STATUS[report.passed]}",
        "",
        f"Model `{report.model}` · {passed}/{total} fixtures passed · {report.generated_at}",
        "",
        "| Gate | Required | Observed | Status |",
        "| --- | --- | --- | --- |",
        *(
            f"| {gate.name} | {gate.required} | {gate.observed} | {_MARK[gate.passed]} {_STATUS[gate.passed]} |"
            for gate in report.gates
        ),
        # Offenders under the table rather than as a column: the ids are long, the list can be,
        # and most gates have none on a green run.
        *_markdown_offenders(report),
        "",
        "| Category | Fixtures | Passed | Failed | Forced matches | Exceptions | Pass rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *(f"| {' | '.join(row)} |" for row in _category_rows(report)),
    ]
    failures = report.failures
    if failures:
        lines += ["", f"### Failures ({len(failures)})"]
        for result in failures:
            # Everything model- or fixture-authored is HTML-escaped inside the <summary> and
            # fenced elsewhere: adversarial prompts are hostile text by design.
            summary = (
                f"<code>{html.escape(result.fixture.id)}</code> [{html.escape(result.fixture.category)}] "
                f"outcome={html.escape(result.outcome)} - {html.escape(result.verdict.detail)}"
            )
            lines += [
                "",
                "<details>",
                f"<summary>{summary}</summary>",
                "",
                "Prompt:",
                "",
                *_fence(result.fixture.prompt, "text"),
                "",
                *_fence(response_diff(result), "diff"),
                "",
                "</details>",
            ]
    return "\n".join(lines) + "\n"


def to_json(report: Report) -> dict[str, Any]:
    """The run as a JSON-serialisable document (the CI artifact)."""
    return {
        "generated_at": report.generated_at,
        "model": report.model,
        "passed": report.passed,
        "min_correctness": report.min_correctness,
        "max_p95_latency": report.max_p95_latency,
        "gates": [
            {
                "name": gate.name,
                "required": gate.required,
                "observed": gate.observed,
                "passed": gate.passed,
                "offenders": gate.offenders,
            }
            for gate in report.gates
        ],
        "categories": [
            {
                "category": summary.category,
                "total": summary.total,
                "passed": summary.passed,
                "failed": summary.failed,
                "forced_matches": summary.forced_matches,
                "exceptions": summary.exceptions,
                "pass_rate": summary.pass_rate,
            }
            for summary in report.categories
        ],
        "fixtures": [
            {
                "id": result.fixture.id,
                "category": result.fixture.category,
                "data_model": result.fixture.data_model,
                "prompt": result.fixture.prompt,
                "passed": result.passed,
                "forced_match": result.verdict.forced_match,
                "detail": result.verdict.detail,
                "outcome": result.outcome,
                "validation_retries": result.validation_retries,
                "latency_seconds": round(result.latency_seconds, 3),
                "timeout_retries": result.timeout_retries,
                "expected": result.fixture.expected,
                "actual": result.actual,
            }
            for result in report.results
        ],
    }
