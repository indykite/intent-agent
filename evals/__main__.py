"""Run the golden eval suite against the configured Gemini model and gate on the result.

    uv run python -m evals
    uv run python -m evals --only 'adversarial_*' --concurrency 4
    uv run python -m evals --markdown report.md --json report.json

Settings come from the environment / `.env` like the service's own (GEMINI_API_KEY at minimum;
GEMINI_MODEL and VALIDATION_MAX_RETRIES are what is under test, so they are read the same way
rather than overridden here). The two timeouts are the exception: the suite runs under its own,
generous ones (--llm-timeout / --request-timeout, see runner.DEFAULT_*_TIMEOUT_SECONDS) so that a
slow provider registers as latency - the p95 gate's job - rather than as a wrong answer. The other
required settings - introspection, platform URL, audit delivery - aren't used by the suite but
Settings still needs them to construct; any placeholder value works (see
.github/workflows/eval-suite.yaml).

Exit status: 0 when every gate passes, 1 when any gate fails, 2 for a usage or fixture error.
"""

import argparse
import asyncio
import fnmatch
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from evals.fixtures import FIXTURES_DIR, Fixture, FixtureError, load_fixtures
from evals.report import build_report, render_markdown, render_text, to_json
from evals.runner import (
    DEFAULT_CONCURRENCY,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_RETRIES,
    RunOptions,
    run_suite,
)
from evals.scoring import MAX_P95_LATENCY_SECONDS, MIN_CORRECTNESS, FixtureResult
from indykite.intent_agent.config import Settings

EXIT_GATE_FAILED = 1
EXIT_USAGE = 2

logger = logging.getLogger("evals")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_DIR, help="directory of fixture JSON files")
    parser.add_argument("--only", metavar="GLOB", help="run only fixtures whose id matches this glob")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="fixtures extracted at the same time (default %(default)s)",
    )
    parser.add_argument(
        "--timeout-retries",
        type=int,
        default=DEFAULT_TIMEOUT_RETRIES,
        help="reruns for a fixture whose only outcome was a timeout, 0 disables (default %(default)s)",
    )
    parser.add_argument(
        "--min-correctness",
        type=float,
        default=MIN_CORRECTNESS,
        help="correctness gate threshold, 0-1 (default %(default)s)",
    )
    parser.add_argument(
        "--max-p95-latency",
        type=float,
        default=MAX_P95_LATENCY_SECONDS,
        help="p95 latency gate ceiling in seconds per prompt, retries included (default %(default)s)",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=DEFAULT_LLM_TIMEOUT_SECONDS,
        help="LLM_TIMEOUT_SECONDS the pipeline runs under, per call (default %(default)s)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="REQUEST_TIMEOUT_SECONDS the pipeline runs under, per prompt (default %(default)s)",
    )
    parser.add_argument("--markdown", type=Path, help="also write the report as Markdown to this file")
    parser.add_argument("--json", type=Path, help="also write the full results as JSON to this file")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.timeout_retries < 0:
        parser.error("--timeout-retries must not be negative")
    if not 0.0 <= args.min_correctness <= 1.0:
        parser.error("--min-correctness must be between 0 and 1")
    if not args.max_p95_latency > 0:  # written this way round so NaN is rejected too
        parser.error("--max-p95-latency must be positive")
    if not args.llm_timeout > 0 or not args.request_timeout > 0:
        parser.error("--llm-timeout and --request-timeout must be positive")
    if args.llm_timeout > args.request_timeout:
        parser.error("--llm-timeout must not exceed --request-timeout")
    return args


def _select(fixtures: list[Fixture], pattern: str | None) -> list[Fixture]:
    if pattern is None:
        return fixtures
    selected = [fixture for fixture in fixtures if fnmatch.fnmatchcase(fixture.id, pattern)]
    if not selected:
        message = f"no fixture id matches {pattern!r}"
        raise FixtureError(message)
    return selected


def _log_progress(result: FixtureResult) -> None:
    logger.info(
        "%s %s [%s] outcome=%s %.1fs - %s",
        "PASS" if result.passed else "FAIL",
        result.fixture.id,
        result.fixture.category,
        result.outcome,
        result.latency_seconds,
        result.verdict.detail,
    )


def main(argv: list[str]) -> int:
    """Load the fixtures, run them, render the report(s), and return the exit status."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    logging.getLogger("indykite").setLevel(logging.WARNING)  # the per-request INFO lines are noise here
    logging.getLogger("httpx").setLevel(logging.WARNING)  # so is the LLM client's line per HTTP call
    args = _parse_args(argv)
    try:
        fixtures = _select(load_fixtures(args.fixtures), args.only)
        settings = Settings(llm_timeout_seconds=args.llm_timeout, request_timeout_seconds=args.request_timeout)
    except (FixtureError, ValidationError) as exc:
        logger.error("%s", exc)  # noqa: TRY400 - a configuration error, not a crash to trace
        return EXIT_USAGE

    logger.info(
        "Running %d fixture(s) against %s (concurrency %d)", len(fixtures), settings.gemini_model, args.concurrency
    )
    results = asyncio.run(
        run_suite(
            fixtures,
            settings,
            options=RunOptions(concurrency=args.concurrency, timeout_retries=args.timeout_retries),
            on_result=_log_progress,
        ),
    )
    report = build_report(
        results,
        model=settings.gemini_model,
        min_correctness=args.min_correctness,
        max_p95_latency=args.max_p95_latency,
        full_suite=args.only is None,
    )

    sys.stdout.write(render_text(report))
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(to_json(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if report.passed else EXIT_GATE_FAILED


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
