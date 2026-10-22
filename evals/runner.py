"""Run golden fixtures through the real IntentAgent.run_extraction and judge each response.

The fixture's data model and policies replace the platform's; no A2A transport or token introspection.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from evals.fixtures import Fixture, load_schema, policies_for_subject
from evals.scoring import FixtureResult, exception_result, judge, parse_response
from indykite.intent_agent.agent import IntentAgent, RequestSubject
from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import llm_timeout_error, request_timeout_error

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 1
# One timeout is provider noise; a fixture that times out twice in a row is a real finding.
DEFAULT_TIMEOUT_RETRIES = 1
# Generous, so a slow provider shows up as latency (the p95 gate) rather than as a wrong answer.
DEFAULT_LLM_TIMEOUT_SECONDS = 20.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0

ResultListener = Callable[[FixtureResult], None]


@dataclass(frozen=True, slots=True)
class RunOptions:
    """How a suite run is paced: fixtures in flight at once, and reruns for a timed-out fixture."""

    concurrency: int = DEFAULT_CONCURRENCY
    timeout_retries: int = DEFAULT_TIMEOUT_RETRIES


DEFAULT_RUN_OPTIONS = RunOptions()


class FixtureSchemaClient:
    """The two SchemaClient methods IntentAgent uses, answered from a fixture data model."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema

    def preload(self) -> None:
        """Nothing to fetch - the fixture is already loaded."""

    def get_cached(self) -> dict[str, Any]:
        """The fixture, unconditionally."""
        return self._schema


class FixturePolicyClient:
    """The one PolicyClient method IntentAgent uses, answered from a fixture policy listing."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._entries = entries

    def get_cached(self, subject_type: str) -> tuple[list[dict[str, Any]], bool]:
        """The entries for `subject_type` (fixtures.policies_for_subject); the fetch never fails."""
        return policies_for_subject(self._entries, subject_type), False


def build_agents(
    fixtures: Iterable[Fixture],
    settings: Settings,
    *,
    llm: BaseChatModel | None = None,
) -> dict[Path, IntentAgent]:
    """One IntentAgent per distinct data model, serving that model's schema and policies.

    `llm` (a test double) is shared by all of them; None builds the configured Gemini model.
    """
    by_model = {fixture.schema_path: fixture for fixture in fixtures}
    return {
        schema_path: IntentAgent(
            settings,
            llm=llm,
            schema_client=FixtureSchemaClient(load_schema(schema_path)),
            policy_client=FixturePolicyClient(fixture.policies),
        )
        for schema_path, fixture in sorted(by_model.items())
    }


def _timed_out(prompt: str, actual: dict[str, Any]) -> bool:
    """True when `actual` is one of the pipeline's two timeout errors for `prompt`."""
    return actual in (llm_timeout_error(prompt), request_timeout_error(prompt))


async def _extract_once(fixture: Fixture, agent: IntentAgent) -> FixtureResult:
    """One extraction, judged; an exception out of the pipeline is recorded against the fixture."""
    subject = RequestSubject(id=fixture.jwt_sub, type=fixture.subject_type)
    started = time.perf_counter()
    try:
        result = await agent.run_extraction(fixture.prompt, subject)
    except Exception as exc:  # per-fixture isolation; the failure is reported, not swallowed
        logger.exception("Fixture %s: extraction raised", fixture.id)
        return exception_result(fixture, exc, latency_seconds=time.perf_counter() - started)
    actual = parse_response(result.text)
    return FixtureResult(
        fixture=fixture,
        actual=actual,
        outcome=result.outcome.value,
        verdict=judge(fixture, actual, result.confidence),
        validation_retries=result.validation_retries,
        latency_seconds=time.perf_counter() - started,
        confidence=result.confidence,
        explanation=result.explanation,
    )


async def run_fixture(
    fixture: Fixture,
    agent: IntentAgent,
    *,
    timeout_retries: int = DEFAULT_TIMEOUT_RETRIES,
) -> FixtureResult:
    """Extract and judge one fixture; an exception is recorded as its result instead of aborting the run.

    A timeout-only response is rerun up to `timeout_retries` times; the reruns spent are set on the result.
    """
    result = await _extract_once(fixture, agent)
    retries = 0
    while retries < timeout_retries and _timed_out(fixture.prompt, result.actual):
        retries += 1
        logger.warning(
            "Fixture %s: timed out (%s); rerunning (%d/%d)", fixture.id, result.outcome, retries, timeout_retries
        )
        result = await _extract_once(fixture, agent)
    return replace(result, timeout_retries=retries) if retries else result


async def run_suite(
    fixtures: list[Fixture],
    settings: Settings,
    *,
    llm: BaseChatModel | None = None,
    options: RunOptions = DEFAULT_RUN_OPTIONS,
    on_result: ResultListener | None = None,
) -> list[FixtureResult]:
    """Run every fixture, at most `options.concurrency` at a time, in one event loop.

    One loop because the LLM client's HTTP session is bound to it. Results come back in fixture
    order; `on_result` is called as each completes. Raises ValueError for concurrency < 1.
    """
    if options.concurrency < 1:
        message = "concurrency must be at least 1"
        raise ValueError(message)
    agents = build_agents(fixtures, settings, llm=llm)
    semaphore = asyncio.Semaphore(options.concurrency)

    async def _run(fixture: Fixture) -> FixtureResult:
        async with semaphore:
            result = await run_fixture(fixture, agents[fixture.schema_path], timeout_retries=options.timeout_retries)
        if on_result is not None:
            on_result(result)
        return result

    return list(await asyncio.gather(*(_run(fixture) for fixture in fixtures)))
