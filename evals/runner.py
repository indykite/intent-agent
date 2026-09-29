"""Run golden fixtures through the real extraction pipeline (IntentAgent.run_extraction) with
the real LLM, the fixture's data model served in place of the platform schema, and judge each
response against its golden answer.

Everything downstream of the model is the production code path - skill activation, structured
output, the validation/retry loop, subject binding, context normalisation - so a regression in
any of it shows up here, not only a regression in the model or the skill text. No A2A transport
or token introspection is involved: the subject is bound the way IAG supplies it per request.
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
# How many times a fixture whose only outcome was a timeout is rerun before that timeout counts
# as its result. One: a single latency spike at the per-call limit is provider noise that would
# otherwise fail an absolute gate (a refusal fixture answered with the timeout error is a miss),
# while a fixture that times out twice in a row is a real latency finding and stays a miss.
DEFAULT_TIMEOUT_RETRIES = 1
# The timeouts the suite runs the pipeline under, in place of the service's production values
# (3 s per call, 10 s per request). Those are sized for the PRD's latency budget, which the p95
# latency gate now measures directly from wall time; left in force here they turn a slow
# provider hour into correctness failures instead ("the model did not respond in time" read as
# a wrong answer), which is the wrong gate failing for the wrong reason. These only bound a
# genuinely stuck call - a slow one still shows up, as the latency it actually took.
DEFAULT_LLM_TIMEOUT_SECONDS = 20.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0

ResultListener = Callable[[FixtureResult], None]


@dataclass(frozen=True, slots=True)
class RunOptions:
    """How a suite run is paced: fixtures in flight at once, and reruns granted to a fixture
    whose only outcome was a timeout (see run_fixture).
    """

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
    """The one PolicyClient method IntentAgent uses, answered from a fixture policy listing:
    `get_cached(subject_type)` returns the entries whose policy subject type matches - the same
    narrowing the platform's `GET /access/v1/policies?subject_type=` applies - and reports the
    fetch as succeeded.
    """

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._entries = entries

    def get_cached(self, subject_type: str) -> tuple[list[dict[str, Any]], bool]:
        """(matching entries, fetch_failed=False) - the same narrowing the loader checks golden
        answers against (fixtures.policies_for_subject).
        """
        return policies_for_subject(self._entries, subject_type), False


def build_agents(
    fixtures: Iterable[Fixture],
    settings: Settings,
    *,
    llm: BaseChatModel | None = None,
) -> dict[Path, IntentAgent]:
    """One IntentAgent per distinct data model the fixtures use, each serving that model's
    schema and access policies. `llm` (a test double) is shared by all of them; None lets each
    build the configured Gemini model the way the service does.
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
        verdict=judge(fixture, actual),
        validation_retries=result.validation_retries,
        latency_seconds=time.perf_counter() - started,
    )


async def run_fixture(
    fixture: Fixture,
    agent: IntentAgent,
    *,
    timeout_retries: int = DEFAULT_TIMEOUT_RETRIES,
) -> FixtureResult:
    """Extract one fixture's prompt and judge the response. An exception out of the pipeline is
    recorded against the fixture instead of aborting the run - one provider hiccup must not
    hide the other results. A response that is only the pipeline's timeout error is rerun up to
    `timeout_retries` times (DEFAULT_TIMEOUT_RETRIES), and the reruns spent are reported on the
    result; the timeouts themselves stay visible in the log.
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
    """Run every fixture, at most `options.concurrency` at a time, in one event loop (the LLM
    client's HTTP session is bound to the loop it first runs on). Results come back in fixture
    order whatever the completion order; `on_result` is called as each one completes, for
    progress.
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
