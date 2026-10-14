"""The eval runner (evals/runner.py) over the real IntentAgent pipeline with a scripted LLM:
subject binding from the fixture, one agent per data model, judged results in fixture order,
per-fixture exception isolation and the concurrency bound.
"""

import asyncio
import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import ToolMessage

from evals.fixtures import ADVERSARIAL_SUBJECT_SPOOF, OUT_OF_SCOPE
from evals.runner import FixtureSchemaClient, RunOptions, build_agents, run_fixture, run_suite
from evals.scoring import OUTCOME_EXCEPTION
from indykite.intent_agent.audit.message import Outcome
from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import no_intent_error

_ACTIVATE_EXTRACT_INTENT = {"name": "activate_skill", "args": {"name": "extract-intent"}, "id": "call-1"}
_PROMPT = "Can I access the document doc123?"
# What the model answers (no subject - the pipeline binds the caller) and what the caller gets
# (the caller bound, and the `parameters` object the pipeline gives every evaluation).
_MODEL_EVALUATION = {"action": {"name": "CAN_ACCESS"}, "resource": {"type": "Document", "id": "doc123"}, "context": {}}
_BOUND_EVALUATION = {
    **_MODEL_EVALUATION,
    "context": {"parameters": {}},
    "subject": {"type": "User", "id": "leslie"},
}


class _ScriptedLLM:
    """A stand-in for ChatGoogleGenerativeAI: activates extract-intent on the first call of every
    conversation, then answers each prompt from `script` (a dict -> its JSON, an exception -> raised,
    a string -> free text without ever activating the skill). Tracks how many conversations are in
    flight at once.
    """

    def __init__(self, script: dict[str, object], *, delay: float = 0.0) -> None:
        self.script = script
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.prompts_seen: list[str] = []

    def bind_tools(self, *_args: object, **_kwargs: object) -> "_ScriptedLLM":
        return self

    async def ainvoke(self, messages: list) -> Mock:
        prompt = messages[1].content
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        answer = self.script[prompt]
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, str):
            return Mock(text=answer, content=answer, tool_calls=[])
        if not any(isinstance(message, ToolMessage) for message in messages):
            return Mock(text="", content="", tool_calls=[_ACTIVATE_EXTRACT_INTENT])
        self.prompts_seen.append(prompt)
        # The model rates every answer (an empty one too - the pipeline ignores it there); a
        # script entry that doesn't rate itself is rated high.
        text = json.dumps({"explanation": "A policy matches the prompt.", "confidence": "high", **answer})
        return Mock(text=text, content=text, tool_calls=[])


@pytest.fixture
def settings(baseline_env):
    baseline_env()
    return Settings(_env_file=None)


def test_fixture_schema_client_serves_the_fixture():
    client = FixtureSchemaClient({"graph": {}})

    client.preload()

    assert client.get_cached() == {"graph": {}}


_CUSTOMER_EXPECTED = {
    "prompt": "Can I access the document doc123?",
    "evaluations": [
        {
            "subject": {"type": "Customer", "id": "leslie"},
            "action": {"name": "CAN_ACCESS"},
            "resource": {"type": "Document", "id": "doc123"},
            "context": {"parameters": {}},
        }
    ],
}
_CUSTOMER_CAN_ACCESS_DOCUMENT = ("Customer", "CAN_ACCESS", "Document")


def _customer_model(eval_schema_file, eval_policies_file, *extra_policies):
    """A customer_model granting Customer CAN_ACCESS Document (plus `extra_policies` triples)."""
    eval_policies_file("customer_model", [_CUSTOMER_CAN_ACCESS_DOCUMENT, *extra_policies])
    return eval_schema_file("customer_model", digital_twins=("Customer",), relations=[_CUSTOMER_CAN_ACCESS_DOCUMENT])


def test_one_agent_per_data_model(settings, eval_fixture, eval_schema_file, eval_policies_file):
    _customer_model(eval_schema_file, eval_policies_file)
    fixtures = [
        eval_fixture("a_test_1"),
        eval_fixture("b_test_1", data_model="customer_model", expected=_CUSTOMER_EXPECTED),
        eval_fixture("c_test_1"),
    ]

    agents = build_agents(fixtures, settings, llm=_ScriptedLLM({}))

    assert sorted(path.name for path in agents) == ["customer_model_schema.json", "user_model_schema.json"]
    assert "Customer" in agents[fixtures[1].schema_path].get_cached_schema()["graph"]["nodes"]
    assert "Customer" not in agents[fixtures[0].schema_path].get_cached_schema()["graph"]["nodes"]


def test_run_fixture_binds_the_fixture_subject_and_judges_the_bound_response(settings, eval_fixture):
    fixture = eval_fixture()
    llm = _ScriptedLLM({_PROMPT: {"evaluations": [_MODEL_EVALUATION]}})
    agent = build_agents([fixture], settings, llm=llm)[fixture.schema_path]

    result = asyncio.run(run_fixture(fixture, agent))

    assert result.passed
    assert result.actual == {"prompt": _PROMPT, "evaluations": [_BOUND_EVALUATION]}
    assert result.outcome == Outcome.SUCCESS.value
    assert result.validation_retries == 0
    assert result.latency_seconds > 0
    assert (result.confidence, result.explanation) == ("high", "A policy matches the prompt.")


def test_run_fixture_passes_the_right_evaluations_rated_with_another_confidence(settings, eval_fixture):
    fixture = eval_fixture(confidence="high")
    answer = {"evaluations": [_MODEL_EVALUATION], "explanation": "Only a related policy fits.", "confidence": "low"}
    llm = _ScriptedLLM({_PROMPT: answer})
    agent = build_agents([fixture], settings, llm=llm)[fixture.schema_path]

    result = asyncio.run(run_fixture(fixture, agent))

    assert result.passed
    assert result.actual == {"prompt": _PROMPT, "evaluations": [_BOUND_EVALUATION]}
    assert result.verdict.detail == "response matches the golden answer (confidence low, golden high)"
    assert (result.confidence, result.explanation) == ("low", "Only a related policy fits.")


def test_run_fixture_reports_a_forced_match_on_a_refusal_fixture(settings, eval_fixture):
    fixture = eval_fixture("out_of_scope_test_1", category=OUT_OF_SCOPE, expected=no_intent_error(_PROMPT))
    llm = _ScriptedLLM({_PROMPT: {"evaluations": [_MODEL_EVALUATION]}})
    agent = build_agents([fixture], settings, llm=llm)[fixture.schema_path]

    result = asyncio.run(run_fixture(fixture, agent))

    assert not result.passed
    assert result.verdict.forced_match
    assert result.outcome == Outcome.SUCCESS.value


def test_run_fixture_passes_a_refusal_fixture_on_the_models_empty_evaluations(settings, eval_fixture):
    """The golden refusal is errors.no_intent_error's exact text, produced by the pipeline itself."""
    fixture = eval_fixture("out_of_scope_test_1", category=OUT_OF_SCOPE, expected=no_intent_error(_PROMPT))
    llm = _ScriptedLLM({_PROMPT: {"evaluations": []}})
    agent = build_agents([fixture], settings, llm=llm)[fixture.schema_path]

    result = asyncio.run(run_fixture(fixture, agent))

    assert result.passed
    assert result.outcome == Outcome.UNMATCHED.value


def test_run_fixture_free_text_without_activation_fails_as_unrecognised(settings, eval_fixture):
    fixture = eval_fixture()
    llm = _ScriptedLLM({_PROMPT: "I would rather chat."})
    agent = build_agents([fixture], settings, llm=llm)[fixture.schema_path]

    result = asyncio.run(run_fixture(fixture, agent))

    assert not result.passed
    assert result.actual == {"text": "I would rather chat."}
    assert result.outcome == Outcome.UNMATCHED.value
    assert result.verdict.detail == "expected evaluations, got an unrecognised response"


def test_run_fixture_records_an_exception_instead_of_raising(settings, eval_fixture, caplog):
    fixture = eval_fixture()
    llm = _ScriptedLLM({_PROMPT: RuntimeError("429 quota exceeded")})
    agent = build_agents([fixture], settings, llm=llm)[fixture.schema_path]

    result = asyncio.run(run_fixture(fixture, agent))

    assert result.outcome == OUTCOME_EXCEPTION
    assert result.verdict.detail == "extraction raised RuntimeError: 429 quota exceeded"
    assert "Fixture single_intent_test_1: extraction raised" in caplog.text


def test_run_suite_returns_results_in_fixture_order_and_reports_progress(settings, eval_fixture):
    other_prompt = "Who can access the document doc123?"
    fixtures = [
        eval_fixture("single_intent_test_1"),
        eval_fixture(
            "single_intent_test_2",
            prompt=other_prompt,
            expected={"prompt": other_prompt, "evaluations": [_BOUND_EVALUATION]},
        ),
        eval_fixture("adversarial_subject_spoof_test_1", category=ADVERSARIAL_SUBJECT_SPOOF),
    ]
    llm = _ScriptedLLM(
        {
            _PROMPT: {"evaluations": [_MODEL_EVALUATION]},
            other_prompt: {"evaluations": [{**_MODEL_EVALUATION, "resource": {"type": "Document"}}]},
        },
        delay=0.01,
    )
    seen = []

    results = asyncio.run(
        run_suite(fixtures, settings, llm=llm, options=RunOptions(concurrency=3), on_result=seen.append)
    )

    assert [result.fixture.id for result in results] == [fixture.id for fixture in fixtures]
    assert [result.passed for result in results] == [True, False, True]
    assert results[1].verdict.detail == "evaluations differ from the golden answer"
    assert sorted(result.fixture.id for result in seen) == sorted(fixture.id for fixture in fixtures)


def test_run_suite_bounds_concurrency(settings, eval_fixture):
    fixtures = [eval_fixture(f"single_intent_test_{i}") for i in range(6)]
    llm = _ScriptedLLM({_PROMPT: {"evaluations": [_MODEL_EVALUATION]}}, delay=0.02)

    results = asyncio.run(run_suite(fixtures, settings, llm=llm, options=RunOptions(concurrency=2)))

    assert all(result.passed for result in results)
    assert llm.max_in_flight == 2  # noqa: PLR2004 - the bound under test


def test_run_suite_defaults_to_one_at_a_time(settings, eval_fixture):
    fixtures = [eval_fixture(f"single_intent_test_{i}") for i in range(3)]
    llm = _ScriptedLLM({_PROMPT: {"evaluations": [_MODEL_EVALUATION]}}, delay=0.01)

    asyncio.run(run_suite(fixtures, settings, llm=llm))

    assert llm.max_in_flight == 1


def test_run_suite_rejects_a_concurrency_below_one(settings, eval_fixture):
    with pytest.raises(ValueError, match="concurrency must be at least 1"):
        asyncio.run(run_suite([eval_fixture()], settings, llm=_ScriptedLLM({}), options=RunOptions(concurrency=0)))


def test_run_suite_isolates_a_failing_fixture(settings, eval_fixture):
    other_prompt = "Who can access the document doc123?"
    fixtures = [eval_fixture("single_intent_test_1"), eval_fixture("single_intent_test_2", prompt=other_prompt)]
    llm = _ScriptedLLM({_PROMPT: {"evaluations": [_MODEL_EVALUATION]}, other_prompt: ConnectionError("boom")})

    results = asyncio.run(run_suite(fixtures, settings, llm=llm))

    assert results[0].passed
    assert results[1].outcome == OUTCOME_EXCEPTION


def test_agents_serve_their_data_models_policies(settings, eval_fixture, eval_schema_file, eval_policies_file):
    _customer_model(eval_schema_file, eval_policies_file, ("Customer", "CHECK_MEMBERSHIP", "Document"))
    fixtures = [
        eval_fixture("a_test_1"),
        eval_fixture("b_test_1", data_model="customer_model", expected=_CUSTOMER_EXPECTED),
    ]

    agents = build_agents(fixtures, settings, llm=_ScriptedLLM({}))

    user_policies, failed = agents[fixtures[0].schema_path].get_cached_policies("User")
    assert not failed
    assert [entry["policy"]["actions"] for entry in user_policies] == [["CAN_ACCESS"]]
    customer_policies, failed = agents[fixtures[1].schema_path].get_cached_policies("Customer")
    assert not failed
    assert [entry["policy"]["actions"] for entry in customer_policies] == [["CAN_ACCESS"], ["CHECK_MEMBERSHIP"]]
    assert agents[fixtures[1].schema_path].get_cached_policies("User") == ([], False)


def test_run_fixture_reruns_once_when_the_only_outcome_was_a_timeout(settings, eval_fixture, monkeypatch, caplog):
    """A per-call timeout is a latency spike, not a judgement of the answer: the fixture is
    rerun once, the rerun's answer is what gets judged, and the rerun is reported.
    """
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0.05")
    from indykite.intent_agent.config import Settings as _Settings

    slow_then_fast = _ScriptedLLM({_PROMPT: {"evaluations": [_MODEL_EVALUATION]}}, delay=0.2)
    fixture = eval_fixture()
    agent = build_agents([fixture], _Settings(_env_file=None), llm=slow_then_fast)[fixture.schema_path]
    calls = {"n": 0}
    original = slow_then_fast.ainvoke

    async def _ainvoke(messages):
        calls["n"] += 1
        slow_then_fast.delay = 0.2 if calls["n"] == 1 else 0.0
        return await original(messages)

    slow_then_fast.ainvoke = _ainvoke

    result = asyncio.run(run_fixture(fixture, agent))

    assert result.passed
    assert result.timeout_retries == 1
    assert "timed out" in caplog.text


def test_run_fixture_keeps_the_timeout_when_reruns_are_exhausted(settings, eval_fixture, monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0.05")
    from indykite.intent_agent.config import Settings as _Settings

    always_slow = _ScriptedLLM({_PROMPT: {"evaluations": [_MODEL_EVALUATION]}}, delay=0.2)
    fixture = eval_fixture()
    agent = build_agents([fixture], _Settings(_env_file=None), llm=always_slow)[fixture.schema_path]

    result = asyncio.run(run_fixture(fixture, agent, timeout_retries=1))
    none = asyncio.run(run_fixture(fixture, agent, timeout_retries=0))

    assert not result.passed
    assert result.timeout_retries == 1
    assert result.actual["reason"] == "The model did not respond in time."
    assert none.timeout_retries == 0
