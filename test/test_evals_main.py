"""The `python -m evals` entrypoint: arguments, fixture selection, report outputs and exit status."""

import json
import logging
from unittest.mock import AsyncMock

import pytest

from evals.__main__ import EXIT_GATE_FAILED, EXIT_USAGE, main
from evals.fixtures import ADVERSARIAL_CATEGORIES
from evals.scoring import FixtureResult, Verdict


def _results_for(fixtures, *, failing=(), latency=0.0):
    return [
        FixtureResult(
            fixture=fixture,
            actual=fixture.expected if fixture.id not in failing else {},
            outcome="success",
            verdict=Verdict(passed=fixture.id not in failing, forced_match=False, detail="scripted"),
            latency_seconds=latency,
        )
        for fixture in fixtures
    ]


@pytest.fixture
def suite(tmp_path, eval_fixture_file, baseline_env, monkeypatch, caplog):
    """A passing fixture directory and the Settings env, run from tmp_path so no developer `.env` leaks in."""
    monkeypatch.chdir(tmp_path)
    # main() quietens the `indykite` logger; caplog restores its level so later tests' log assertions still work.
    caplog.set_level(logging.INFO, logger="evals")
    caplog.set_level(logging.getLogger("indykite").level, logger="indykite")
    baseline_env()
    eval_fixture_file("single_intent_test_1")
    eval_fixture_file("single_intent_test_2")
    for category in ADVERSARIAL_CATEGORIES:  # the injection gate requires every adversarial category
        eval_fixture_file(f"{category.replace(' ', '_')}_test_1", category=category)
    return tmp_path / "evals"


@pytest.fixture
def run_suite(mocker):
    async def _run(fixtures, settings, *, options, on_result):
        results = _results_for(fixtures, failing=run_suite.failing, latency=run_suite.latency)
        for result in results:
            on_result(result)
        return results

    run_suite = mocker.patch("evals.__main__.run_suite", new=AsyncMock(side_effect=_run))
    run_suite.failing = ()
    run_suite.latency = 0.0
    return run_suite


def test_green_run_exits_zero_and_prints_the_report(suite, run_suite, capsys, caplog):
    status = main(["--fixtures", str(suite)])

    assert status == 0
    out = capsys.readouterr().out
    assert out.startswith("Golden eval suite - model gemini-3.5-flash-lite - 5/5 fixtures passed")
    assert out.endswith("Result: PASS\n")
    assert "Running 5 fixture(s) against gemini-3.5-flash-lite (concurrency 1)" in caplog.text
    assert "PASS single_intent_test_1 [single intent] outcome=success" in caplog.text
    _fixtures, settings = run_suite.await_args.args
    assert settings.gemini_model == "gemini-3.5-flash-lite"
    assert run_suite.await_args.kwargs["options"].concurrency == 1
    assert run_suite.await_args.kwargs["options"].timeout_retries == 1


def test_gate_failure_exits_one(suite, run_suite, capsys, caplog):
    run_suite.failing = ("single_intent_test_1",)

    status = main(["--fixtures", str(suite)])

    assert status == EXIT_GATE_FAILED
    assert "[FAIL] correctness: 80.0% (4/5)" in capsys.readouterr().out
    assert "FAIL single_intent_test_1 [single intent] outcome=success" in caplog.text


def test_latency_failure_exits_one(suite, run_suite, capsys):
    run_suite.latency = 30.0

    assert main(["--fixtures", str(suite)]) == EXIT_GATE_FAILED
    assert "[FAIL] p95 latency: p95 30.00s" in capsys.readouterr().out


def test_reports_are_written_where_asked_creating_parent_dirs(suite, run_suite, tmp_path):
    markdown = tmp_path / "out" / "report.md"
    json_path = tmp_path / "out" / "nested" / "report.json"

    status = main(["--fixtures", str(suite), "--markdown", str(markdown), "--json", str(json_path)])

    assert status == 0
    assert markdown.read_text(encoding="utf-8").startswith("## Golden eval suite: ✅ PASS")
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["passed"] is True
    assert [entry["id"] for entry in document["fixtures"]] == [
        "adversarial_oversized_input_test_1",
        "adversarial_sql_injection_test_1",
        "adversarial_subject_spoof_test_1",
        "single_intent_test_1",
        "single_intent_test_2",
    ]


def test_only_filters_the_fixtures_run(suite, run_suite):
    concurrency = 4

    status = main(["--fixtures", str(suite), "--only", "single_intent_*", "--concurrency", str(concurrency)])

    fixtures, _settings = run_suite.await_args.args
    assert [fixture.id for fixture in fixtures] == ["single_intent_test_1", "single_intent_test_2"]
    assert run_suite.await_args.kwargs["options"].concurrency == concurrency
    assert status == EXIT_GATE_FAILED  # no adversarial fixture selected, so the injection gate can't pass


def test_only_subset_with_an_adversarial_fixture_passes_without_the_other_categories(suite, run_suite, capsys):
    status = main(["--fixtures", str(suite), "--only", "adversarial_subject_*"])

    assert status == 0
    assert "[PASS] injection resistance: 100.0% (1/1)" in capsys.readouterr().out


def test_suite_runs_under_its_own_timeouts_not_the_services(suite, run_suite, monkeypatch):
    """Generous suite timeouts (overridable by flag) replace the env ones, so slowness lands on the latency gate."""
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "10")

    assert main(["--fixtures", str(suite)]) == 0
    settings = run_suite.await_args.args[1]
    assert (settings.llm_timeout_seconds, settings.request_timeout_seconds) == (20.0, 60.0)

    assert main(["--fixtures", str(suite), "--llm-timeout", "5", "--request-timeout", "15"]) == 0
    settings = run_suite.await_args.args[1]
    assert (settings.llm_timeout_seconds, settings.request_timeout_seconds) == (5.0, 15.0)


def test_min_correctness_override_reaches_the_gate(suite, run_suite, capsys):
    run_suite.failing = ("single_intent_test_1",)

    status = main(["--fixtures", str(suite), "--min-correctness", "0.5"])

    assert status == 0
    assert "[PASS] correctness: 80.0% (4/5) (required: >= 50.0% of fixtures match)" in capsys.readouterr().out


def test_unmatched_only_pattern_is_a_usage_error(suite, run_suite, caplog):
    assert main(["--fixtures", str(suite), "--only", "nothing_*"]) == EXIT_USAGE
    assert "no fixture id matches 'nothing_*'" in caplog.text
    run_suite.assert_not_awaited()


def test_malformed_fixture_is_a_usage_error(suite, run_suite, eval_fixture_file, caplog):
    eval_fixture_file("broken_test_1", category="nope")

    assert main(["--fixtures", str(suite)]) == EXIT_USAGE
    assert "broken_test_1.json: unknown category 'nope'" in caplog.text
    run_suite.assert_not_awaited()


def test_missing_settings_are_a_usage_error(suite, run_suite, monkeypatch, caplog):
    monkeypatch.delenv("GEMINI_API_KEY")

    assert main(["--fixtures", str(suite)]) == EXIT_USAGE
    assert "gemini_api_key" in caplog.text
    run_suite.assert_not_awaited()


@pytest.mark.parametrize(
    "argv",
    [
        ["--concurrency", "0"],
        ["--min-correctness", "1.5"],
        ["--min-correctness", "-0.1"],
        ["--max-p95-latency", "0"],
        ["--max-p95-latency", "nan"],
        ["--max-p95-latency", "inf"],  # would make `p95 <= limit` true even for a timed-out sample
        ["--llm-timeout", "0"],
        ["--llm-timeout", "inf"],
        ["--request-timeout", "nan"],
        ["--llm-timeout", "30", "--request-timeout", "20"],
    ],
)
def test_out_of_range_arguments_are_rejected_by_argparse(suite, run_suite, argv):
    with pytest.raises(SystemExit) as exc_info:
        main(["--fixtures", str(suite), *argv])

    assert exc_info.value.code == EXIT_USAGE
    run_suite.assert_not_awaited()


def test_malformed_schema_file_is_a_usage_error(suite, run_suite, caplog):
    from evals.fixtures import load_schema

    (suite / "schema" / "user_model_schema.json").write_text("{", encoding="utf-8")
    load_schema.cache_clear()

    assert main(["--fixtures", str(suite)]) == EXIT_USAGE
    assert "user_model_schema.json: data model schema is not valid JSON" in caplog.text
    run_suite.assert_not_awaited()
