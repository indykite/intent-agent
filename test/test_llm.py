import httpx2
import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import APITimeoutError, RateLimitError

from indykite.intent_agent import llm
from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import LlmProviderError

_OPENAI_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
_OPENAI_TIMEOUT = APITimeoutError(request=_OPENAI_REQUEST)
_OPENAI_RATE_LIMIT = RateLimitError("slow down", response=httpx2.Response(429, request=_OPENAI_REQUEST), body=None)


def _openai_settings(baseline_env, monkeypatch, **env):
    baseline_env()
    for name, value in {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "openai-test-key", **env}.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def test_build_chat_model_builds_the_configured_gemini_model(baseline_env, monkeypatch):
    baseline_env()
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    model = llm.build_chat_model(Settings(_env_file=None))

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model.removeprefix("models/") == "gemini-3.1-flash-lite"
    assert model.temperature == 0
    assert model.google_api_key.get_secret_value() == "test-key"


def test_gemini_binding_asks_for_json_following_the_response_schema(mocker):
    chat_model = mocker.Mock()
    tools = [mocker.Mock()]
    response_schema = {"type": "object"}

    bound = llm.bind_structured_output("gemini", chat_model, tools, response_schema)

    assert bound is chat_model.bind_tools.return_value
    chat_model.bind_tools.assert_called_once_with(
        tools, response_mime_type="application/json", response_schema=response_schema
    )


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        pytest.param(None, None, id="openai-itself"),
        pytest.param("https://llm-proxy.example.com/v1", "https://llm-proxy.example.com/v1", id="compatible-endpoint"),
    ],
)
def test_build_chat_model_builds_the_configured_openai_model(baseline_env, monkeypatch, base_url, expected):
    env = {"OPENAI_MODEL": "openai-test-model", **({"OPENAI_BASE_URL": base_url} if base_url else {})}

    model = llm.build_chat_model(_openai_settings(baseline_env, monkeypatch, **env))

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "openai-test-model"
    assert model.openai_api_key.get_secret_value() == "openai-test-key"
    assert model.openai_api_base == expected
    # Unset, so a reasoning model that only takes its default temperature isn't sent one.
    assert model.temperature is None
    assert model.max_retries == 0


def test_openai_binding_asks_for_the_response_schema_without_strict_mode(mocker):
    chat_model = mocker.Mock()
    tools = [mocker.Mock()]
    response_schema = {"type": "object"}

    bound = llm.bind_structured_output("openai", chat_model, tools, response_schema)

    assert bound is chat_model.bind_tools.return_value
    chat_model.bind_tools.assert_called_once_with(
        tools, strict=True, response_format={"name": "intent_extraction", "schema": response_schema, "strict": False}
    )


@pytest.mark.parametrize(
    ("provider", "exc", "expected"),
    [
        pytest.param("openai", _OPENAI_TIMEOUT, True, id="openai-client-timeout"),
        pytest.param("openai", _OPENAI_RATE_LIMIT, False, id="openai-other-error"),
        pytest.param("gemini", _OPENAI_TIMEOUT, False, id="another-providers-timeout"),
    ],
)
def test_is_timeout_recognises_only_the_providers_own_timeout_error(provider, exc, expected):
    assert llm.is_timeout(provider, exc) is expected


def test_without_provider_text_replaces_an_openai_error_by_its_class_and_status():
    safe = llm.without_provider_text("openai", _OPENAI_RATE_LIMIT)

    assert isinstance(safe, LlmProviderError)
    assert str(safe) == "openai call failed: RateLimitError (status 429)"


def test_without_provider_text_names_only_the_class_when_there_was_no_response():
    assert str(llm.without_provider_text("openai", _OPENAI_TIMEOUT)) == "openai call failed: APITimeoutError"


@pytest.mark.parametrize(
    ("provider", "exc"),
    [
        pytest.param("openai", ValueError("not the client's"), id="not-an-openai-error"),
        pytest.param("gemini", _OPENAI_RATE_LIMIT, id="gemini-is-left-as-it-is"),
    ],
)
def test_without_provider_text_returns_any_other_error_unchanged(provider, exc):
    assert llm.without_provider_text(provider, exc) is exc
