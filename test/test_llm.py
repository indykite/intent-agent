from langchain_google_genai import ChatGoogleGenerativeAI

from indykite.intent_agent import llm
from indykite.intent_agent.config import Settings


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
