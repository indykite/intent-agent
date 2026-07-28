from indykite.intent_agent.agent import _discover_skills, build_app
from indykite.intent_agent.config import Settings

_MIN_EXPECTED_ROUTES = 2


def test_discovers_extract_intent_skill():
    skills = _discover_skills()
    names = {s["name"] for s in skills}
    assert "extract-intent" in names


def test_build_app_registers_routes(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("INDYKITE_BASE_URL", raising=False)
    monkeypatch.delenv("IK_APP_AGENT_KEY", raising=False)
    settings = Settings(_env_file=None)

    app, agent = build_app(settings)

    assert agent.settings.gemini_api_key == "test-key"
    assert agent.get_cached_schema() is None
    assert len(app.routes) >= _MIN_EXPECTED_ROUTES
