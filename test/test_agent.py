from indykite.intent_agent.agent import build_app
from indykite.intent_agent.config import Settings
from indykite.intent_agent.skills import discover_skills

_MIN_EXPECTED_ROUTES = 2


def test_discovers_extract_intent_skill():
    """The extract-intent SKILL.md must be discovered from the skills/ directory."""
    skills = discover_skills()
    names = {s["name"] for s in skills}
    assert "extract-intent" in names


def test_build_app_registers_routes(monkeypatch):
    """build_app must wire up the agent-card and JSON-RPC routes on the app."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.delenv("INDYKITE_BASE_URL", raising=False)
    monkeypatch.delenv("IK_APP_AGENT_KEY", raising=False)
    settings = Settings(_env_file=None)

    app, agent = build_app(settings)

    assert agent.settings.gemini_api_key == "test-key"
    assert agent.get_cached_schema() is None
    assert len(app.routes) >= _MIN_EXPECTED_ROUTES
