"""The LLM provider seam: what differs per provider - building the chat model, and binding a
request's tools and structured-output schema to it. A new provider is one `_PROVIDERS` entry.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI

from indykite.intent_agent.config import LlmProvider, Settings

type _BoundModel = Runnable[Any, BaseMessage]


@dataclass(frozen=True)
class _Provider:
    """How one provider's chat model is built and bound."""

    build: Callable[[Settings], BaseChatModel]
    bind: Callable[[BaseChatModel, Sequence[StructuredTool], dict[str, Any]], _BoundModel]


def _build_gemini(settings: Settings) -> BaseChatModel:
    return ChatGoogleGenerativeAI(model=settings.gemini_model, api_key=settings.gemini_api_key, temperature=0)


def _bind_gemini(llm: BaseChatModel, tools: Sequence[StructuredTool], response_schema: dict[str, Any]) -> _BoundModel:
    return llm.bind_tools(tools, response_mime_type="application/json", response_schema=response_schema)


_PROVIDERS: dict[LlmProvider, _Provider] = {
    "gemini": _Provider(build=_build_gemini, bind=_bind_gemini),
}


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Build the chat model of the configured LLM_PROVIDER."""
    return _PROVIDERS[settings.llm_provider].build(settings)


def bind_structured_output(
    provider: LlmProvider, llm: BaseChatModel, tools: Sequence[StructuredTool], response_schema: dict[str, Any]
) -> _BoundModel:
    """Bind the tools and the JSON schema the final answer must follow, the way `provider` expects them."""
    return _PROVIDERS[provider].bind(llm, tools, response_schema)
