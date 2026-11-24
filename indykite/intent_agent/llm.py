"""The LLM provider seam: what differs per provider - building the chat model, binding a
request's tools and structured-output schema to it, and which of its errors are timeouts.
A new provider is one `_PROVIDERS` entry.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import APITimeoutError, OpenAIError

from indykite.intent_agent.config import OPENAI_RESPONSE_FORMAT_NAME, LlmProvider, Settings
from indykite.intent_agent.errors import LlmProviderError

type _BoundModel = Runnable[Any, BaseMessage]


@dataclass(frozen=True)
class _Provider:
    """How one provider's chat model is built and bound, and the errors its client raises for a timeout."""

    build: Callable[[Settings], BaseChatModel]
    bind: Callable[[BaseChatModel, Sequence[StructuredTool], dict[str, Any]], _BoundModel]
    timeout_errors: tuple[type[Exception], ...] = ()
    # Errors whose message must not leave the service: OpenAI's quotes part of the API key.
    redacted_errors: tuple[type[Exception], ...] = ()


def _build_gemini(settings: Settings) -> BaseChatModel:
    return ChatGoogleGenerativeAI(model=settings.gemini_model, api_key=settings.gemini_api_key, temperature=0)


def _bind_gemini(llm: BaseChatModel, tools: Sequence[StructuredTool], response_schema: dict[str, Any]) -> _BoundModel:
    return llm.bind_tools(tools, response_mime_type="application/json", response_schema=response_schema)


def _build_openai(settings: Settings) -> BaseChatModel:
    # No temperature: OpenAI's reasoning models reject any value but their default.
    # No client timeout either: LLM_TIMEOUT_SECONDS bounds each call, as one TimeoutError.
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=str(settings.openai_base_url) if settings.openai_base_url else None,
    )


def _bind_openai(llm: BaseChatModel, tools: Sequence[StructuredTool], response_schema: dict[str, Any]) -> _BoundModel:
    # Not strict: strict mode wants every field required and no free-form object, which
    # context.parameters and context.qualifiers are. The answer is validated afterwards either way.
    response_format = {"name": OPENAI_RESPONSE_FORMAT_NAME, "schema": response_schema, "strict": False}
    return llm.bind_tools(tools, response_format=response_format)


_PROVIDERS: dict[LlmProvider, _Provider] = {
    "gemini": _Provider(build=_build_gemini, bind=_bind_gemini),
    "openai": _Provider(
        build=_build_openai, bind=_bind_openai, timeout_errors=(APITimeoutError,), redacted_errors=(OpenAIError,)
    ),
}


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Build the chat model of the configured LLM_PROVIDER."""
    return _PROVIDERS[settings.llm_provider].build(settings)


def bind_structured_output(
    provider: LlmProvider, llm: BaseChatModel, tools: Sequence[StructuredTool], response_schema: dict[str, Any]
) -> _BoundModel:
    """Bind the tools and the JSON schema the final answer must follow, the way `provider` expects them."""
    return _PROVIDERS[provider].bind(llm, tools, response_schema)


def is_timeout(provider: LlmProvider, exc: BaseException) -> bool:
    """Whether `exc` is `provider`'s own client giving up on a call that took too long."""
    return isinstance(exc, _PROVIDERS[provider].timeout_errors)


def without_provider_text(provider: LlmProvider, exc: Exception) -> Exception:
    """`exc` itself, or an LlmProviderError in its place when `provider`'s error text can't be shown to a caller."""
    if not isinstance(exc, _PROVIDERS[provider].redacted_errors):
        return exc
    status = getattr(exc, "status_code", None)
    detail = f"{type(exc).__name__} (status {status})" if isinstance(status, int) else type(exc).__name__
    return LlmProviderError(f"{provider} call failed: {detail}")
