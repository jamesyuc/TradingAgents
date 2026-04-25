import os
from typing import Any, Optional

import httpx
from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


# AI Builders gateway 500s on the 'x-stainless-raw-response' header that
# the openai SDK injects via `with_raw_response`. The SDK, however, also
# relies on this header being present on `response.request.headers` to
# decide how to parse the response. So we must strip it *only on the wire*,
# not from the in-memory Request object. A custom transport does exactly
# that: copy the request, drop the header, send the copy.
class _StripStainlessRawTransport(httpx.HTTPTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if "x-stainless-raw-response" in request.headers:
            sanitized = httpx.Request(
                method=request.method,
                url=request.url,
                headers=[(k, v) for k, v in request.headers.raw
                         if k.lower() != b"x-stainless-raw-response"],
                content=request.content,
                extensions=request.extensions,
            )
            return super().handle_request(sanitized)
        return super().handle_request(request)


class _AsyncStripStainlessRawTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if "x-stainless-raw-response" in request.headers:
            sanitized = httpx.Request(
                method=request.method,
                url=request.url,
                headers=[(k, v) for k, v in request.headers.raw
                         if k.lower() != b"x-stainless-raw-response"],
                content=request.content,
                extensions=request.extensions,
            )
            return await super().handle_async_request(sanitized)
        return await super().handle_async_request(request)


def _make_aibuilders_http_client() -> httpx.Client:
    return httpx.Client(
        transport=_StripStainlessRawTransport(),
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


def _make_aibuilders_async_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=_AsyncStripStainlessRawTransport(),
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        """Wrap with structured output, defaulting to function_calling for OpenAI.

        langchain-openai's Responses-API-parse path (the default for json_schema
        when use_responses_api=True) calls response.model_dump(...) on the OpenAI
        SDK's union-typed parsed response, which makes Pydantic emit ~20
        PydanticSerializationUnexpectedValue warnings per call. The function-calling
        path returns a plain tool-call shape that does not trigger that
        serialization, so it is the cleaner choice for our combination of
        use_responses_api=True + with_structured_output. Both paths use OpenAI's
        strict mode and produce the same typed Pydantic instance.
        """
        if method is None:
            method = "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm": ("https://api.z.ai/api/paas/v4/", "ZHIPU_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "aibuilders": ("https://space.ai-builders.com/backend/v1", "AI_BUILDER_TOKEN"),
}


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        # Provider-specific base URL and auth
        if self.provider in _PROVIDER_CONFIG:
            base_url, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = base_url
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        # AI Builders gateway 500s on the 'x-stainless-raw-response' header
        # that langchain_openai injects via with_raw_response. Strip it.
        if self.provider == "aibuilders":
            llm_kwargs.setdefault("http_client", _make_aibuilders_http_client())
            llm_kwargs.setdefault("http_async_client", _make_aibuilders_async_http_client())

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
