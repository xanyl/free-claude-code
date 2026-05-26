"""Google AI Studio Gemini provider (OpenAI-compatible chat completions)."""

from __future__ import annotations

import json
from typing import Any

import openai
from loguru import logger

from providers.base import ProviderConfig
from providers.defaults import GEMINI_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body, clone_body_without_thinking


class GeminiProvider(OpenAIChatTransport):
    """Gemini API using ``https://generativelanguage.googleapis.com/v1beta/openai/``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="GEMINI",
            base_url=config.base_url or GEMINI_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        status_code = getattr(error, "status_code", None)
        if not isinstance(error, openai.BadRequestError) and status_code != 400:
            return None
        retry_body = clone_body_without_thinking(body)
        if retry_body is None:
            return None
        error_text = str(error)
        error_body = getattr(error, "body", None)
        if error_body is not None:
            error_text = f"{error_text} {json.dumps(error_body, default=str)}"
        logger.warning(
            "GEMINI_STREAM: retrying without thinking fields after 400 error: {}",
            error_text,
        )
        return retry_body
