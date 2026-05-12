"""OpenAI-compatible LLM client wrappers."""

import logging
import random
import time

from openai import APIError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from core.config import settings

logger = logging.getLogger(__name__)


def call_with_retries(
    func,
    *args,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    **kwargs,
):
    """Call an LLM provider function with exponential backoff for transient errors."""
    for attempt in range(max_retries):
        try:
            logger.debug("Calling LLM function. attempt=%s max_retries=%s", attempt + 1, max_retries)
            return func(*args, **kwargs)
        except (APITimeoutError, RateLimitError, APIError, InternalServerError) as exc:
            status = getattr(exc, "status", None)
            if isinstance(exc, APIError) and status is not None and status < 500:
                raise

            if attempt == max_retries - 1:
                raise

            delay = min(base_delay * (2**attempt), max_delay)
            if jitter:
                delay *= 0.5 + random.random() / 2.0

            logger.warning(
                "LLM transient error. error_type=%s delay=%.1fs attempt=%s/%s error=%s",
                type(exc).__name__,
                delay,
                attempt + 1,
                max_retries,
                exc,
            )
            time.sleep(delay)


class OpenAIClient:
    """OpenAI-compatible chat client."""

    def __init__(self, api_key=None, base_url=None, model_name=None):
        """Create an OpenAI-compatible client from explicit values or settings."""
        logger.debug(
            "Initializing OpenAIClient with model_name=%s base_url=%s",
            model_name or settings.model_name,
            base_url or settings.base_url,
        )
        self.client = OpenAI(
            api_key=api_key or settings.require_api_key(),
            base_url=base_url or settings.base_url,
        )
        self.model_name = model_name or settings.model_name

    def create_chat_completion(
        self,
        messages,
        model_name: str | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: bool = True,
        **kwargs,
    ):
        """Create a chat completion using the configured OpenAI-compatible provider."""
        return call_with_retries(
            self.client.chat.completions.create,
            model=model_name or self.model_name,
            messages=messages,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
            **kwargs,
        )
