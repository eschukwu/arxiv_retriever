"""Anthropic Claude API provider."""

import os
from typing import Optional

from arxiv_retriever.summary_util.config import LLMConfig
from arxiv_retriever.summary_util.exceptions import LLMProviderError, MissingAPIKeyError


class ClaudeProvider:
    """Anthropic Claude API provider."""

    ENV_VAR = "ANTHROPIC_API_KEY"

    def __init__(self, config: LLMConfig):
        self._model_name = config.model_name
        self._api_key = config.api_key or os.getenv(self.ENV_VAR)
        self._temperature = config.temperature

        if not self._api_key:
            raise MissingAPIKeyError("Anthropic Claude", self.ENV_VAR)

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Get response from Claude API."""
        try:
            import anthropic
        except ImportError:
            raise LLMProviderError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=self._api_key)

        messages = [{"role": "user", "content": prompt}]

        try:
            kwargs = {
                "model": self._model_name,
                "max_tokens": 1024,
                "messages": messages,
            }

            # Only add temperature if non-zero (Claude default is 1.0)
            if self._temperature > 0:
                kwargs["temperature"] = self._temperature

            if system_prompt:
                kwargs["system"] = system_prompt

            response = client.messages.create(**kwargs)
            return response.content[0].text
        except anthropic.APIError as e:
            raise LLMProviderError(f"Claude API error: {e}")
        except Exception as e:
            raise LLMProviderError(f"Claude request failed: {e}")
