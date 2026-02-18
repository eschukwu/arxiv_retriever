"""Google Gemini API provider."""

import os
from typing import Optional

from arxiv_retriever.summary_util.config import LLMConfig
from arxiv_retriever.summary_util.exceptions import LLMProviderError, MissingAPIKeyError


class GeminiProvider:
    """Google Gemini API provider."""

    ENV_VAR = "GEMINI_API_KEY"

    def __init__(self, config: LLMConfig):
        self._model_name = config.model_name
        self._api_key = config.api_key or os.getenv(self.ENV_VAR)
        self._temperature = config.temperature

        if not self._api_key:
            raise MissingAPIKeyError("Google Gemini", self.ENV_VAR)

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Get response from Gemini API."""
        try:
            from google import genai
        except ImportError:
            raise LLMProviderError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

        client = genai.Client(api_key=self._api_key)

        # Build the content with optional system prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        try:
            response = client.models.generate_content(
                model=self._model_name,
                contents=full_prompt,
                config={
                    "temperature": self._temperature,
                },
            )
            return response.text
        except Exception as e:
            raise LLMProviderError(f"Gemini API error: {e}")
