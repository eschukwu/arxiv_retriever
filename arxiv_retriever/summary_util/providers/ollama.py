"""Ollama local LLM provider."""

import os
from typing import Optional

import httpx

from arxiv_retriever.summary_util.config import LLMConfig
from arxiv_retriever.summary_util.exceptions import (
    LLMProviderError,
    ModelNotAvailableError,
)


class OllamaProvider:
    """Ollama local LLM provider."""

    DEFAULT_BASE_URL = "http://localhost:11434"
    ENV_VAR_BASE_URL = "OLLAMA_BASE_URL"

    def __init__(self, config: LLMConfig):
        self._model_name = config.model_name
        self._base_url = config.base_url or os.getenv(
            self.ENV_VAR_BASE_URL, self.DEFAULT_BASE_URL
        )
        self._temperature = config.temperature

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Get response from Ollama API."""
        url = f"{self._base_url}/api/generate"

        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._temperature},
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()["response"]
        except httpx.ConnectError:
            raise LLMProviderError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Is Ollama running? Start it with 'ollama serve'."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotAvailableError(
                    self._model_name,
                    f"Run 'ollama pull {self._model_name}' to download it.",
                )
            raise LLMProviderError(f"Ollama API error: {e}")
        except Exception as e:
            raise LLMProviderError(f"Ollama request failed: {e}")
