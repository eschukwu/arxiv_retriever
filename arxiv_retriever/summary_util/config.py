"""Configuration for LLM providers."""

import os
from dataclasses import dataclass
from typing import Dict, Optional

from arxiv_retriever.summary_util.exceptions import InvalidModelFormatError


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""

    provider: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0


# Default model (can be overridden by env var)
DEFAULT_MODEL = "ollama:llama3"
DEFAULT_MODEL_ENV_VAR = "ARXIV_RETRIEVER_DEFAULT_MODEL"

# Default model names for each provider (used when only provider name is given)
PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "ollama": "llama3",
    "claude": "claude-sonnet-4-6",
    "gemini": "gemini-3-flash-preview",
}


def get_default_model() -> str:
    """Get the default model from environment or fallback."""
    return os.getenv(DEFAULT_MODEL_ENV_VAR, DEFAULT_MODEL)


def parse_model_string(model_str: str) -> LLMConfig:
    """
    Parse a model string into LLMConfig.

    Format: provider:model_name or just provider (uses default model)
    Examples:
        - ollama:llama3.1
        - claude:claude-sonnet-4-6
        - gemini:gemini-3-flash-preview
        - claude          (uses default: claude-sonnet-4-6)
        - gemini          (uses default: gemini-3-flash-preview)

    Args:
        model_str: Model string in provider:model_name format, or just provider name

    Returns:
        LLMConfig with parsed values

    Raises:
        InvalidModelFormatError: If format is invalid
    """
    # Support shorthand: just the provider name (e.g., "claude", "gemini")
    if ":" not in model_str:
        provider = model_str.strip().lower()
        if provider in PROVIDER_DEFAULT_MODELS:
            return LLMConfig(
                provider=provider,
                model_name=PROVIDER_DEFAULT_MODELS[provider],
            )
        raise InvalidModelFormatError(model_str)

    parts = model_str.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise InvalidModelFormatError(model_str)

    provider, model_name = parts
    return LLMConfig(provider=provider.lower(), model_name=model_name)
