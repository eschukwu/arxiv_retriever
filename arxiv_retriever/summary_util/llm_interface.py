"""LLM interface with multi-provider support."""

from typing import Optional, Union

from dotenv import load_dotenv

from arxiv_retriever.summary_util.config import (
    LLMConfig,
    get_default_model,
    parse_model_string,
)
from arxiv_retriever.summary_util.exceptions import ProviderNotSupportedError
from arxiv_retriever.summary_util.providers.claude import ClaudeProvider
from arxiv_retriever.summary_util.providers.gemini import GeminiProvider
from arxiv_retriever.summary_util.providers.ollama import OllamaProvider

# Load environment variables
load_dotenv()

# Type alias for any provider
LLMProviderType = Union[OllamaProvider, GeminiProvider, ClaudeProvider]

# Provider registry
PROVIDERS = {
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "claude": ClaudeProvider,
}

# System prompt for paper summarization
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in summarizing scientific papers and "
    "extracting the most meaningful parts of the paper as simply and concisely as possible."
)


def get_llm_client(model: Optional[str] = None) -> LLMProviderType:
    """
    Create an LLM client for the specified model.

    Args:
        model: Model string in format "provider:model_name"
               (e.g., "ollama:llama3", "claude:claude-3-haiku")
               If None, uses default from config or env var.

    Returns:
        An LLM provider instance

    Raises:
        ProviderNotSupportedError: If provider is not supported
        InvalidModelFormatError: If model string format is invalid
    """
    config = parse_model_string(model or get_default_model())

    provider_class = PROVIDERS.get(config.provider)
    if not provider_class:
        raise ProviderNotSupportedError(config.provider, list(PROVIDERS.keys()))

    return provider_class(config)


def get_llm_response(prompt: str, model: Optional[str] = None) -> str:
    """
    Get an LLM response using the specified model.

    This is a convenience function that creates a client and gets a response
    in one call. For multiple requests, prefer using get_llm_client() directly.

    Args:
        prompt: The prompt to send
        model: Optional model string (provider:model_name)

    Returns:
        The LLM response text
    """
    client = get_llm_client(model)
    return client.get_response(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT)
