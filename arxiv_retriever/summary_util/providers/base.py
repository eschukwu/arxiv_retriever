"""Base protocol for LLM providers."""

from typing import Optional, Protocol


class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers."""

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Get a response from the LLM.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt for context

        Returns:
            The LLM's response text

        Raises:
            LLMProviderError: If the request fails
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        ...
