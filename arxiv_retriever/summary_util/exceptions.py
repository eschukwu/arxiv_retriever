"""Custom exceptions for LLM provider operations."""


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    pass


class MissingAPIKeyError(LLMProviderError):
    """Raised when required API key is not configured."""

    def __init__(self, provider: str, env_var: str):
        self.provider = provider
        self.env_var = env_var
        super().__init__(
            f"API key for {provider} not found. "
            f"Please set the {env_var} environment variable."
        )


class InvalidModelFormatError(LLMProviderError):
    """Raised when the model string format is invalid."""

    def __init__(self, model_str: str):
        self.model_str = model_str
        super().__init__(
            f"Invalid model format: '{model_str}'. "
            "Expected format: 'provider:model_name' "
            "(e.g., 'ollama:llama3', 'claude:claude-3-haiku', 'gemini:gemini-1.5-flash')"
        )


class ProviderNotSupportedError(LLMProviderError):
    """Raised when an unsupported provider is requested."""

    def __init__(self, provider: str, supported: list):
        self.provider = provider
        self.supported = supported
        super().__init__(
            f"Provider '{provider}' not supported. "
            f"Supported providers: {', '.join(supported)}"
        )


class ModelNotAvailableError(LLMProviderError):
    """Raised when the requested model is not available."""

    def __init__(self, model_name: str, details: str = ""):
        self.model_name = model_name
        message = f"Model '{model_name}' is not available."
        if details:
            message += f" {details}"
        super().__init__(message)
