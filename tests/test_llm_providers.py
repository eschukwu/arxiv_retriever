"""Tests for LLM provider implementations."""

import pytest

from arxiv_retriever.summary_util.config import LLMConfig, parse_model_string
from arxiv_retriever.summary_util.exceptions import (
    InvalidModelFormatError,
    LLMProviderError,
    MissingAPIKeyError,
    ProviderNotSupportedError,
)
from arxiv_retriever.summary_util.llm_interface import get_llm_client, get_llm_response


class TestParseModelString:
    """Tests for parse_model_string function."""

    def test_parse_valid_ollama_model(self):
        config = parse_model_string("ollama:llama3")
        assert config.provider == "ollama"
        assert config.model_name == "llama3"

    def test_parse_valid_claude_model(self):
        config = parse_model_string("claude:claude-3-haiku-20240307")
        assert config.provider == "claude"
        assert config.model_name == "claude-3-haiku-20240307"

    def test_parse_valid_gemini_model(self):
        config = parse_model_string("gemini:gemini-1.5-flash")
        assert config.provider == "gemini"
        assert config.model_name == "gemini-1.5-flash"

    def test_parse_provider_case_insensitive(self):
        config = parse_model_string("OLLAMA:llama3")
        assert config.provider == "ollama"

    def test_parse_model_with_colons_in_name(self):
        # Model names can contain colons (e.g., some version strings)
        config = parse_model_string("ollama:llama3:latest")
        assert config.provider == "ollama"
        assert config.model_name == "llama3:latest"

    def test_parse_invalid_no_colon(self):
        with pytest.raises(InvalidModelFormatError) as exc_info:
            parse_model_string("ollama-llama3")
        assert "Invalid model format" in str(exc_info.value)

    def test_parse_invalid_empty_provider(self):
        with pytest.raises(InvalidModelFormatError):
            parse_model_string(":llama3")

    def test_parse_invalid_empty_model(self):
        with pytest.raises(InvalidModelFormatError):
            parse_model_string("ollama:")


class TestGetLLMClient:
    """Tests for get_llm_client function."""

    def test_get_ollama_client(self):
        # Ollama doesn't require API key
        client = get_llm_client("ollama:llama3")
        assert client.model_name == "llama3"

    def test_get_claude_client_missing_key(self, mocker):
        mocker.patch.dict("os.environ", {}, clear=True)
        with pytest.raises(MissingAPIKeyError) as exc_info:
            get_llm_client("claude:claude-3-haiku")
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_get_gemini_client_missing_key(self, mocker):
        mocker.patch.dict("os.environ", {}, clear=True)
        with pytest.raises(MissingAPIKeyError) as exc_info:
            get_llm_client("gemini:gemini-1.5-flash")
        assert "GEMINI_API_KEY" in str(exc_info.value)

    def test_unsupported_provider(self):
        with pytest.raises(ProviderNotSupportedError) as exc_info:
            get_llm_client("openai:gpt-4")
        assert "openai" in str(exc_info.value)
        assert "ollama" in str(exc_info.value)

    def test_default_model_used_when_none(self, mocker):
        # Mock environment to ensure default is used
        mocker.patch.dict("os.environ", {"ARXIV_RETRIEVER_DEFAULT_MODEL": "ollama:mistral"})
        client = get_llm_client(None)
        assert client.model_name == "mistral"


class TestOllamaProvider:
    """Tests for Ollama provider."""

    def test_ollama_connection_error(self, mocker):
        import httpx

        # Mock httpx.Client to raise ConnectError
        mock_client = mocker.MagicMock()
        mock_client.__enter__ = mocker.MagicMock(return_value=mock_client)
        mock_client.__exit__ = mocker.MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        mocker.patch("httpx.Client", return_value=mock_client)

        client = get_llm_client("ollama:llama3")
        with pytest.raises(LLMProviderError) as exc_info:
            client.get_response("test prompt")
        assert "Cannot connect to Ollama" in str(exc_info.value)

    def test_ollama_model_not_found(self, mocker):
        import httpx

        # Mock httpx.Client to return 404
        mock_response = mocker.MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=mocker.MagicMock(), response=mock_response
        )

        mock_client = mocker.MagicMock()
        mock_client.__enter__ = mocker.MagicMock(return_value=mock_client)
        mock_client.__exit__ = mocker.MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        mocker.patch("httpx.Client", return_value=mock_client)

        client = get_llm_client("ollama:nonexistent-model")
        with pytest.raises(LLMProviderError) as exc_info:
            client.get_response("test prompt")
        assert "nonexistent-model" in str(exc_info.value)

    def test_ollama_successful_response(self, mocker):
        # Mock successful response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = mocker.MagicMock()
        mock_response.json.return_value = {"response": "This is a test response"}

        mock_client = mocker.MagicMock()
        mock_client.__enter__ = mocker.MagicMock(return_value=mock_client)
        mock_client.__exit__ = mocker.MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        mocker.patch("httpx.Client", return_value=mock_client)

        client = get_llm_client("ollama:llama3")
        response = client.get_response("test prompt", system_prompt="You are helpful")

        assert response == "This is a test response"
        # Verify the payload included system prompt
        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["system"] == "You are helpful"

    def test_ollama_custom_base_url(self, mocker):
        mocker.patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://custom:11434"})

        from arxiv_retriever.summary_util.providers.ollama import OllamaProvider

        config = LLMConfig(provider="ollama", model_name="llama3")
        provider = OllamaProvider(config)

        assert provider._base_url == "http://custom:11434"


class TestClaudeProvider:
    """Tests for Claude provider."""

    def test_claude_with_api_key(self, mocker):
        mocker.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})

        client = get_llm_client("claude:claude-3-haiku")
        assert client.model_name == "claude-3-haiku"

    def test_claude_successful_response(self, mocker):
        mocker.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})

        # Mock anthropic module
        mock_anthropic = mocker.MagicMock()
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.content = [mocker.MagicMock(text="Claude response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic})

        client = get_llm_client("claude:claude-3-haiku")
        response = client.get_response("test prompt", system_prompt="Be helpful")

        assert response == "Claude response"


class TestGeminiProvider:
    """Tests for Gemini provider."""

    def test_gemini_with_api_key(self, mocker):
        mocker.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})

        client = get_llm_client("gemini:gemini-1.5-flash")
        assert client.model_name == "gemini-1.5-flash"

    def test_gemini_successful_response(self, mocker):
        mocker.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})

        # Mock google.genai module (new API)
        mock_genai = mocker.MagicMock()
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.text = "Gemini response"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        # Mock the google module with genai submodule
        mock_google = mocker.MagicMock()
        mock_google.genai = mock_genai
        mocker.patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai})

        client = get_llm_client("gemini:gemini-1.5-flash")
        response = client.get_response("test prompt", system_prompt="Be helpful")

        assert response == "Gemini response"


class TestGetLLMResponse:
    """Tests for the convenience get_llm_response function."""

    def test_get_llm_response_uses_default_system_prompt(self, mocker):
        # Mock successful Ollama response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = mocker.MagicMock()
        mock_response.json.return_value = {"response": "Test response"}

        mock_client = mocker.MagicMock()
        mock_client.__enter__ = mocker.MagicMock(return_value=mock_client)
        mock_client.__exit__ = mocker.MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        mocker.patch("httpx.Client", return_value=mock_client)

        response = get_llm_response("test prompt", model="ollama:llama3")

        assert response == "Test response"
        # Verify default system prompt was used
        call_args = mock_client.post.call_args
        assert "scientific papers" in call_args[1]["json"]["system"]
