import pytest

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.openai_compatible import OpenAICompatibleProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_openai_provider_parses_embeddings(monkeypatch):
    async def fake_post(self, url, headers=None, json=None):
        assert url == "http://litellm:4000/v1/embeddings"
        assert json["model"] == "qwen3-embedding-8b"
        assert json["encoding_format"] == "float"
        return DummyResponse(
            {
                "model": "qwen3-embedding-8b",
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ],
            }
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    provider = OpenAICompatibleProvider(
        base_url="http://litellm:4000",
        model_name="qwen3-embedding-8b",
    )
    embeddings = await provider.embed_documents(["a", "b"])
    assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert await provider.get_vector_size() == 3


@pytest.mark.asyncio
async def test_openai_provider_rejects_model_mismatch(monkeypatch):
    async def fake_post(self, url, headers=None, json=None):
        return DummyResponse(
            {
                "model": "unexpected-model",
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
            }
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    provider = OpenAICompatibleProvider(
        base_url="http://litellm:4000",
        model_name="qwen3-embedding-8b",
    )
    with pytest.raises(ValueError, match="model mismatch"):
        await provider.embed_query("hello")


def test_openai_provider_settings(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://litellm:4000")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding-8b")
    settings = EmbeddingProviderSettings()
    assert settings.provider_type == EmbeddingProviderType.OPENAI
    assert settings.base_url == "http://litellm:4000"
    assert settings.model_name == "qwen3-embedding-8b"


def test_embedding_factory_creates_openai_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://litellm:4000")
    settings = EmbeddingProviderSettings()
    provider = create_embedding_provider(settings)
    assert isinstance(provider, OpenAICompatibleProvider)
