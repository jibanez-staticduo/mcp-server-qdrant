from __future__ import annotations

import logging
from typing import Any

import httpx

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(EmbeddingProvider):
    """Embedding provider for OpenAI-compatible APIs such as LiteLLM."""

    def __init__(
        self,
        *,
        base_url: str | None,
        model_name: str,
        api_key: str | None = None,
        expected_response_model: str | None = None,
        vector_name: str | None = None,
    ):
        if not base_url:
            raise ValueError("base_url is required for OpenAICompatibleProvider")

        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.expected_response_model = expected_response_model or model_name
        self.vector_name = vector_name if vector_name is not None else self._sanitize_vector_name(model_name)
        self._vector_size: int | None = None

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        payload = await self._request_embeddings(documents)
        return self._parse_embeddings(payload, expected_count=len(documents))

    async def embed_query(self, query: str) -> list[float]:
        payload = await self._request_embeddings([query])
        return self._parse_embeddings(payload, expected_count=1)[0]

    def get_vector_name(self) -> str | None:
        return self.vector_name

    async def get_vector_size(self) -> int:
        if self._vector_size is None:
            probe_vector = (await self.embed_documents(["dimension probe"]))[0]
            self._vector_size = len(probe_vector)
        return self._vector_size

    async def _request_embeddings(self, texts: list[str]) -> dict[str, Any]:
        endpoint = f"{self.base_url}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "input": texts,
            "model": self.model_name,
            "encoding_format": "float",
        }
        logger.info(
            "embedding_request provider=openai endpoint=%s model=%s text_count=%s",
            endpoint,
            self.model_name,
            len(texts),
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            return response.json()

    def _parse_embeddings(
        self, payload: dict[str, Any], *, expected_count: int
    ) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected_count:
            raise ValueError("Embedding response does not contain the expected data items")

        response_model = payload.get("model")
        if response_model is not None and response_model != self.expected_response_model:
            raise ValueError(
                f"Embedding response model mismatch: expected {self.expected_response_model}, got {response_model}"
            )

        embeddings: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Embedding response contains a non-object data entry")
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("Embedding response contains an invalid embedding")
            if not all(isinstance(value, (int, float)) for value in embedding):
                raise ValueError("Embedding response contains non-numeric embedding values")
            embeddings.append([float(value) for value in embedding])

        first_dim = len(embeddings[0])
        if not all(len(embedding) == first_dim for embedding in embeddings):
            raise ValueError("Embedding response contains inconsistent vector dimensions")

        if self._vector_size is None:
            self._vector_size = first_dim
        elif self._vector_size != first_dim:
            raise ValueError(
                f"Embedding dimension changed from {self._vector_size} to {first_dim}"
            )

        return embeddings

    @staticmethod
    def _sanitize_vector_name(model_name: str) -> str:
        lowered = model_name.lower()
        safe = [ch if ch.isalnum() else "-" for ch in lowered]
        collapsed = "".join(safe).strip("-")
        while "--" in collapsed:
            collapsed = collapsed.replace("--", "-")
        return collapsed or "openai-embedding"
