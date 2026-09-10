from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from math import isfinite, sqrt
from pathlib import Path
from threading import Lock
from typing import Protocol

from openai import AsyncOpenAI

BGE_EMBEDDING_DIMENSIONS = 1024


class EmbeddingProvider(Protocol):
    dimensions: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    """Thin adapter over the official embeddings API."""

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self._api_key = api_key
        self.client = client

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self.client or AsyncOpenAI(api_key=self._api_key)
        response = await client.embeddings.create(
            model=self.model,
            input=list(texts),
            dimensions=self.dimensions,
        )
        return [item.embedding for item in sorted(response.data, key=lambda value: value.index)]


class BgeEmbeddingProvider:
    """Local normalized BGE-M3 embeddings, adapted from reviewAgent."""

    dimensions = BGE_EMBEDDING_DIMENSIONS

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        cache_dir: Path,
        device: str = "cpu",
        max_length: int = 1024,
        batch_size: int = 8,
        model_factory: Callable[..., object] | None = None,
    ) -> None:
        self.model = model
        self.revision = revision
        self.cache_dir = cache_dir
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model_factory = model_factory or _load_sentence_transformer
        self._model: object | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed, list(texts))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        with self._inference_lock:
            raw_vectors = model.encode(  # type: ignore[attr-defined]
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        vectors = [_normalized_vector(vector) for vector in raw_vectors]
        if len(vectors) != len(texts):
            raise ValueError("BGE embedding result count must match input count")
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError(f"BGE embedding dimension must be {self.dimensions}")
        return vectors

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = self._model_factory(
                    self.model,
                    revision=self.revision,
                    cache_folder=str(self.cache_dir),
                    device=self.device,
                    max_length=self.max_length,
                )
        return self._model


def _load_sentence_transformer(
    model_name: str,
    *,
    revision: str,
    cache_folder: str,
    device: str,
    max_length: int,
) -> object:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        revision=revision,
        cache_folder=cache_folder,
        device=device,
        trust_remote_code=False,
    )
    model.max_seq_length = max_length
    return model


def _normalized_vector(raw_vector: object) -> list[float]:
    values = [float(value) for value in raw_vector]  # type: ignore[union-attr]
    if not values or not all(isfinite(value) for value in values):
        raise ValueError("BGE embedding returned an invalid vector")
    length = sqrt(sum(value * value for value in values))
    if not length:
        raise ValueError("BGE embedding returned a zero vector")
    return [value / length for value in values]
