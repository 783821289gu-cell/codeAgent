from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from math import isfinite
from pathlib import Path
from threading import Lock
from typing import Protocol

from repopilot.core.models import RetrievalResult


class Reranker(Protocol):
    async def rerank(
        self, query: str, documents: Sequence[RetrievalResult]
    ) -> list[RetrievalResult]: ...


class _CrossEncoder(Protocol):
    def predict(self, sentences: Sequence[tuple[str, str]], **kwargs: object) -> object: ...


class BgeCrossEncoderReranker:
    """Second-stage BGE Cross-Encoder reranker, adapted from reviewAgent."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        cache_dir: Path,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 8,
        model_factory: Callable[..., _CrossEncoder] | None = None,
    ) -> None:
        self.model = model
        self.revision = revision
        self.cache_dir = cache_dir
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model_factory = model_factory or _load_cross_encoder
        self._model: _CrossEncoder | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    async def rerank(
        self, query: str, documents: Sequence[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Score query/document pairs and return documents in Cross-Encoder order."""

        if not documents:
            return []
        scores = await asyncio.to_thread(
            self._score,
            query,
            [document.content for document in documents],
        )
        rescored = [
            document.model_copy(update={"score": score, "rerank_score": score})
            for document, score in zip(documents, scores, strict=True)
        ]
        return sorted(
            rescored,
            key=lambda document: (
                document.rerank_score if document.rerank_score is not None else float("-inf")
            ),
            reverse=True,
        )

    def warmup(self) -> None:
        self._score("repository code reranker warmup", ["repository code reranker warmup"])

    def _score(self, query: str, documents: list[str]) -> list[float]:
        model = self._get_model()
        pairs = [(query, document) for document in documents]
        with self._inference_lock:
            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        values = raw_scores.reshape(-1).tolist() if hasattr(raw_scores, "reshape") else raw_scores
        scores = [float(value) for value in values]  # type: ignore[union-attr]
        if len(scores) != len(documents) or not all(isfinite(value) for value in scores):
            raise ValueError("BGE reranker returned invalid scores")
        return scores

    def _get_model(self) -> _CrossEncoder:
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


def _load_cross_encoder(
    model_name: str,
    *,
    revision: str,
    cache_folder: str,
    device: str,
    max_length: int,
) -> _CrossEncoder:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name,
        revision=revision,
        cache_folder=cache_folder,
        device=device,
        trust_remote_code=False,
        max_length=max_length,
    )
