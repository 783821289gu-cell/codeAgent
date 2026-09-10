from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from repopilot.core.models import (
    ExtractedMemory,
    MemoryItem,
    MemoryKind,
    MemoryPolarity,
)
from repopilot.infrastructure.postgres import MemoryStore
from repopilot.repository.embeddings import EmbeddingProvider


@dataclass(frozen=True, slots=True)
class MemoryWriteOutcome:
    action: str
    item: MemoryItem
    matched_memory_id: int | None = None


class MemoryLifecycle:
    """Hybrid memory recall plus semantic deduplication and conflict replacement."""

    def __init__(
        self,
        store: MemoryStore,
        embeddings: EmbeddingProvider,
        *,
        duplicate_threshold: float = 0.94,
        topic_threshold: float = 0.80,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.duplicate_threshold = duplicate_threshold
        self.topic_threshold = topic_threshold

    async def remember_many(
        self, scope: str, candidates: list[ExtractedMemory]
    ) -> list[MemoryWriteOutcome]:
        outcomes: list[MemoryWriteOutcome] = []
        for candidate in candidates:
            outcomes.append(await self.remember(scope, candidate))
        return outcomes

    async def retrieve(self, scope: str, query: str, limit: int = 8) -> list[MemoryItem]:
        """Fuse keyword and embedding ranks without falling back to unrelated recency."""

        if limit < 1 or not query.strip():
            return []
        active_items = await self._ensure_embeddings(self.store.list_active(scope))
        if not active_items:
            return []

        candidate_limit = max(limit * 2, limit)
        keyword_items = self.store.keyword_search(scope, query, candidate_limit)
        query_vector = (await self.embeddings.embed([query]))[0]
        semantic_items = self.store.semantic_search(scope, query_vector, candidate_limit)

        by_id = {item.id: item for item in active_items if item.id is not None}
        fused_scores: dict[int, float] = {}
        for item in active_items:
            if item.id is not None and item.kind in {
                MemoryKind.USER_PREFERENCE,
                MemoryKind.PROJECT_CONSTRAINT,
            }:
                fused_scores[item.id] = 0.0025 + item.importance * 0.001
        for rank, item in enumerate(keyword_items, start=1):
            if item.id is not None:
                fused_scores[item.id] = fused_scores.get(item.id, 0.0) + 1 / (60 + rank)
        for rank, item in enumerate(semantic_items, start=1):
            if item.id is not None:
                fused_scores[item.id] = fused_scores.get(item.id, 0.0) + 1.25 / (60 + rank)

        ranked_ids = sorted(
            fused_scores,
            key=lambda item_id: (
                fused_scores[item_id],
                by_id[item_id].importance,
                by_id[item_id].updated_at,
            ),
            reverse=True,
        )
        return [by_id[item_id] for item_id in ranked_ids[:limit]]

    async def remember(self, scope: str, candidate: ExtractedMemory) -> MemoryWriteOutcome:
        topic_vector, content_vector = await self.embeddings.embed(
            [candidate.topic, candidate.content]
        )
        existing_items = self.store.list_active(scope, candidate.kind.value)
        existing_items = await self._ensure_embeddings(existing_items)
        matched = self._best_match(topic_vector, content_vector, existing_items)
        new_item = MemoryItem(
            scope=scope,
            kind=candidate.kind,
            topic=candidate.topic,
            content=candidate.content,
            polarity=candidate.polarity,
            importance=candidate.importance,
            embedding=content_vector,
            topic_embedding=topic_vector,
        )
        if matched is None:
            stored = self.store.save(new_item)
            return MemoryWriteOutcome("created", stored)

        matched_item, topic_score, content_score = matched
        if _opposes(matched_item.polarity, candidate.polarity):
            stored = self.store.save(
                new_item.model_copy(
                    update={
                        "supersedes_id": matched_item.id,
                        "conflict_with_id": matched_item.id,
                    }
                )
            )
            if matched_item.id is not None:
                self.store.mark_superseded(matched_item.id, stored.id)
            return MemoryWriteOutcome("conflict_replaced", stored, matched_item.id)

        if content_score >= self.duplicate_threshold:
            stored = self.store.save(
                matched_item.model_copy(
                    update={
                        "importance": max(matched_item.importance, candidate.importance),
                        "topic": candidate.topic if topic_score > 0.99 else matched_item.topic,
                    }
                )
            )
            return MemoryWriteOutcome("semantic_duplicate", stored, matched_item.id)

        stored = self.store.save(
            new_item.model_copy(
                update={
                    "id": matched_item.id,
                    "created_at": matched_item.created_at,
                    "supersedes_id": matched_item.supersedes_id,
                    "conflict_with_id": matched_item.conflict_with_id,
                }
            )
        )
        return MemoryWriteOutcome("updated", stored, matched_item.id)

    async def _ensure_embeddings(self, items: list[MemoryItem]) -> list[MemoryItem]:
        hydrated: list[MemoryItem] = []
        for item in items:
            if item.embedding and item.topic_embedding:
                hydrated.append(item)
                continue
            topic_vector, content_vector = await self.embeddings.embed(
                [item.topic or item.content, item.content]
            )
            hydrated.append(
                self.store.save(
                    item.model_copy(
                        update={
                            "embedding": content_vector,
                            "topic_embedding": topic_vector,
                        }
                    )
                )
            )
        return hydrated

    def _best_match(
        self,
        topic_vector: list[float],
        content_vector: list[float],
        existing_items: list[MemoryItem],
    ) -> tuple[MemoryItem, float, float] | None:
        ranked: list[tuple[float, float, MemoryItem]] = []
        for item in existing_items:
            topic_score = _cosine(topic_vector, item.topic_embedding)
            content_score = _cosine(content_vector, item.embedding)
            if topic_score >= self.topic_threshold or content_score >= self.duplicate_threshold:
                ranked.append((topic_score, content_score, item))
        if not ranked:
            return None
        topic_score, content_score, item = max(
            ranked,
            key=lambda value: (value[0], value[1], value[2].updated_at),
        )
        return item, topic_score, content_score


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    denominator = sqrt(sum(value * value for value in left)) * sqrt(
        sum(value * value for value in right)
    )
    if not denominator:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def _opposes(left: MemoryPolarity, right: MemoryPolarity) -> bool:
    return frozenset((left, right)) in {
        frozenset((MemoryPolarity.REQUIRE, MemoryPolarity.FORBID)),
        frozenset((MemoryPolarity.PREFER, MemoryPolarity.AVOID)),
    }
