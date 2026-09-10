from __future__ import annotations

from collections.abc import Sequence

import pytest
from sqlalchemy import select

from repopilot.application.workflow import RepoPilot
from repopilot.core.models import (
    ExtractedMemory,
    MemoryExtractionReport,
    MemoryKind,
    MemoryPolarity,
)
from repopilot.infrastructure.postgres import MemoryStore
from repopilot.knowledge.memory import MemoryLifecycle


class SemanticEmbeddings:
    dimensions = 4

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        if "dependencies" in lowered:
            if "must add" in lowered:
                return [0.8, 0.2, 0.0, 0.0]
            return [1.0, 0.0, 0.0, 0.0]
        if "pytest" in lowered:
            if "strict markers" in lowered:
                return [0.0, 0.6, 0.8, 0.0]
            if lowered == "test framework":
                return [0.0, 1.0, 0.0, 0.0]
            return [0.0, 0.8, 0.2, 0.0]
        return [0.0, 0.0, 0.0, 1.0]


@pytest.mark.asyncio
async def test_memory_semantic_duplicate_update_and_conflict(database) -> None:
    store = MemoryStore(database)
    lifecycle = MemoryLifecycle(store, SemanticEmbeddings())

    created = await lifecycle.remember(
        "repo",
        ExtractedMemory(
            kind=MemoryKind.PROJECT_CONSTRAINT,
            topic="third-party dependencies",
            content="Do not add new third-party dependencies",
            polarity=MemoryPolarity.FORBID,
        ),
    )
    duplicate = await lifecycle.remember(
        "repo",
        ExtractedMemory(
            kind=MemoryKind.PROJECT_CONSTRAINT,
            topic="third-party dependencies",
            content="Never add new third-party dependencies",
            polarity=MemoryPolarity.FORBID,
        ),
    )
    convention = await lifecycle.remember(
        "repo",
        ExtractedMemory(
            kind=MemoryKind.CODING_CONVENTION,
            topic="test framework",
            content="Use pytest for repository tests",
        ),
    )
    updated = await lifecycle.remember(
        "repo",
        ExtractedMemory(
            kind=MemoryKind.CODING_CONVENTION,
            topic="test framework",
            content="Use pytest with strict markers for repository tests",
        ),
    )
    conflict = await lifecycle.remember(
        "repo",
        ExtractedMemory(
            kind=MemoryKind.PROJECT_CONSTRAINT,
            topic="third-party dependencies",
            content="New tasks must add third-party dependencies",
            polarity=MemoryPolarity.REQUIRE,
        ),
    )

    assert created.action == "created"
    assert duplicate.action == "semantic_duplicate"
    assert duplicate.item.id == created.item.id
    assert convention.action == "created"
    assert updated.action == "updated"
    assert updated.item.id == convention.item.id
    assert conflict.action == "conflict_replaced"
    assert conflict.item.supersedes_id == created.item.id
    active = store.list_active("repo")
    assert {item.content for item in active} == {
        "Use pytest with strict markers for repository tests",
        "New tasks must add third-party dependencies",
    }
    with database.engine.connect() as connection:
        old_status = (
            connection.execute(
                select(
                    database.memories.c.status,
                    database.memories.c.conflict_with_id,
                ).where(database.memories.c.id == created.item.id)
            )
            .mappings()
            .one()
        )
    assert old_status["status"] == "superseded"
    assert old_status["conflict_with_id"] == conflict.item.id


@pytest.mark.asyncio
async def test_memory_recall_fuses_keyword_and_vector_results(database) -> None:
    store = MemoryStore(database)
    lifecycle = MemoryLifecycle(store, SemanticEmbeddings())
    await lifecycle.remember(
        "repo",
        ExtractedMemory(
            kind=MemoryKind.CODING_CONVENTION,
            topic="test framework",
            content="Use pytest with strict markers for repository tests",
        ),
    )
    await lifecycle.remember(
        "repo",
        ExtractedMemory(
            kind=MemoryKind.PROJECT_CONSTRAINT,
            topic="third-party dependencies",
            content="Do not add new third-party dependencies",
        ),
    )

    assert store.keyword_search("repo", "test framework") == []
    recalled = await lifecycle.retrieve("repo", "test framework", limit=1)

    assert [item.content for item in recalled] == [
        "Use pytest with strict markers for repository tests"
    ]


def test_memory_extraction_enforces_provenance_failure_gate_and_limit() -> None:
    report = MemoryExtractionReport(
        items=[
            ExtractedMemory(
                kind=MemoryKind.USER_PREFERENCE,
                topic="type hints",
                content="Always use type hints",
                source="user_task",
                evidence="always use type hints",
                importance=1.0,
            ),
            ExtractedMemory(
                kind=MemoryKind.USER_PREFERENCE,
                topic="review workflow",
                content="必须经过Reviewer",
                source="user_task",
                evidence="always use type hints",
                importance=0.99,
            ),
            ExtractedMemory(
                kind=MemoryKind.REUSABLE_EXPERIENCE,
                topic="unverified fix",
                content="Changing the timeout solves the issue",
                source="validated_result",
                evidence="task summary",
                importance=0.98,
            ),
            ExtractedMemory(
                kind=MemoryKind.PROJECT_CONSTRAINT,
                topic="Python version",
                content="The project requires Python 3.12 or newer",
                source="project_evidence",
                evidence="pyproject.toml requires-python",
                importance=0.9,
            ),
            ExtractedMemory(
                kind=MemoryKind.CODING_CONVENTION,
                topic="tests",
                content="Repository tests use pytest",
                source="project_evidence",
                evidence="pyproject.toml pytest configuration",
                importance=0.8,
            ),
            ExtractedMemory(
                kind=MemoryKind.ARCHITECTURE_DECISION,
                topic="storage",
                content="Memory metadata is stored in PostgreSQL",
                source="project_evidence",
                evidence="src/repopilot/infrastructure/postgres.py",
                importance=0.7,
            ),
        ]
    )

    filtered = RepoPilot._filter_extracted_memories(
        "Please always use type hints", report, completed=False
    )

    assert len(filtered.items) == 3
    assert {item.topic for item in filtered.items} == {
        "type hints",
        "Python version",
        "tests",
    }
    assert all(item.kind != MemoryKind.REUSABLE_EXPERIENCE for item in filtered.items)
