from __future__ import annotations

import os
from collections.abc import Sequence
from uuid import uuid4

import pytest

from repopilot.core.models import RetrievalResult
from repopilot.infrastructure.postgres import PostgresDatabase
from repopilot.repository.vector_store import PgVectorStore


class FakeEmbeddings:
    dimensions = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.casefold()
            vectors.append(
                [
                    float(lowered.count("login") + lowered.count("auth")),
                    float(lowered.count("user")),
                    float(lowered.count("error") + lowered.count("exception")),
                    float(lowered.count("test")),
                ]
            )
        return vectors


class FakeReranker:
    async def rerank(
        self, query: str, documents: Sequence[RetrievalResult]
    ) -> list[RetrievalResult]:
        return list(documents)


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch):
    database_url = (
        os.getenv("REVIEW_AGENT_DATABASE_URL")
        or os.getenv("REPOPILOT_DATABASE_URL")
        or os.getenv("DATABASE_URL")
    )
    if not database_url:
        pytest.skip("PostgreSQL is required; set DATABASE_URL")
    schema = f"repopilot_test_{uuid4().hex[:10]}"
    monkeypatch.setenv("REPOPILOT_PGVECTOR_SCHEMA", schema)
    database = PostgresDatabase(database_url, dimensions=4, schema=schema)
    try:
        yield database
    finally:
        with database.engine.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        database.dispose()


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def fake_reranker():
    return FakeReranker()


@pytest.fixture
def vector_store(database):
    return PgVectorStore(
        database.database_url,
        dimensions=4,
        schema=database.schema,
        engine=database.engine,
    )
