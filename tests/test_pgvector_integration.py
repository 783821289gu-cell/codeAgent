from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from repopilot.repository.vector_store import IndexedChunk, IndexedFile, PgVectorStore


@pytest.mark.skipif(
    os.getenv("REPOPILOT_RUN_PGVECTOR_TESTS") != "1",
    reason="set REPOPILOT_RUN_PGVECTOR_TESTS=1 for the portable PostgreSQL integration test",
)
def test_pgvector_hnsw_cosine_and_postgres_fts() -> None:
    database_url = os.environ["REVIEW_AGENT_DATABASE_URL"]
    schema = f"repopilot_test_{uuid4().hex[:8]}"
    cleanup_engine = create_engine(database_url)
    try:
        store = PgVectorStore(database_url, 4, schema=schema)
        store.sync_files(
            "repository",
            [
                IndexedFile(
                    path="auth.py",
                    content_hash="a" * 64,
                    chunks=(
                        IndexedChunk(
                            path="auth.py",
                            start_line=1,
                            end_line=3,
                            content="password recovery token authentication",
                            content_hash="1" * 64,
                            embedding=[1.0, 0.0, 0.0, 0.0],
                        ),
                    ),
                ),
                IndexedFile(
                    path="math.py",
                    content_hash="b" * 64,
                    chunks=(
                        IndexedChunk(
                            path="math.py",
                            start_line=1,
                            end_line=2,
                            content="add two integer values",
                            content_hash="2" * 64,
                            embedding=[0.0, 1.0, 0.0, 0.0],
                        ),
                    ),
                ),
            ],
            [],
        )

        vector_hits = store.vector_search("repository", [1.0, 0.0, 0.0, 0.0], 2)
        keyword_hits = store.keyword_search("repository", "password recovery", 2)
        with store.engine.connect() as connection:
            index_definition = connection.exec_driver_sql(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND indexname = %s",
                (schema, f"idx_{schema}_repo_hnsw"),
            ).scalar_one()

        assert vector_hits[0].chunk.path == "auth.py"
        assert keyword_hits[0].chunk.path == "auth.py"
        assert "USING hnsw" in index_definition
        assert "vector_cosine_ops" in index_definition

        store.sync_files(
            "repository",
            [
                IndexedFile(
                    path="auth.py",
                    content_hash="c" * 64,
                    chunks=(
                        IndexedChunk(
                            path="auth.py",
                            start_line=1,
                            end_line=1,
                            content="updated password recovery",
                            content_hash="3" * 64,
                            embedding=[1.0, 0.0, 0.0, 0.0],
                        ),
                    ),
                )
            ],
            ["math.py"],
        )
        assert store.file_hashes("repository") == {"auth.py": "c" * 64}
        assert store.count("repository") == 1
    finally:
        assert schema.startswith("repopilot_test_")
        with cleanup_engine.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        cleanup_engine.dispose()
