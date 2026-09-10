from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine, RowMapping


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    path: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class IndexedFile:
    path: str
    content_hash: str
    chunks: tuple[IndexedChunk, ...]


@dataclass(frozen=True, slots=True)
class StoredChunk:
    id: int
    path: str
    start_line: int
    end_line: int
    content: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    chunk: StoredChunk
    distance: float


@dataclass(frozen=True, slots=True)
class KeywordSearchHit:
    chunk: StoredChunk
    score: float


class VectorStore(Protocol):
    dimensions: int

    def count(self, scope: str) -> int: ...

    def file_hashes(self, scope: str) -> dict[str, str]: ...

    def sync_files(
        self, scope: str, files: list[IndexedFile], deleted_paths: list[str]
    ) -> None: ...

    def vector_search(
        self, scope: str, query_embedding: list[float], limit: int
    ) -> list[VectorSearchHit]: ...

    def keyword_search(self, scope: str, query: str, limit: int) -> list[KeywordSearchHit]: ...


class PgVectorStore:
    """Shared PostgreSQL store using pgvector cosine HNSW and PostgreSQL FTS."""

    def __init__(
        self,
        database_url: str,
        dimensions: int,
        *,
        schema: str = "repopilot",
        engine: Engine | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
            raise ValueError("pgvector schema must be a simple SQL identifier")
        self.dimensions = dimensions
        self.schema = schema
        self.engine = engine or create_engine(database_url, pool_pre_ping=True)
        metadata = MetaData(schema=schema)
        self.chunks = Table(
            "repository_chunks",
            metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=True),
            Column("scope", Text, nullable=False),
            Column("path", Text, nullable=False),
            Column("start_line", Integer, nullable=False),
            Column("end_line", Integer, nullable=False),
            Column("content", Text, nullable=False),
            Column("content_hash", Text, nullable=False),
            Column("embedding", Vector(dimensions), nullable=False),
        )
        self.files = Table(
            "repository_files",
            metadata,
            Column("scope", Text, primary_key=True),
            Column("path", Text, primary_key=True),
            Column("content_hash", Text, nullable=False),
        )
        Index(f"idx_{schema}_repo_scope", self.chunks.c.scope, self.chunks.c.path)
        Index(
            f"idx_{schema}_repo_hnsw",
            self.chunks.c.embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        )
        with self.engine.begin() as connection:
            connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
            connection.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.exec_driver_sql(
                f'CREATE INDEX IF NOT EXISTS "idx_{schema}_repo_fts" '
                f'ON "{schema}"."repository_chunks" '
                "USING gin (to_tsvector('simple', content))"
            )
        self._validate_dimension()

    def _validate_dimension(self) -> None:
        statement = """
            SELECT format_type(attribute.atttypid, attribute.atttypmod) AS vector_type
            FROM pg_attribute AS attribute
            JOIN pg_class AS relation ON relation.oid = attribute.attrelid
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = :schema
              AND relation.relname = 'repository_chunks'
              AND attribute.attname = 'embedding'
        """
        with self.engine.connect() as connection:
            vector_type = connection.execute(text(statement), {"schema": self.schema}).scalar_one()
        if vector_type != f"vector({self.dimensions})":
            raise ValueError(
                f"pgvector table dimension is {vector_type}; configured {self.dimensions}"
            )

    def count(self, scope: str) -> int:
        statement = (
            select(func.count()).select_from(self.chunks).where(self.chunks.c.scope == scope)
        )
        with self.engine.connect() as connection:
            return int(connection.execute(statement).scalar_one())

    def file_hashes(self, scope: str) -> dict[str, str]:
        statement = select(self.files.c.path, self.files.c.content_hash).where(
            self.files.c.scope == scope
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement).all()
        return {str(path): str(content_hash) for path, content_hash in rows}

    def sync_files(self, scope: str, files: list[IndexedFile], deleted_paths: list[str]) -> None:
        target_paths = list(dict.fromkeys([*deleted_paths, *(file.path for file in files)]))
        with self.engine.begin() as connection:
            if target_paths:
                connection.execute(
                    delete(self.chunks).where(
                        self.chunks.c.scope == scope,
                        self.chunks.c.path.in_(target_paths),
                    )
                )
                connection.execute(
                    delete(self.files).where(
                        self.files.c.scope == scope,
                        self.files.c.path.in_(target_paths),
                    )
                )
            for file in files:
                connection.execute(
                    insert(self.files),
                    {
                        "scope": scope,
                        "path": file.path,
                        "content_hash": file.content_hash,
                    },
                )
                if file.chunks:
                    connection.execute(
                        insert(self.chunks),
                        [
                            {
                                "scope": scope,
                                "path": chunk.path,
                                "start_line": chunk.start_line,
                                "end_line": chunk.end_line,
                                "content": chunk.content,
                                "content_hash": chunk.content_hash,
                                "embedding": chunk.embedding,
                            }
                            for chunk in file.chunks
                        ],
                    )

    def vector_search(
        self, scope: str, query_embedding: list[float], limit: int
    ) -> list[VectorSearchHit]:
        distance = self.chunks.c.embedding.cosine_distance(query_embedding).label("distance")
        statement = (
            select(self.chunks, distance)
            .where(self.chunks.c.scope == scope)
            .order_by(distance, self.chunks.c.path, self.chunks.c.start_line)
            .limit(limit)
        )
        with self.engine.begin() as connection:
            connection.exec_driver_sql("SET LOCAL hnsw.iterative_scan = strict_order")
            connection.exec_driver_sql("SET LOCAL hnsw.ef_search = 80")
            rows = connection.execute(statement).mappings().all()
        return [
            VectorSearchHit(chunk=_postgres_chunk(row), distance=float(row["distance"]))
            for row in rows
        ]

    def keyword_search(self, scope: str, query: str, limit: int) -> list[KeywordSearchHit]:
        tokens = _query_tokens(query)
        if not tokens:
            return []
        tsquery = " | ".join(token.replace("'", "") for token in tokens)
        document_vector = func.to_tsvector("simple", self.chunks.c.content)
        query_vector = func.to_tsquery("simple", tsquery)
        keyword_score = func.ts_rank_cd(document_vector, query_vector).label("keyword_score")
        statement = (
            select(self.chunks, keyword_score)
            .where(self.chunks.c.scope == scope, document_vector.op("@@")(query_vector))
            .order_by(keyword_score.desc(), self.chunks.c.path, self.chunks.c.start_line)
            .limit(limit)
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [
            KeywordSearchHit(chunk=_postgres_chunk(row), score=float(row["keyword_score"]))
            for row in rows
        ]


def _query_tokens(query: str) -> list[str]:
    return re.findall(r"[\w-]+", query, flags=re.UNICODE)[:16]


def _postgres_chunk(row: RowMapping) -> StoredChunk:
    return StoredChunk(
        id=int(row["id"]),
        path=str(row["path"]),
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        content=str(row["content"]),
        content_hash=str(row["content_hash"]),
    )
