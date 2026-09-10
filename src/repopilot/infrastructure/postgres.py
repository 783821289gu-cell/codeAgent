from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime
from typing import Any

from agents.items import TResponseInputItem
from agents.memory.session_settings import SessionSettings, resolve_session_limit
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.engine import Engine, RowMapping, make_url

from repopilot.core.models import MemoryItem, TaskState, TraceEvent, utc_now


class PostgresDatabase:
    """Own the PostgreSQL schema for state, memory, trace, and SDK sessions."""

    def __init__(
        self,
        database_url: str,
        dimensions: int,
        *,
        schema: str = "repopilot",
        engine: Engine | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
            raise ValueError("PostgreSQL schema must be a simple SQL identifier")
        self.database_url = normalize_postgres_url(database_url)
        self.dimensions = dimensions
        self.schema = schema
        self.engine = engine or create_engine(self.database_url, pool_pre_ping=True)
        self.metadata = MetaData(schema=schema)
        self._define_tables()
        self.initialize()

    def _define_tables(self) -> None:
        self.memories = Table(
            "memories",
            self.metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=True),
            Column("scope", Text, nullable=False),
            Column("kind", Text, nullable=False),
            Column("topic", Text, nullable=False, server_default=""),
            Column("content", Text, nullable=False),
            Column("polarity", Text, nullable=False, server_default="fact"),
            Column("status", Text, nullable=False, server_default="active"),
            Column("digest", Text, nullable=False),
            Column("importance", Float, nullable=False),
            Column("supersedes_id", BigInteger),
            Column("conflict_with_id", BigInteger),
            Column("embedding", Vector(self.dimensions)),
            Column("topic_embedding", Vector(self.dimensions)),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("scope", "kind", "digest", name="uq_memories_scope_kind_digest"),
        )
        Index(f"idx_{self.schema}_memory_scope", self.memories.c.scope, self.memories.c.status)
        Index(
            f"idx_{self.schema}_memory_embedding_hnsw",
            self.memories.c.embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        )
        Index(
            f"idx_{self.schema}_memory_topic_hnsw",
            self.memories.c.topic_embedding,
            postgresql_using="hnsw",
            postgresql_ops={"topic_embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        )
        self.checkpoints = Table(
            "checkpoints",
            self.metadata,
            Column("scope", Text, primary_key=True),
            Column("task_id", Text, primary_key=True),
            Column("state_json", JSONB, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )
        self.trace_events = Table(
            "trace_events",
            self.metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=True),
            Column("scope", Text, nullable=False),
            Column("trace_id", Text, nullable=False),
            Column("task_id", Text, nullable=False),
            Column("category", Text, nullable=False),
            Column("name", Text, nullable=False),
            Column("started_at", DateTime(timezone=True), nullable=False),
            Column("duration_ms", Float, nullable=False),
            Column("ok", Boolean, nullable=False),
            Column("details_json", JSONB, nullable=False),
        )
        Index(
            f"idx_{self.schema}_trace_task",
            self.trace_events.c.scope,
            self.trace_events.c.task_id,
            self.trace_events.c.id,
        )
        self.agent_sessions = Table(
            "agent_sessions",
            self.metadata,
            Column("session_id", Text, primary_key=True),
            Column(
                "created_at",
                DateTime(timezone=False),
                nullable=False,
                server_default=text("CURRENT_TIMESTAMP"),
            ),
            Column(
                "updated_at",
                DateTime(timezone=False),
                nullable=False,
                server_default=text("CURRENT_TIMESTAMP"),
            ),
        )
        self.agent_messages = Table(
            "agent_messages",
            self.metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=True),
            Column(
                "session_id",
                Text,
                ForeignKey(f"{self.schema}.agent_sessions.session_id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("message_data", Text, nullable=False),
            Column(
                "created_at",
                DateTime(timezone=False),
                nullable=False,
                server_default=text("CURRENT_TIMESTAMP"),
            ),
        )
        Index(
            f"idx_{self.schema}_agent_messages_session_time",
            self.agent_messages.c.session_id,
            self.agent_messages.c.created_at,
            self.agent_messages.c.id,
        )

    def initialize(self) -> None:
        with self.engine.begin() as connection:
            connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
            connection.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
        self.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.exec_driver_sql(
                f'CREATE INDEX IF NOT EXISTS "idx_{self.schema}_memory_fts" '
                f'ON "{self.schema}"."memories" '
                "USING gin (to_tsvector('simple', content))"
            )
            for legacy, current in (
                ("constraint", "project_constraint"),
                ("preference", "user_preference"),
                ("decision", "architecture_decision"),
                ("experience", "reusable_experience"),
            ):
                connection.execute(
                    update(self.memories).where(self.memories.c.kind == legacy).values(kind=current)
                )
        self._validate_vector_dimensions()

    def _validate_vector_dimensions(self) -> None:
        statement = """
            SELECT attribute.attname,
                   format_type(attribute.atttypid, attribute.atttypmod) AS vector_type
            FROM pg_attribute AS attribute
            JOIN pg_class AS relation ON relation.oid = attribute.attrelid
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = :schema
              AND relation.relname = 'memories'
              AND attribute.attname IN ('embedding', 'topic_embedding')
        """
        with self.engine.connect() as connection:
            rows = connection.execute(text(statement), {"schema": self.schema}).all()
        expected = f"vector({self.dimensions})"
        mismatches = {str(name): str(value) for name, value in rows if value != expected}
        if mismatches:
            raise ValueError(
                f"memory vector dimensions are {mismatches}; configured {self.dimensions}"
            )

    def dispose(self) -> None:
        self.engine.dispose()


def normalize_postgres_url(database_url: str) -> str:
    """Normalize PostgreSQL URLs onto the installed psycopg 3 driver."""

    parsed = make_url(database_url)
    if parsed.drivername == "postgresql":
        parsed = parsed.set(drivername="postgresql+psycopg")
    if parsed.drivername != "postgresql+psycopg":
        raise ValueError("DATABASE_URL must use PostgreSQL with the psycopg driver")
    return parsed.render_as_string(hide_password=False)


class PostgresSession:
    """Agents SDK Session implementation backed by the shared synchronous engine."""

    def __init__(
        self,
        session_id: str,
        database: PostgresDatabase,
        *,
        session_settings: SessionSettings | None = None,
    ) -> None:
        self.session_id = session_id
        self.database = database
        self.session_settings = session_settings or SessionSettings()

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        session_limit = resolve_session_limit(limit, self.session_settings)

        def load() -> list[TResponseInputItem]:
            statement = select(self.database.agent_messages.c.message_data).where(
                self.database.agent_messages.c.session_id == self.session_id
            )
            reverse = session_limit is not None
            if reverse:
                statement = statement.order_by(self.database.agent_messages.c.id.desc()).limit(
                    session_limit
                )
            else:
                statement = statement.order_by(self.database.agent_messages.c.id)
            with self.database.engine.connect() as connection:
                rows = connection.execute(statement).scalars().all()
            if reverse:
                rows.reverse()
            items: list[TResponseInputItem] = []
            for value in rows:
                try:
                    items.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    continue
            return items

        return await asyncio.to_thread(load)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        if not items:
            return

        def add() -> None:
            now = utc_now().replace(tzinfo=None)
            session_insert = postgres_insert(self.database.agent_sessions).values(
                session_id=self.session_id,
                created_at=now,
                updated_at=now,
            )
            session_insert = session_insert.on_conflict_do_update(
                index_elements=[self.database.agent_sessions.c.session_id],
                set_={"updated_at": now},
            )
            payload = [
                {
                    "session_id": self.session_id,
                    "message_data": json.dumps(item, ensure_ascii=False, separators=(",", ":")),
                    "created_at": now,
                }
                for item in items
            ]
            with self.database.engine.begin() as connection:
                connection.execute(session_insert)
                connection.execute(insert(self.database.agent_messages), payload)

        await asyncio.to_thread(add)

    async def pop_item(self) -> TResponseInputItem | None:
        def pop() -> TResponseInputItem | None:
            while True:
                with self.database.engine.begin() as connection:
                    row = connection.execute(
                        select(
                            self.database.agent_messages.c.id,
                            self.database.agent_messages.c.message_data,
                        )
                        .where(self.database.agent_messages.c.session_id == self.session_id)
                        .order_by(self.database.agent_messages.c.id.desc())
                        .limit(1)
                        .with_for_update()
                    ).one_or_none()
                    if row is None:
                        return None
                    connection.execute(
                        self.database.agent_messages.delete().where(
                            self.database.agent_messages.c.id == row.id
                        )
                    )
                try:
                    return json.loads(row.message_data)
                except (json.JSONDecodeError, TypeError):
                    continue

        return await asyncio.to_thread(pop)

    async def clear_session(self) -> None:
        def clear() -> None:
            with self.database.engine.begin() as connection:
                connection.execute(
                    self.database.agent_messages.delete().where(
                        self.database.agent_messages.c.session_id == self.session_id
                    )
                )
                connection.execute(
                    self.database.agent_sessions.delete().where(
                        self.database.agent_sessions.c.session_id == self.session_id
                    )
                )

        await asyncio.to_thread(clear)


def _memory_digest(content: str) -> str:
    normalized = " ".join(content.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class MemoryStore:
    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database
        self.table = database.memories

    def write(self, item: MemoryItem) -> MemoryItem:
        digest = _memory_digest(item.content)
        statement = select(self.table).where(
            self.table.c.scope == item.scope,
            self.table.c.kind == item.kind.value,
            self.table.c.digest == digest,
        )
        with self.database.engine.connect() as connection:
            existing = connection.execute(statement).mappings().one_or_none()
        if existing:
            current = _memory_from_row(existing)
            return self.save(
                item.model_copy(
                    update={
                        "id": current.id,
                        "importance": max(current.importance, item.importance),
                        "embedding": item.embedding or current.embedding,
                        "topic_embedding": item.topic_embedding or current.topic_embedding,
                        "created_at": current.created_at,
                    }
                )
            )
        return self.save(item)

    def save(self, item: MemoryItem) -> MemoryItem:
        now = utc_now()
        values = {
            "scope": item.scope,
            "kind": item.kind.value,
            "topic": item.topic,
            "content": item.content,
            "polarity": item.polarity.value,
            "status": item.status.value,
            "digest": _memory_digest(item.content),
            "importance": item.importance,
            "supersedes_id": item.supersedes_id,
            "conflict_with_id": item.conflict_with_id,
            "embedding": item.embedding or None,
            "topic_embedding": item.topic_embedding or None,
            "updated_at": now,
        }
        with self.database.engine.begin() as connection:
            if item.id is None:
                memory_id = int(
                    connection.execute(
                        insert(self.table)
                        .values(**values, created_at=now)
                        .returning(self.table.c.id)
                    ).scalar_one()
                )
                created_at = now
            else:
                existing = connection.execute(
                    select(self.table.c.created_at).where(self.table.c.id == item.id)
                ).one_or_none()
                if existing is None:
                    raise ValueError(f"unknown memory id: {item.id}")
                memory_id = item.id
                created_at = existing.created_at
                connection.execute(
                    update(self.table).where(self.table.c.id == memory_id).values(**values)
                )
        return item.model_copy(
            update={"id": memory_id, "created_at": created_at, "updated_at": now}
        )

    def list_active(self, scope: str, kind: str | None = None) -> list[MemoryItem]:
        statement = select(self.table).where(
            self.table.c.scope == scope,
            self.table.c.status == "active",
        )
        if kind is not None:
            statement = statement.where(self.table.c.kind == kind)
        statement = statement.order_by(
            self.table.c.importance.desc(), self.table.c.updated_at.desc()
        )
        with self.database.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [_memory_from_row(row) for row in rows]

    def mark_superseded(self, memory_id: int, conflict_with_id: int | None = None) -> None:
        with self.database.engine.begin() as connection:
            connection.execute(
                update(self.table)
                .where(self.table.c.id == memory_id)
                .values(
                    status="superseded",
                    conflict_with_id=conflict_with_id,
                    updated_at=utc_now(),
                )
            )

    def retrieve(self, scope: str, query: str, limit: int = 8) -> list[MemoryItem]:
        rows = self.keyword_search(scope, query, limit)
        if len(rows) >= limit:
            return rows[:limit]
        seen = {item.id for item in rows}
        statement = (
            select(self.table)
            .where(self.table.c.scope == scope, self.table.c.status == "active")
            .order_by(self.table.c.importance.desc(), self.table.c.updated_at.desc())
            .limit(limit)
        )
        with self.database.engine.connect() as connection:
            recent = connection.execute(statement).mappings().all()
        rows.extend(
            item for item in (_memory_from_row(row) for row in recent) if item.id not in seen
        )
        return rows[:limit]

    def keyword_search(self, scope: str, query: str, limit: int = 8) -> list[MemoryItem]:
        """Return PostgreSQL FTS matches for hybrid recall."""

        tokens = _query_tokens(query)
        if not tokens:
            return []
        tsquery = " | ".join(token.replace("'", "") for token in tokens)
        document = func.to_tsvector("simple", self.table.c.content)
        query_vector = func.to_tsquery("simple", tsquery)
        rank = func.ts_rank_cd(document, query_vector)
        statement = (
            select(self.table)
            .where(
                self.table.c.scope == scope,
                self.table.c.status == "active",
                document.op("@@")(query_vector),
            )
            .order_by(rank.desc(), self.table.c.importance.desc())
            .limit(limit)
        )
        with self.database.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [_memory_from_row(row) for row in rows]

    def semantic_search(
        self, scope: str, query_embedding: list[float], limit: int = 8
    ) -> list[MemoryItem]:
        """Return pgvector cosine matches across memory content and topic embeddings."""

        def nearest(column):
            distance = column.cosine_distance(query_embedding).label("distance")
            return (
                select(self.table, distance)
                .where(
                    self.table.c.scope == scope,
                    self.table.c.status == "active",
                    column.is_not(None),
                    distance < 1.0,
                )
                .order_by(distance, self.table.c.importance.desc())
                .limit(limit)
            )

        with self.database.engine.begin() as connection:
            connection.exec_driver_sql("SET LOCAL hnsw.iterative_scan = strict_order")
            connection.exec_driver_sql("SET LOCAL hnsw.ef_search = 80")
            rows = [
                *connection.execute(nearest(self.table.c.embedding)).mappings().all(),
                *connection.execute(nearest(self.table.c.topic_embedding)).mappings().all(),
            ]
        best: dict[int, tuple[float, RowMapping]] = {}
        for row in rows:
            memory_id = int(row["id"])
            candidate = (float(row["distance"]), row)
            if memory_id not in best or candidate[0] < best[memory_id][0]:
                best[memory_id] = candidate
        ranked = sorted(
            best.values(),
            key=lambda candidate: (
                candidate[0],
                -float(candidate[1]["importance"]),
            ),
        )
        return [_memory_from_row(row) for _, row in ranked[:limit]]


def _memory_from_row(row: RowMapping) -> MemoryItem:
    return MemoryItem(
        id=int(row["id"]),
        scope=str(row["scope"]),
        kind=str(row["kind"]),
        topic=str(row["topic"]),
        content=str(row["content"]),
        polarity=str(row["polarity"]),
        status=str(row["status"]),
        importance=float(row["importance"]),
        supersedes_id=row["supersedes_id"],
        conflict_with_id=row["conflict_with_id"],
        embedding=_vector_values(row["embedding"]),
        topic_embedding=_vector_values(row["topic_embedding"]),
        created_at=_datetime_value(row["created_at"]),
        updated_at=_datetime_value(row["updated_at"]),
    )


def _vector_values(value: object) -> list[float]:
    if value is None:
        return []
    return [float(component) for component in value]  # type: ignore[union-attr]


def _datetime_value(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _query_tokens(query: str) -> list[str]:
    return re.findall(r"[\w-]+", query, flags=re.UNICODE)[:16]


class CheckpointStore:
    def __init__(self, database: PostgresDatabase, scope: str) -> None:
        self.database = database
        self.scope = scope
        self.table = database.checkpoints

    def save(self, state: TaskState) -> None:
        state.updated_at = utc_now()
        statement = postgres_insert(self.table).values(
            scope=self.scope,
            task_id=state.task_id,
            state_json=state.model_dump(mode="json"),
            updated_at=state.updated_at,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[self.table.c.scope, self.table.c.task_id],
            set_={"state_json": statement.excluded.state_json, "updated_at": state.updated_at},
        )
        with self.database.engine.begin() as connection:
            connection.execute(statement)

    def load(self, task_id: str) -> TaskState | None:
        statement = select(self.table.c.state_json).where(
            self.table.c.scope == self.scope,
            self.table.c.task_id == task_id,
        )
        with self.database.engine.connect() as connection:
            value = connection.execute(statement).scalar_one_or_none()
        return TaskState.model_validate(value) if value is not None else None

    def latest_incomplete(self) -> TaskState | None:
        statement = (
            select(self.table.c.state_json)
            .where(self.table.c.scope == self.scope)
            .order_by(self.table.c.updated_at.desc())
        )
        with self.database.engine.connect() as connection:
            rows = connection.execute(statement).scalars().all()
        for value in rows:
            state = TaskState.model_validate(value)
            if state.status.value not in {"completed", "failed"}:
                return state
        return None

    def list_all(self) -> list[TaskState]:
        statement = (
            select(self.table.c.state_json)
            .where(self.table.c.scope == self.scope)
            .order_by(self.table.c.updated_at.desc())
        )
        with self.database.engine.connect() as connection:
            rows = connection.execute(statement).scalars().all()
        return [TaskState.model_validate(value) for value in rows]


class TraceStore:
    def __init__(self, database: PostgresDatabase, scope: str) -> None:
        self.database = database
        self.scope = scope
        self.table = database.trace_events

    def append(self, event: TraceEvent) -> None:
        with self.database.engine.begin() as connection:
            connection.execute(
                insert(self.table),
                {
                    "scope": self.scope,
                    "trace_id": event.trace_id,
                    "task_id": event.task_id,
                    "category": event.category,
                    "name": event.name,
                    "started_at": event.started_at,
                    "duration_ms": event.duration_ms,
                    "ok": event.ok,
                    "details_json": json_details(**event.details),
                },
            )

    def list_for_task(self, task_id: str) -> list[TraceEvent]:
        statement = (
            select(self.table)
            .where(
                self.table.c.scope == self.scope,
                self.table.c.task_id == task_id,
            )
            .order_by(self.table.c.id)
        )
        with self.database.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [
            TraceEvent(
                trace_id=str(row["trace_id"]),
                task_id=str(row["task_id"]),
                category=str(row["category"]),
                name=str(row["name"]),
                started_at=_datetime_value(row["started_at"]),
                duration_ms=float(row["duration_ms"]),
                ok=bool(row["ok"]),
                details=dict(row["details_json"]),
            )
            for row in rows
        ]


def json_details(**values: Any) -> dict[str, Any]:
    """Return JSON-safe trace details without leaking arbitrary model objects."""

    return json.loads(json.dumps(values, ensure_ascii=False, default=str))
