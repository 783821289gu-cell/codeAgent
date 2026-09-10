import pytest

from repopilot.core.models import MemoryItem, MemoryKind, TaskState, TaskStatus
from repopilot.infrastructure.postgres import (
    CheckpointStore,
    MemoryStore,
    PostgresDatabase,
    PostgresSession,
)


def test_memory_write_retrieve_update_and_deduplicate(database) -> None:
    store = MemoryStore(database)
    first = store.write(
        MemoryItem(
            scope="repo",
            kind=MemoryKind.CONSTRAINT,
            content="Service layer must not access the database directly",
            importance=0.6,
        )
    )
    duplicate = store.write(
        MemoryItem(
            scope="repo",
            kind=MemoryKind.CONSTRAINT,
            content="  Service layer MUST not access the database directly ",
            importance=0.9,
        )
    )

    assert duplicate.id == first.id
    found = store.retrieve("repo", "service database")
    assert len(found) == 1
    assert found[0].importance == 0.9


def test_checkpoint_round_trip_and_resume(database) -> None:
    store = CheckpointStore(database, "repo")
    state = TaskState(task_id="task-1", goal="fix login", status=TaskStatus.RUNNING)
    state.changed_files.append("app.py")
    store.save(state)

    loaded = store.load("task-1")
    assert loaded is not None
    assert loaded.changed_files == ["app.py"]
    assert store.latest_incomplete().task_id == "task-1"  # type: ignore[union-attr]

    loaded.status = TaskStatus.COMPLETED
    store.save(loaded)
    assert store.latest_incomplete() is None


@pytest.mark.asyncio
async def test_agents_sdk_session_round_trip_uses_postgres(database) -> None:
    session = PostgresSession("repo:task-1", database)
    await session.add_items(
        [
            {"role": "user", "content": "修复登录错误"},
            {"role": "assistant", "content": "开始分析"},
        ]
    )

    assert await session.get_items() == [
        {"role": "user", "content": "修复登录错误"},
        {"role": "assistant", "content": "开始分析"},
    ]
    assert await session.pop_item() == {
        "role": "assistant",
        "content": "开始分析",
    }
    await session.clear_session()
    assert await session.get_items() == []


def test_memory_vector_dimension_mismatch_is_rejected(database) -> None:
    with pytest.raises(ValueError, match="memory vector dimensions"):
        PostgresDatabase(
            database.database_url,
            dimensions=6,
            schema=database.schema,
        )
