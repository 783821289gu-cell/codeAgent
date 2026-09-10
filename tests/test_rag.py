from pathlib import Path

import pytest

from repopilot.core.models import RetrievalResult
from repopilot.repository.index import RepositoryIndex, chunk_text
from repopilot.repository.reranker import BgeCrossEncoderReranker
from repopilot.repository.tools import WorkspaceTools


def test_chunk_text_preserves_line_numbers() -> None:
    content = "\n".join(f"line {number}" for number in range(1, 11))
    chunks = chunk_text("a.py", content, chunk_lines=5, overlap_lines=2)
    assert [(item[1], item[2]) for item in chunks] == [(1, 5), (4, 8), (7, 10)]


@pytest.mark.asyncio
async def test_hybrid_repository_retrieval(
    tmp_path: Path, vector_store, fake_embeddings, fake_reranker
) -> None:
    (tmp_path / "auth.py").write_text(
        "def login(user_id):\n    user = find_user(user_id)\n    return user.name\n",
        encoding="utf-8",
    )
    (tmp_path / "math.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    index = RepositoryIndex(vector_store, WorkspaceTools(tmp_path), fake_embeddings, fake_reranker)

    stats = await index.rebuild()
    results = await index.search("login missing user error", top_k=2)

    assert stats == {"files": 2, "chunks": 2}
    assert results[0].path == "auth.py"
    assert results[0].vector_rank == 1
    assert results[0].keyword_rank == 1


@pytest.mark.asyncio
async def test_incremental_index_embeds_only_modified_file_and_deletes_removed(
    tmp_path: Path, vector_store, fake_embeddings, fake_reranker
) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("def first():\n    return 'old'\n", encoding="utf-8")
    second.write_text("def second():\n    return 'stable'\n", encoding="utf-8")
    index = RepositoryIndex(vector_store, WorkspaceTools(tmp_path), fake_embeddings, fake_reranker)

    initial = await index.sync()
    fake_embeddings.calls.clear()
    first.write_text("def first():\n    return 'changed'\n", encoding="utf-8")
    updated = await index.sync()

    assert initial["processed"] == 2
    assert updated == {
        "files": 2,
        "scanned": 2,
        "processed": 1,
        "skipped": 1,
        "deleted": 0,
        "chunks": 1,
    }
    assert fake_embeddings.calls == [["def first():\n    return 'changed'"]]

    fake_embeddings.calls.clear()
    second.unlink()
    deleted = await index.sync()

    assert deleted["processed"] == 0
    assert deleted["deleted"] == 1
    assert deleted["files"] == 1
    assert fake_embeddings.calls == []


@pytest.mark.asyncio
async def test_bge_cross_encoder_reranker_changes_initial_order(tmp_path: Path) -> None:
    class FakeCrossEncoder:
        def predict(self, pairs, **kwargs):
            assert pairs == [("find auth bug", "first result"), ("find auth bug", "auth fix")]
            return [0.1, 0.9]

    factory_calls = []

    def factory(model_name: str, **kwargs):
        factory_calls.append((model_name, kwargs))
        return FakeCrossEncoder()

    reranker = BgeCrossEncoderReranker(
        model="BAAI/bge-reranker-base",
        revision="2cfc18c9415c912f9d8155881c133215df768a70",
        cache_dir=tmp_path,
        max_length=512,
        model_factory=factory,
    )
    first = RetrievalResult(
        path="first.py", start_line=1, end_line=1, content="first result", score=0.9
    )
    second = RetrievalResult(
        path="auth.py", start_line=1, end_line=1, content="auth fix", score=0.8
    )

    results = await reranker.rerank("find auth bug", [first, second])

    assert [item.path for item in results] == ["auth.py", "first.py"]
    assert [item.rerank_score for item in results] == [0.9, 0.1]
    assert factory_calls[0][0] == "BAAI/bge-reranker-base"
    assert factory_calls[0][1]["revision"] == "2cfc18c9415c912f9d8155881c133215df768a70"
    assert factory_calls[0][1]["max_length"] == 512
