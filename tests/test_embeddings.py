from pathlib import Path

import pytest

from repopilot.repository.embeddings import BGE_EMBEDDING_DIMENSIONS, BgeEmbeddingProvider


@pytest.mark.asyncio
async def test_bge_embedding_uses_fixed_revision_and_normalizes(tmp_path: Path) -> None:
    class FakeModel:
        def encode(self, texts, **kwargs):
            assert texts == ["auth code"]
            return [[2.0, *([0.0] * (BGE_EMBEDDING_DIMENSIONS - 1))]]

    calls = []

    def factory(model_name: str, **kwargs):
        calls.append((model_name, kwargs))
        return FakeModel()

    provider = BgeEmbeddingProvider(
        model="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        cache_dir=tmp_path,
        max_length=1024,
        model_factory=factory,
    )

    vectors = await provider.embed(["auth code"])

    assert len(vectors[0]) == 1024
    assert vectors[0][0] == 1.0
    assert calls[0][1]["revision"] == "5617a9f61b028005a4858fdac845db406aefb181"
    assert calls[0][1]["max_length"] == 1024
