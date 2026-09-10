from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from repopilot.core.models import RetrievalResult
from repopilot.repository.embeddings import EmbeddingProvider
from repopilot.repository.reranker import Reranker
from repopilot.repository.tools import WorkspaceTools
from repopilot.repository.vector_store import IndexedChunk, IndexedFile, VectorStore

INDEXED_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True, slots=True)
class _PendingFile:
    path: str
    content_hash: str
    chunks: tuple[tuple[str, int, int, str], ...]


def chunk_text(
    path: str, content: str, *, chunk_lines: int = 80, overlap_lines: int = 15
) -> list[tuple[str, int, int, str]]:
    lines = content.splitlines()
    if not lines:
        return []
    step = chunk_lines - overlap_lines
    chunks: list[tuple[str, int, int, str]] = []
    for start_index in range(0, len(lines), step):
        chosen = lines[start_index : start_index + chunk_lines]
        if not chosen:
            break
        start_line = start_index + 1
        end_line = start_index + len(chosen)
        chunks.append((path, start_line, end_line, "\n".join(chosen)))
        if end_line == len(lines):
            break
    return chunks


class RepositoryIndex:
    def __init__(
        self,
        vector_store: VectorStore,
        tools: WorkspaceTools,
        embeddings: EmbeddingProvider,
        reranker: Reranker,
        *,
        chunk_lines: int = 80,
        overlap_lines: int = 15,
        max_file_bytes: int = 1_000_000,
    ) -> None:
        if vector_store.dimensions != embeddings.dimensions:
            raise ValueError("vector store and embedding dimensions must match")
        self.vector_store = vector_store
        self.tools = tools
        self.embeddings = embeddings
        self.reranker = reranker
        self.chunk_lines = chunk_lines
        self.overlap_lines = overlap_lines
        self.max_file_bytes = max_file_bytes
        self.scope = repository_scope(tools.boundary.workspace)

    def count(self) -> int:
        return self.vector_store.count(self.scope)

    async def rebuild(self) -> dict[str, int]:
        stats = await self._sync(None, force=True)
        return {"files": stats["files"], "chunks": stats["chunks"]}

    async def sync(self, changed_files: Sequence[str] | None = None) -> dict[str, int]:
        """Synchronize changed files by content hash; a full scan also detects deletions."""

        return await self._sync(changed_files, force=False)

    async def _sync(self, changed_files: Sequence[str] | None, *, force: bool) -> dict[str, int]:
        existing_hashes = self.vector_store.file_hashes(self.scope)
        if changed_files is None:
            listing = self.tools.list_files(limit=100_000)
            if not listing.ok or listing.data is None:
                raise RuntimeError(listing.error or "could not enumerate repository")
            candidate_paths = set(listing.data)
        else:
            candidate_paths = {
                self.tools.boundary.relative(self.tools.boundary.resolve(path))
                for path in changed_files
            }

        eligible_paths: set[str] = set()
        pending: list[_PendingFile] = []
        skipped = 0
        deleted_paths: set[str] = set()
        for relative in sorted(candidate_paths):
            path = self.tools.boundary.resolve(relative)
            if (
                not path.is_file()
                or path.suffix.casefold() not in INDEXED_SUFFIXES
                or path.stat().st_size > self.max_file_bytes
            ):
                if relative in existing_hashes:
                    deleted_paths.add(relative)
                continue
            eligible_paths.add(relative)
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                skipped += 1
                continue
            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if not force and existing_hashes.get(relative) == file_hash:
                skipped += 1
                continue
            chunks = tuple(
                chunk_text(
                    relative,
                    content,
                    chunk_lines=self.chunk_lines,
                    overlap_lines=self.overlap_lines,
                )
            )
            pending.append(_PendingFile(relative, file_hash, chunks))

        if changed_files is None:
            deleted_paths.update(set(existing_hashes) - eligible_paths)

        raw_chunks = [chunk for file in pending for chunk in file.chunks]
        vectors: list[list[float]] = []
        for offset in range(0, len(raw_chunks), 64):
            vectors.extend(
                await self.embeddings.embed(
                    [chunk[3] for chunk in raw_chunks[offset : offset + 64]]
                )
            )
        if len(vectors) != len(raw_chunks):
            raise RuntimeError("embedding provider returned a different number of vectors")

        indexed_files: list[IndexedFile] = []
        vector_offset = 0
        for file in pending:
            file_vectors = vectors[vector_offset : vector_offset + len(file.chunks)]
            vector_offset += len(file.chunks)
            indexed_files.append(
                IndexedFile(
                    path=file.path,
                    content_hash=file.content_hash,
                    chunks=tuple(
                        IndexedChunk(
                            path=path,
                            start_line=start_line,
                            end_line=end_line,
                            content=content,
                            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                            embedding=vector,
                        )
                        for (path, start_line, end_line, content), vector in zip(
                            file.chunks, file_vectors, strict=True
                        )
                    ),
                )
            )

        self.vector_store.sync_files(self.scope, indexed_files, sorted(deleted_paths))
        return {
            "files": len(self.vector_store.file_hashes(self.scope)),
            "scanned": len(eligible_paths),
            "processed": len(indexed_files),
            "skipped": skipped,
            "deleted": len(deleted_paths),
            "chunks": len(raw_chunks),
        }

    async def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        if not query.strip():
            return []
        query_vectors = await self.embeddings.embed([query])
        vector_ranks: dict[int, tuple[int, float]] = {}
        keyword_ranks: dict[int, int] = {}
        candidates: dict[int, object] = {}
        recall_limit = max(top_k * 3, 12)
        vector_rows = self.vector_store.vector_search(self.scope, query_vectors[0], recall_limit)
        for rank, hit in enumerate(vector_rows, start=1):
            rowid = hit.chunk.id
            vector_ranks[rowid] = (rank, hit.distance)
            candidates[rowid] = hit.chunk

        keyword_rows = self.vector_store.keyword_search(self.scope, query, recall_limit)
        for rank, hit in enumerate(keyword_rows, start=1):
            rowid = hit.chunk.id
            keyword_ranks[rowid] = rank
            candidates[rowid] = hit.chunk

        # Reciprocal-rank fusion is robust across incomparable FTS and vector score scales.
        ranked: list[tuple[float, int]] = []
        for rowid in candidates:
            score = 0.0
            if rowid in vector_ranks:
                score += 1.0 / (60 + vector_ranks[rowid][0])
            if rowid in keyword_ranks:
                score += 1.0 / (60 + keyword_ranks[rowid])
            ranked.append((score, rowid))
        ranked.sort(reverse=True)
        results: list[RetrievalResult] = []
        candidate_limit = max(top_k * 2, 12)
        for score, rowid in ranked[:candidate_limit]:
            row = candidates[rowid]
            results.append(
                RetrievalResult(
                    path=row.path,
                    start_line=row.start_line,
                    end_line=row.end_line,
                    content=row.content,
                    score=score,
                    rrf_score=score,
                    vector_rank=vector_ranks.get(rowid, (None, 0.0))[0],
                    keyword_rank=keyword_ranks.get(rowid),
                )
            )
        reranked = await self.reranker.rerank(query, results)
        return reranked[:top_k]


def repository_scope(workspace: Path) -> str:
    return hashlib.sha256(str(workspace.resolve()).casefold().encode()).hexdigest()[:16]
