# Technical decisions

## Verified environment

- Python 3.13, Git, ripgrep, and the OpenAI Agents SDK are available.
- The portable runtime at `D:\demo-runtime` provides PostgreSQL 16, pgvector 0.8.3, cached
  BGE-M3/BGE-reranker models, and a compatible Python ML runtime.
- PostgreSQL/pgvector integration runs in RepoPilot's `repopilot` schema; reviewAgent application
  data remains in its own schema.
- A real OpenAI-compatible DeepSeek endpoint was used for the retained Eval, MCP, and E2E Demo
  artifacts. Credentials are read from the environment and never copied into this repository.

## Decisions

1. **Keep the OpenAI Agents SDK runtime.** It continues to own model turns, function tools, MCP
   conversion, sessions, structured output validation, and tracing.
2. **Fix the macro workflow.** Workflow always runs Main planning, isolated Explorer, Main
   execution/test repair, and isolated Reviewer. Main retains final edit authority.
3. **Use PostgreSQL/pgvector as the only durable backend.** PgVectorStore supplies repository
   HNSW/cosine/FTS. The same scoped schema stores memory vectors/FTS, checkpoints, SDK sessions,
   and trace events; there is no SQLite runtime fallback.
4. **Use two-stage retrieval correctly.** RRF only fuses vector and keyword ranks. A real
   CrossEncoder separately scores the fused candidate set and may change its order.
5. **Index by file content hash.** File-level manifests make incremental update/delete behavior
   explicit and allow write tools to refresh one path without rebuilding a repository.
6. **Make memory extraction automatic.** A dedicated extractor and deterministic provenance,
   completion, deduplication, and conflict gates close the read/write loop.
7. **Separate per-call context from session compaction.** ContextManager controls which sources and
   tool results enter the next model call. SDK Responses compaction controls growing conversation
   history. They have different thresholds and traces.
8. **Adapt compatible providers at the model boundary.** OpenAI remains on Responses. Compatible
   providers use the SDK Chat Completions model; when JSON Schema response formats are absent, the
   schema is injected into instructions and SDK Pydantic validation still enforces the boundary.
9. **Make evaluation executable, not just aggregate.** EvalRunner copies each baseline, runs
   RepoPilot, executes an independent success command, compares file snapshots, checks forbidden
   changes, saves trace, and computes six high-value metrics.
10. **Use MCP for an external contract, not local file work.** The Issue demo server proves a real
    stdio discovery/call/result path. Local repository operations remain deterministic function
    tools; optional GitHub MCP tools are filtered to read-only names.

## Explicit trade-offs

- No web UI, distributed worker, queue, custom agent runtime, or plugin framework was added.
- Local shell policy is defense in depth, not VM/container isolation.
- Local BGE and CrossEncoder inference uses significant memory; Eval deliberately reuses one
  read-only model instance across isolated cases.
- OpenAI-compatible providers without a Responses endpoint do not receive SDK Responses history
  compaction; their session remains durable and per-call token pruning still applies.
- Real Eval and Demo runs consume provider tokens and are not part of the default pytest suite.
