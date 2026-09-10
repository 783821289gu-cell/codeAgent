# Architecture

## Source layout by execution chain

```text
src/repopilot/
├─ interfaces/       CLI, Demo, Eval: receive a task and present evidence
├─ application/      fixed Main → Explorer → Main → Reviewer orchestration
├─ agents/           role definitions, prompts, and role-specific tool binding
├─ repository/       workspace tools and code retrieval/indexing chain
├─ knowledge/        durable memory plus per-call context/token selection
├─ infrastructure/   PostgreSQL/pgvector, model, MCP, and trace adapters
└─ core/             shared Settings and typed cross-layer contracts
```

Read the packages in that order to follow a user task. Dependency direction starts at
`interfaces`, enters `application`, and then calls the role and capability packages. `core` is
shared contracts rather than an execution step.

## Responsibility map

| Package | Main modules | Responsibility |
|---|---|---|
| `interfaces` | `cli.py`, `demo.py`, `evaluation.py` | Commands, isolated runs, reports |
| `application` | `workflow.py` | Task lifecycle, fixed agent phases, completion gate |
| `agents` | `factory.py`, `prompts.py` | Agent roles, permissions, structured outputs |
| `repository` | `tools.py`, `index.py`, `vector_store.py` | Workspace execution, RAG, embeddings, reranking |
| `knowledge` | `memory.py`, `context.py`, `tokens.py` | Recall/write lifecycle and model-input budgets |
| `infrastructure` | `postgres.py`, `llm.py`, `mcp.py`, `tracing.py` | External and persistence adapters |
| `core` | `config.py`, `models.py` | Configuration and stable data contracts |

Model-provider selection does not change agents, tools, prompts, or the macro workflow.

## Retrieval pipeline

```text
repository file -> SHA-256 manifest -> changed-file chunking -> BGE-M3 embedding
                                                     |
                        pgvector cosine/HNSW + PostgreSQL full-text search
                                                     |
                                              reciprocal-rank fusion
                                                     |
                                        BGE CrossEncoder candidate scoring
                                                     |
                                                  final Top-K
```

`RepositoryIndex.sync()` compares file manifests. Unchanged files are skipped; changed files are
replaced atomically at file scope; missing files are deleted. A successful write tool calls
`sync([path])`, so Agent context does not rely on a stale pre-task index.

## Three-agent design

The Workflow explicitly fixes the macro order as Main planning, isolated Explorer, Main execution
and test repair, then isolated Reviewer. Explorer and Reviewer run without Main's SDK session, so
each receives fresh specialist context. Only Main receives edit, shell, and test tools. Explorer
receives read/search/RAG tools; Reviewer additionally receives Git diff/status.

The workflow does not trust a final model message as completion. The stored task becomes complete
only when the latest deterministic test passed and Reviewer returned `approved=true`.

## Memory and context

Long-term memory is scoped by a workspace hash. Retrieval uses the configured embedding service.
A dedicated post-task agent extracts only the five durable categories, and `MemoryLifecycle`
performs semantic duplicate, update, and conflict handling before storage.

`ContextManager` owns per-call selection. It assigns token budgets to Task, State, Memory, RAG,
Tool, Test, Explorer, Reviewer, and history; it prunes function-call/output pairs together and
semantically compresses large outputs. The SDK session owns conversation persistence. OpenAI
Responses runs additionally use `OpenAIResponsesCompactionSession` for long history; compatible
Chat Completions providers retain the normal PostgreSQL session when no Responses compaction
endpoint is available.

## Persistence and evidence

Checkpoint state answers what work remains, what changed, and whether tests/review passed. SDK
session rows answer what the model and tools exchanged. PostgreSQL trace events answer which
application, agent, retrieval, tool, test, memory, context, MCP, and usage steps ran. Workspace
scope is part of checkpoint and trace keys so repositories sharing one database remain isolated.
Eval and Demo runners retain modified workspaces plus machine-readable reports.
