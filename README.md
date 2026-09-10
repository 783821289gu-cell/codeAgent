# RepoPilot

RepoPilot is a compact multi-agent coding agent for real repositories. A Main agent coordinates a
read-only Explorer and an independent Reviewer, makes bounded edits, runs and repairs tests, keeps
the repository index current, extracts durable memory, and persists checkpoints and traces.

The project uses the OpenAI Agents SDK for agent loops, function tools, sessions, structured
outputs, MCP, and model tracing. RepoPilot owns the fixed multi-agent coding workflow,
retrieval, memory lifecycle, context selection, safety policy, local trace, and evaluation runner.

## Implemented system

- **Three isolated roles.** Main owns edits and tests; Explorer owns read-only investigation;
  Reviewer independently inspects the final diff and evidence. Specialist boundaries use typed
  `ExplorationReport` and `ReviewReport` results.
- **Production hybrid RAG.** Normalized BGE-M3 embeddings feed pgvector cosine search with an HNSW
  index. PostgreSQL full-text search supplies keyword recall, RRF fuses both rank lists, and
  `BAAI/bge-reranker-base` performs true CrossEncoder second-stage ranking before Top-K selection.
- **Unified PostgreSQL storage.** Repository and memory embeddings use pgvector; PostgreSQL also
  owns FTS, checkpoints, SDK session history, durable memory, and application traces.
- **Incremental repository indexing.** File SHA-256 manifests skip unchanged content, re-chunk and
  re-embed changed files, and remove deleted files. Successful write tools refresh only the changed
  path; the final sync verifies those files are already current.
- **Closed-loop memory.** Each task begins with semantic memory retrieval and ends with a dedicated
  Memory Extractor. Only project constraints, user preferences, architecture decisions, coding
  conventions, and reusable experience are eligible. Semantic duplicates are skipped; updates and
  opposing rules supersede old entries.
- **Token-level context engineering.** `tiktoken` budgets Task, State, Memory, RAG, Tool, Test,
  Explorer, Reviewer, and history sources. Large tool output is pruned semantically, passing tests
  collapse to their summary, and failing tests retain assertions, exceptions, trace evidence, and
  the failure summary. Function calls and outputs are pruned as pairs.
- **Durable execution.** PostgreSQL stores typed checkpoints, SDK session items, long-term memory,
  and trace events. The task is complete only when the latest tests pass and Reviewer approves.
- **Executable evaluation and demos.** `repopilot eval` runs 3-5 tasks in isolated baseline copies
  and records success, tests, changed files, turns, tool calls, tokens, latency, and trace.
  `repopilot demo` and `repopilot mcp-demo` retain report, trace, diff, and independent test output.

## Architecture

```mermaid
flowchart TD
    CLI["CLI: run / eval / demo"] --> APP["RepoPilot workflow"]
    APP --> MEMORY["Memory retrieval + extraction"]
    APP --> PLAN["Main: understand + plan"]
    PLAN --> EXPLORER["Explorer: isolated read-only investigation"]
    EXPLORER --> MAIN["Main: edit + test + repair"]
    MAIN --> REVIEWER["Reviewer: isolated diff review"]
    REVIEWER --> MAIN
    MAIN --> TOOLS["Bounded file / shell / test / Git tools"]
    EXPLORER --> RAG["Hybrid repository retrieval"]
    RAG --> VEC["BGE-M3 + pgvector HNSW"]
    RAG --> FTS["PostgreSQL FTS"]
    VEC --> RRF["RRF fusion"]
    FTS --> RRF
    RRF --> RERANK["BGE CrossEncoder reranker"]
    RERANK --> MAIN
    MAIN --> MCP["Optional GitHub or Issue MCP"]
    APP --> CONTEXT["Per-call token budget + pruning"]
    APP --> SESSION["SDK session / Responses compaction"]
    APP --> STATE["Checkpoint + local trace"]
```

Agents depend on retrieval, memory, and deterministic tools; those services never call an agent.
See [architecture](docs/architecture.md) and [technical decisions](docs/technical-decisions.md).

### Source layout follows the runtime chain

```text
src/repopilot/
├─ interfaces/       CLI, Demo, Eval entrypoints
├─ application/      fixed multi-agent workflow
├─ agents/           Main, Explorer, Reviewer roles and prompts
├─ repository/       bounded code tools and hybrid RAG
├─ knowledge/        Memory, ContextManager, token budgets
├─ infrastructure/   PostgreSQL/pgvector, LLM, MCP, tracing
└─ core/             Settings and typed contracts
```

For a first read, follow `interfaces/cli.py → application/workflow.py → agents/factory.py`, then
enter `repository`, `knowledge`, and `infrastructure` only when the workflow calls them.

## Quick start

Python 3.12+, Git, and ripgrep are required.

```powershell
cd D:\demo-projects\codeAgent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

RepoPilot requires PostgreSQL with the `vector` extension for all durable state and retrieval:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://USER:PASSWORD@127.0.0.1:5432/DATABASE"
```

BGE-M3 and the CrossEncoder run locally through sentence-transformers. Set
`REPOPILOT_EMBEDDING_CACHE_DIR` and `REPOPILOT_RERANKER_CACHE_DIR` when models live outside the
standard Hugging Face cache.

### Model provider

For the OpenAI Responses API:

```powershell
$env:OPENAI_API_KEY = "..."
$env:REPOPILOT_MODEL = "YOUR_AVAILABLE_MODEL"
```

For an OpenAI-compatible Chat Completions provider:

```powershell
$env:REPOPILOT_LLM_BASE_URL = "https://provider.example/v1"
$env:REPOPILOT_LLM_API_KEY = "..."
$env:REPOPILOT_MODEL = "provider-model"
```

The equivalent `REVIEW_AGENT_LLM_BASE_URL`, `REVIEW_AGENT_LLM_API_KEY`,
`REVIEW_AGENT_LLM_MODEL`, `REVIEW_AGENT_LLM_TIMEOUT_SECONDS`, and
`REVIEW_AGENT_DATABASE_URL` variables are accepted so the verified local reviewAgent runtime can
be reused without copying credentials into this repository. Compatible providers that lack JSON
Schema response formats receive the schema in the system instruction and the SDK still validates
the returned Pydantic object.

```powershell
.\.venv\Scripts\repopilot.exe doctor
.\.venv\Scripts\repopilot.exe index --workspace D:\path\to\repository
.\.venv\Scripts\repopilot.exe run --workspace D:\path\to\repository `
  "Fix the login endpoint returning HTTP 500 for an unknown user and add a regression test."
```

All RepoPilot state is stored in PostgreSQL. Repository vectors, memories, checkpoints, traces,
and session IDs are scoped by a stable workspace hash. The schema defaults to `repopilot`, separate
from reviewAgent's application schema. Legacy `<workspace>/.repopilot/repopilot.sqlite3` files are
not read by the current runtime.

## Commands

```text
repopilot doctor                         Inspect provider, retrieval, and local prerequisites
repopilot index -w PATH                  Incrementally synchronize the repository index
repopilot run -w PATH "TASK"             Execute an autonomous coding task
repopilot resume -w PATH TASK_ID         Continue checkpoint and SDK session state
repopilot trace -w PATH TASK_ID          Display the persisted local trace
repopilot eval                           Run the 3-5 case isolated coding benchmark
repopilot eval-report -w PATH            Aggregate already-persisted task metrics
repopilot demo                           Run the real-model login E2E and retain evidence
repopilot mcp-demo                       Read Issue 101 via stdio MCP, then fix and verify it
```

Use `repopilot COMMAND --help` for options. `--allow-dangerous` is an explicit per-run opt-in for
otherwise blocked shell syntax; it does not remove workspace path boundaries.

## Runtime workflow

1. Incrementally synchronize the repository index.
2. Retrieve relevant long-term memory and hybrid RAG context.
3. Workflow runs Explorer with isolated read-only context and receives a structured report.
4. Workflow returns the report to Main, which reads files and applies exact, stale-safe edits.
5. Each changed path is immediately re-indexed.
6. Main runs tests; failures return bounded diagnostic evidence and increment retry state.
7. Main repairs failures; Workflow then runs the isolated Reviewer.
8. Main addresses findings and reruns final tests when needed.
9. The completion gate requires passing tests and `review.approved=true`.
10. Memory Extractor persists durable knowledge; checkpoint, session, and trace remain inspectable.

The custom ContextManager decides what enters each model call and enforces source token budgets.
For OpenAI Responses models, `OpenAIResponsesCompactionSession` separately compresses long SDK
conversation history. OpenAI-compatible providers retain the PostgreSQL SDK session when they do
not implement the Responses compaction endpoint; per-call ContextManager pruning remains active.

## Reproducible evidence

The following runs were executed against isolated copies of `demo/bug_repo` on 2026-08-14 with a
real OpenAI-compatible `deepseek-v4-pro` model, local BGE-M3/CrossEncoder, and PostgreSQL/pgvector:

| Run | Result | Evidence |
|---|---|---|
| `repopilot demo` | completed; 2 changed files; two in-agent test passes; Reviewer approved; independent `2 passed` | `artifacts/demo/demo-20260814T030127Z-5a901a/` |
| `repopilot mcp-demo` | real `get_issue(101)` MCP call; completed; Reviewer approved; independent `3 passed` | `artifacts/mcp-demo/mcp-demo-20260814T025708Z-a03a8f/` |
| `repopilot eval` | 3/3 task success; 3/3 test pass; 9.67 average turns; 22 average tool calls; 354,058 tokens | `artifacts/evals/20260814T024314Z-a50167/` |

Each Demo directory contains `report.json`, `trace.json`, `git.diff`, `test-output.txt`, and the
modified workspace. Each Eval case contains its isolated workspace, `result.json`, and
`trace.json`; the run root contains `summary.json`.

## Configuration

| Setting | Default | Purpose |
|---|---:|---|
| `REPOPILOT_MODEL` | `gpt-5.6-terra` | Main, specialists, memory extractor, and compactor model |
| `REPOPILOT_LLM_BASE_URL` | unset | Enable an OpenAI-compatible Chat Completions provider |
| `REPOPILOT_LLM_API_KEY` | unset | Compatible-provider credential |
| `REPOPILOT_LLM_TIMEOUT_SECONDS` | `120` | External LLM request timeout |
| `REPOPILOT_EMBEDDING_PROVIDER` | `bge` | `bge` or `openai` embeddings |
| `REPOPILOT_EMBEDDING_MODEL` | `BAAI/bge-m3` | Repository and memory embedding model |
| `REPOPILOT_EMBEDDING_DIMENSIONS` | `1024` | Vector dimension |
| `REPOPILOT_RERANKER_MODEL` | `BAAI/bge-reranker-base` | CrossEncoder reranker |
| `DATABASE_URL` | required | PostgreSQL/pgvector connection URL |
| `REPOPILOT_PGVECTOR_SCHEMA` | `repopilot` | Isolated PostgreSQL schema |
| `REPOPILOT_MAX_TURNS` | `30` | SDK turn limit |
| `REPOPILOT_MAX_TOOL_OUTPUT_TOKENS` | `3000` | Tool-result limit |
| `REPOPILOT_CONTEXT_BUDGET_TOKENS` | `12000` | Per-model-call context budget |
| `REPOPILOT_COMPACTION_THRESHOLD_TOKENS` | `9000` | Initial structured context trigger |
| `REPOPILOT_SDK_COMPACTION_THRESHOLD_TOKENS` | `100000` | OpenAI Responses history trigger |
| `REPOPILOT_RAG_TOP_K` | `8` | Final reranked retrieval count |
| `REPOPILOT_GITHUB_MCP_URL` | disabled | Optional read-only GitHub MCP endpoint |

Changing embedding dimensions requires a fresh compatible vector index.

## Quality checks

The default suite injects fake embeddings/reranking, uses isolated PostgreSQL schemas, and does not
make paid model calls:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest -q
```

Run the portable PostgreSQL integration explicitly:

```powershell
$env:REPOPILOT_RUN_PGVECTOR_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -q tests\test_pgvector_integration.py
```

Tests cover workspace escape prevention, command policy, token-bounded output, exact edits,
CrossEncoder reordering, hybrid retrieval, pgvector HNSW/FTS, incremental update/delete, semantic
memory lifecycle, cross-session recall, long-task context pruning, three-agent completion, real
stdio MCP transport, Eval isolation, checkpoints, and local trace evidence.

## Safety and limits

- Every file path resolves inside the configured workspace, including symlink resolution.
- Shell commands have a timeout and token-bounded output. Destructive filesystem/Git/SQL syntax,
  redirection, parent/home references, and shutdown commands are blocked by default.
- Exact replacements fail on stale or ambiguous occurrence counts.
- Explorer and Reviewer never receive editing or general shell tools.
- Mutation-like tools are filtered from the optional GitHub MCP connection.
- Secrets are excluded from local trace details and sensitive SDK trace payloads are disabled.

Local command filtering is defense in depth, not a VM-grade sandbox. Run RepoPilot only against
repositories and commands you trust.
