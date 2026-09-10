from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from agents import (
    Agent,
    OpenAIResponsesCompactionSession,
    RunConfig,
    Runner,
    gen_trace_id,
    trace,
)
from agents.mcp import MCPServer, MCPServerManager
from agents.models.interface import Model
from openai import AsyncOpenAI

from repopilot.agents.factory import AgentFactory, RuntimeState
from repopilot.agents.prompts import (
    COMPACTION_PROMPT,
    MAIN_PLANNING_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
)
from repopilot.core.config import Settings
from repopilot.core.models import (
    ContextItem,
    ContextSource,
    ExtractedMemory,
    MemoryExtractionReport,
    MemoryKind,
    TaskOutcome,
    TaskState,
    TaskStatus,
)
from repopilot.infrastructure.llm import OpenAICompatibleChatModel
from repopilot.infrastructure.mcp import build_mcp_servers
from repopilot.infrastructure.postgres import (
    CheckpointStore,
    MemoryStore,
    PostgresDatabase,
    PostgresSession,
    TraceStore,
)
from repopilot.infrastructure.tracing import LocalRunHooks, LocalTracer
from repopilot.knowledge.context import ContextManager
from repopilot.knowledge.memory import MemoryLifecycle
from repopilot.knowledge.tokens import TokenCounter
from repopilot.repository.embeddings import (
    BgeEmbeddingProvider,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from repopilot.repository.index import RepositoryIndex, repository_scope
from repopilot.repository.reranker import BgeCrossEncoderReranker, Reranker
from repopilot.repository.tools import WorkspaceTools
from repopilot.repository.vector_store import PgVectorStore


@asynccontextmanager
async def connected_mcp_servers(servers: list[MCPServer]) -> AsyncIterator[list[MCPServer]]:
    if not servers:
        yield []
        return
    async with MCPServerManager(servers, drop_failed_servers=True) as manager:
        yield list(manager.active_servers)


class RepoPilot:
    """Application orchestration around the SDK runtime and deterministic services."""

    def __init__(
        self,
        settings: Settings,
        *,
        repository_index: RepositoryIndex | None = None,
        embeddings: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        model: str | Model | None = None,
    ) -> None:
        self.settings = settings
        self._model_injected = model is not None
        self.model = model or self._configured_model(settings)
        self.token_counter = TokenCounter(settings.model)
        if settings.postgres_url is None:
            raise ValueError("DATABASE_URL is required; RepoPilot uses PostgreSQL + pgvector")
        postgres_url = settings.postgres_url.get_secret_value()
        self.scope = repository_scope(settings.workspace)
        self.database = PostgresDatabase(
            postgres_url,
            settings.embedding_dimensions,
            schema=settings.pgvector_schema,
        )
        self.tools = WorkspaceTools(
            settings.workspace,
            timeout_seconds=settings.shell_timeout_seconds,
            max_output_tokens=settings.max_tool_output_tokens,
            token_counter=self.token_counter,
            allow_dangerous=settings.allow_dangerous_commands,
        )
        if repository_index is None:
            if embeddings is not None:
                embedding_service = embeddings
            elif settings.embedding_provider == "bge":
                embedding_service = BgeEmbeddingProvider(
                    model=settings.embedding_model,
                    revision=settings.embedding_revision,
                    cache_dir=settings.embedding_cache_dir,
                    device=settings.embedding_device,
                    max_length=settings.embedding_max_length,
                    batch_size=settings.embedding_batch_size,
                )
            else:
                embedding_service = OpenAIEmbeddingProvider(
                    model=settings.embedding_model,
                    dimensions=settings.embedding_dimensions,
                    api_key=(
                        settings.openai_api_key.get_secret_value()
                        if settings.openai_api_key
                        else None
                    ),
                )
            if embedding_service.dimensions != settings.embedding_dimensions:
                raise ValueError("embedding provider and configured dimensions must match")
            vector_store = PgVectorStore(
                postgres_url,
                settings.embedding_dimensions,
                schema=settings.pgvector_schema,
                engine=self.database.engine,
            )
            reranking_service = reranker or BgeCrossEncoderReranker(
                model=settings.reranker_model,
                revision=settings.reranker_revision,
                cache_dir=settings.reranker_cache_dir,
                device=settings.reranker_device,
                max_length=settings.reranker_max_length,
                batch_size=settings.reranker_batch_size,
            )
            repository_index = RepositoryIndex(
                vector_store,
                self.tools,
                embedding_service,
                reranking_service,
                chunk_lines=settings.chunk_lines,
                overlap_lines=settings.chunk_overlap_lines,
            )
        self.repository_index = repository_index
        self.memory = MemoryStore(self.database)
        self.memory_lifecycle = MemoryLifecycle(
            self.memory,
            self.repository_index.embeddings,
            duplicate_threshold=settings.memory_duplicate_threshold,
            topic_threshold=settings.memory_topic_threshold,
        )
        self.checkpoints = CheckpointStore(self.database, self.scope)
        self.trace_store = TraceStore(self.database, self.scope)
        self.context = ContextManager(
            settings.context_budget_tokens,
            settings.compaction_threshold_tokens,
            self.token_counter,
        )
        self.factory = AgentFactory(
            settings,
            self.tools,
            self.repository_index,
            self.memory,
            self.checkpoints,
            memory_lifecycle=self.memory_lifecycle,
            model=self.model,
        )

    async def index(self) -> dict[str, int]:
        return await self.repository_index.sync()

    async def run(
        self,
        goal: str,
        *,
        task_id: str | None = None,
        resume: bool = False,
        ensure_index: bool = True,
    ) -> TaskOutcome:
        if not goal.strip():
            raise ValueError("goal cannot be empty")
        if not self._model_injected and not self._has_model_credentials():
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAI, or configure "
                "REPOPILOT_LLM_BASE_URL and REPOPILOT_LLM_API_KEY for an "
                "OpenAI-compatible provider"
            )
        state = self._load_or_create_state(goal, task_id, resume)
        state.status = TaskStatus.RUNNING
        state.current_step = "loading_context"
        self.checkpoints.save(state)

        trace_id = gen_trace_id()
        local_trace = LocalTracer(self.trace_store, trace_id, state.task_id)
        runtime = RuntimeState(state, repository_scope(self.settings.workspace), tracer=local_trace)
        try:
            with local_trace.span("application", "task", goal=goal) as task_details:
                if ensure_index:
                    with local_trace.span("rag", "incremental_index") as details:
                        details.update(await self.repository_index.sync())
                servers = build_mcp_servers(self.settings)
                async with connected_mcp_servers(servers) as active_servers:
                    main_agent = self.factory.build_main(runtime, active_servers)
                    planning_agent = main_agent.clone(
                        instructions=MAIN_PLANNING_PROMPT,
                        tools=[],
                        mcp_servers=[],
                    )
                    planning_prompt = await self._build_planning_prompt(runtime)
                    with local_trace.span("agent", "main_planning") as planning_details:
                        planning_result = await self._run_main(
                            planning_agent,
                            planning_prompt,
                            runtime,
                            trace_id,
                            max_turns=2,
                        )
                        planning_details["model_responses"] = len(planning_result.raw_responses)
                    runtime.initial_plan = str(planning_result.final_output)
                    state.current_step = "initial_plan_complete"
                    self.checkpoints.save(state)

                    explorer = self.factory.build_explorer(runtime)
                    explorer_prompt = await self._build_explorer_prompt(runtime)
                    explorer_result = await self._run_isolated(explorer, explorer_prompt, runtime)
                    self.factory.record_exploration(runtime, explorer_result)

                    execution_prompt = await self._build_execution_prompt(runtime)
                    main_results = [planning_result]
                    with local_trace.span("agent", "main_execution") as main_details:
                        result = await self._run_main(
                            main_agent, execution_prompt, runtime, trace_id
                        )
                        main_details["model_responses"] = len(result.raw_responses)
                    main_results.append(result)

                    remaining_repairs = 3
                    while (
                        not result.interruptions
                        and state.test_status != "passed"
                        and remaining_repairs > 0
                    ):
                        remaining_repairs -= 1
                        result = await self._run_main(
                            main_agent,
                            "Tests are not passing yet. Inspect the latest failure evidence, "
                            "make the smallest valid repair, and run the relevant tests again.",
                            runtime,
                            trace_id,
                        )
                        main_results.append(result)

                    reviewer = self.factory.build_reviewer(runtime)
                    review_rounds = 0
                    while not result.interruptions and review_rounds < 3:
                        review_rounds += 1
                        review_prompt = await self._build_review_prompt(runtime)
                        review_result = await self._run_isolated(reviewer, review_prompt, runtime)
                        review = self.factory.record_review(runtime, review_result)
                        if review.approved or remaining_repairs <= 0:
                            break
                        remaining_repairs -= 1
                        result = await self._run_main(
                            main_agent,
                            "# Independent Reviewer report\n"
                            + review.model_dump_json(indent=2)
                            + "\n\nAssess every finding. Repair valid findings and run tests; "
                            "do not modify code for invalid findings.",
                            runtime,
                            trace_id,
                        )
                        main_results.append(result)
                        while (
                            not result.interruptions
                            and state.test_status != "passed"
                            and remaining_repairs > 0
                        ):
                            remaining_repairs -= 1
                            result = await self._run_main(
                                main_agent,
                                "The review repair is not verified. Run the relevant tests and "
                                "fix only evidence-backed failures before review resumes.",
                                runtime,
                                trace_id,
                            )
                            main_results.append(result)
                if state.changed_files:
                    with local_trace.span("rag", "changed_files_index") as details:
                        details.update(await self.repository_index.sync(state.changed_files))
                if result.interruptions:
                    state.current_step = "approval_required"
                    state.status = TaskStatus.RUNNING
                    self.checkpoints.save(state)
                    task_details.update(
                        status=state.status.value,
                        current_step=state.current_step,
                        interruptions=len(result.interruptions),
                    )
                    return TaskOutcome(
                        task_id=state.task_id,
                        status=state.status,
                        summary="An MCP or dangerous tool call is waiting for human approval.",
                        changed_files=state.changed_files,
                        tests=runtime.tests,
                        review=runtime.review,
                        trace_id=trace_id,
                    )
                summary = str(result.final_output)
                completed = (
                    state.test_status == "passed"
                    and runtime.review is not None
                    and runtime.review.approved
                )
                with local_trace.span("memory", "extract") as memory_details:
                    extraction = await self._extract_memories(
                        runtime, summary, local_trace, completed=completed
                    )
                    outcomes = await self.memory_lifecycle.remember_many(
                        runtime.memory_scope, extraction.items
                    )
                    memory_details.update(
                        candidates=len(extraction.items),
                        stored=len(outcomes),
                        actions=[outcome.action for outcome in outcomes],
                    )
                state.status = TaskStatus.COMPLETED if completed else TaskStatus.FAILED
                state.current_step = "completed" if completed else "verification_incomplete"
                self.checkpoints.save(state)
                for main_result in main_results:
                    self._record_usage(local_trace, main_result.raw_responses)
                task_details.update(
                    status=state.status.value,
                    current_step=state.current_step,
                    changed_files=len(state.changed_files),
                    tests=len(runtime.tests),
                    review_approved=bool(runtime.review and runtime.review.approved),
                )
                return TaskOutcome(
                    task_id=state.task_id,
                    status=state.status,
                    summary=summary,
                    changed_files=state.changed_files,
                    tests=runtime.tests,
                    review=runtime.review,
                    trace_id=trace_id,
                )
        except Exception:
            state.status = TaskStatus.FAILED
            state.current_step = "error"
            self.checkpoints.save(state)
            raise

    async def resume(self, task_id: str) -> TaskOutcome:
        state = self.checkpoints.load(task_id)
        if state is None:
            raise ValueError(f"unknown task id: {task_id}")
        if state.status == TaskStatus.COMPLETED:
            raise ValueError(f"task is already complete: {task_id}")
        return await self.run(state.goal, task_id=task_id, resume=True, ensure_index=False)

    @staticmethod
    def _configured_model(settings: Settings) -> str | Model:
        if settings.llm_base_url and settings.agent_api_key:
            client = AsyncOpenAI(
                api_key=settings.agent_api_key.get_secret_value(),
                base_url=settings.llm_base_url,
                timeout=settings.llm_timeout_seconds,
            )
            return OpenAICompatibleChatModel(settings.model, client)
        return settings.model

    def _has_model_credentials(self) -> bool:
        if self.settings.llm_base_url:
            return self.settings.agent_api_key is not None
        return self.settings.openai_api_key is not None

    def _load_or_create_state(self, goal: str, task_id: str | None, resume: bool) -> TaskState:
        if resume:
            if not task_id:
                raise ValueError("task_id is required when resuming")
            existing = self.checkpoints.load(task_id)
            if existing is None:
                raise ValueError(f"unknown task id: {task_id}")
            if existing.goal != goal:
                raise ValueError("resume goal does not match checkpoint goal")
            return existing
        return TaskState(
            task_id=task_id or uuid4().hex[:12],
            goal=goal,
            plan=[
                "Load project memory and repository context",
                "Explore with keyword and semantic retrieval",
                "Implement a focused change",
                "Run tests and repair failures",
                "Review independently and run final tests",
            ],
        )

    async def _build_planning_prompt(self, runtime: RuntimeState) -> str:
        runtime.recalled_memories = await self.memory_lifecycle.retrieve(
            runtime.memory_scope,
            runtime.task.goal,
            limit=self.settings.rag_top_k,
        )
        items = [
            ContextItem(
                source=ContextSource.TASK,
                content=runtime.task.goal,
                priority=100,
                critical=True,
            ),
            ContextItem(
                source=ContextSource.STATE,
                content=self._task_state_context(runtime.task),
                priority=95,
                critical=True,
            ),
        ]
        items.extend(
            ContextItem(source=ContextSource.MEMORY, content=item.content, priority=70)
            for item in runtime.recalled_memories
        )
        return await self._render_context(runtime, items, "planning_context_budget")

    async def _build_explorer_prompt(self, runtime: RuntimeState) -> str:
        related = await self.repository_index.search(
            runtime.task.goal, top_k=self.settings.rag_top_k
        )
        items = [
            ContextItem(
                source=ContextSource.TASK,
                content=runtime.task.goal,
                priority=100,
                critical=True,
            ),
            ContextItem(
                source=ContextSource.STATE,
                content=(
                    self._task_state_context(runtime.task)
                    + "\nInitial Main plan:\n"
                    + runtime.initial_plan
                ),
                priority=95,
                critical=True,
            ),
        ]
        items.extend(
            ContextItem(source=ContextSource.MEMORY, content=item.content, priority=70)
            for item in runtime.recalled_memories
        )
        items.extend(
            ContextItem(
                source=ContextSource.RAG,
                content=(f"{item.path}:{item.start_line}-{item.end_line}\n{item.content}"),
                priority=75,
            )
            for item in related
        )
        return await self._render_context(runtime, items, "explorer_context_budget")

    async def _build_execution_prompt(self, runtime: RuntimeState) -> str:
        if runtime.exploration is None:
            raise RuntimeError("mandatory Explorer did not produce a report")
        items = [
            ContextItem(
                source=ContextSource.EXPLORER,
                content=runtime.exploration.model_dump_json(indent=2),
                priority=100,
                critical=True,
            )
        ]
        return await self._render_context(runtime, items, "execution_context_budget")

    async def _build_review_prompt(self, runtime: RuntimeState) -> str:
        items = [
            ContextItem(
                source=ContextSource.TASK,
                content=runtime.task.goal,
                priority=100,
                critical=True,
            ),
            ContextItem(
                source=ContextSource.STATE,
                content=self._task_state_context(runtime.task),
                priority=95,
                critical=True,
            ),
        ]
        if runtime.tests:
            latest_test = runtime.tests[-1]
            items.append(
                ContextItem(
                    source=ContextSource.TEST,
                    content=latest_test.model_dump_json(indent=2),
                    priority=90,
                    critical=not latest_test.passed,
                )
            )
        return await self._render_context(runtime, items, "review_context_budget")

    async def _render_context(
        self,
        runtime: RuntimeState,
        items: list[ContextItem],
        trace_name: str,
    ) -> str:

        compactor = Agent(
            name="Context Compactor",
            instructions=COMPACTION_PROMPT,
            model=self.model,
        )

        async def summarize(payload: str) -> str:
            result = await Runner.run(compactor, payload, max_turns=2)
            return str(result.final_output)

        prompt = await self.context.build(items, runtime.task, summarizer=summarize)
        runtime.context_reports.append(self.context.last_report)
        if runtime.tracer is not None:
            report = self.context.last_report
            with runtime.tracer.span(
                "context",
                trace_name,
                initial_tokens=report.initial_tokens,
                final_tokens=report.final_tokens,
                budget_tokens=report.budget_tokens,
                category_tokens={
                    source.value: count for source, count in report.category_tokens.items()
                },
                reduction_trace=report.reduction_trace,
            ):
                pass
        self.checkpoints.save(runtime.task)
        return prompt

    @staticmethod
    def _task_state_context(state: TaskState) -> str:
        return json.dumps(
            {
                "goal": state.goal,
                "plan": state.plan,
                "current_step": state.current_step,
                "explored_files": state.explored_files,
                "changed_files": state.changed_files,
                "test_status": state.test_status,
                "retry_count": state.retry_count,
                "review_status": state.review_status,
            },
            ensure_ascii=False,
        )

    async def _run_isolated(self, agent: Agent, prompt: str, runtime: RuntimeState):
        """Run a SubAgent without a session so it cannot inherit Main history."""

        return await Runner.run(
            agent,
            prompt,
            max_turns=min(15, self.settings.max_turns),
            hooks=LocalRunHooks(runtime.tracer, self.token_counter),
            context=runtime,
        )

    async def _run_main(
        self,
        main_agent: Agent,
        prompt: str,
        runtime: RuntimeState,
        trace_id: str,
        *,
        max_turns: int | None = None,
    ):
        session_id = f"{runtime.memory_scope}:{runtime.task.task_id}"
        base_session = PostgresSession(session_id, self.database)
        if self.settings.llm_base_url:
            session = base_session
        else:
            session = OpenAIResponsesCompactionSession(
                session_id,
                base_session,
                model=self.settings.model,
                should_trigger_compaction=lambda payload: (
                    self.token_counter.count(json.dumps(payload, ensure_ascii=False, default=str))
                    >= self.settings.sdk_compaction_threshold_tokens
                ),
            )

        def filter_model_input(call_data):
            live_items = [
                *runtime.context_items,
                ContextItem(
                    source=ContextSource.STATE,
                    content=self._task_state_context(runtime.task),
                    priority=95,
                    critical=True,
                ),
            ]
            filtered, report = self.context.filter_model_input(call_data.model_data, live_items)
            runtime.context_reports.append(report)
            if report.pruned_items:
                with runtime.tracer.span(
                    "context",
                    "model_input_pruned",
                    initial_tokens=report.initial_tokens,
                    final_tokens=report.final_tokens,
                    budget_tokens=report.budget_tokens,
                    pruned_items=report.pruned_items,
                    category_tokens={
                        source.value: count for source, count in report.category_tokens.items()
                    },
                    reduction_trace=report.reduction_trace,
                ):
                    pass
            return filtered

        sdk_tracing_enabled = (
            self.settings.tracing_enabled
            and self.settings.openai_api_key is not None
            and not self.settings.llm_base_url
        )
        run_config = RunConfig(
            workflow_name="RepoPilot coding task",
            trace_id=trace_id,
            group_id=runtime.task.task_id,
            tracing_disabled=not sdk_tracing_enabled,
            trace_include_sensitive_data=False,
            trace_metadata={
                "task_id": runtime.task.task_id,
                "workspace": self.settings.workspace.name,
            },
            call_model_input_filter=filter_model_input,
        )
        with trace(
            "RepoPilot coding task",
            trace_id=trace_id,
            group_id=runtime.task.task_id,
            metadata={"task_id": runtime.task.task_id},
            disabled=not sdk_tracing_enabled,
        ):
            return await Runner.run(
                main_agent,
                prompt,
                max_turns=max_turns or self.settings.max_turns,
                hooks=LocalRunHooks(runtime.tracer, self.token_counter),
                run_config=run_config,
                session=session,
                context=runtime,
            )

    async def _extract_memories(
        self,
        runtime: RuntimeState,
        summary: str,
        tracer: LocalTracer,
        *,
        completed: bool,
    ) -> MemoryExtractionReport:
        extractor = Agent(
            name="Memory Extractor",
            instructions=MEMORY_EXTRACTION_PROMPT,
            model=self.model,
            output_type=MemoryExtractionReport,
        )
        payload = json.dumps(
            {
                "user_task": runtime.task.goal,
                "task_status": "completed" if completed else "failed",
                "result_summary": summary,
                "changed_files": runtime.task.changed_files,
                "tests": [
                    {
                        "command": test.command,
                        "passed": test.passed,
                        "exit_code": test.exit_code,
                    }
                    for test in runtime.tests
                ],
                "exploration": (
                    runtime.exploration.model_dump() if runtime.exploration is not None else None
                ),
                "review": runtime.review.model_dump() if runtime.review is not None else None,
            },
            ensure_ascii=False,
        )
        result = await Runner.run(extractor, payload, max_turns=2)
        self._record_usage(tracer, result.raw_responses)
        extracted = result.final_output_as(MemoryExtractionReport)
        return self._filter_extracted_memories(runtime.task.goal, extracted, completed=completed)

    @staticmethod
    def _filter_extracted_memories(
        user_task: str,
        report: MemoryExtractionReport,
        *,
        completed: bool,
    ) -> MemoryExtractionReport:
        """Apply deterministic provenance and outcome gates after model extraction."""

        workflow_markers = (
            "system prompt",
            "系统prompt",
            "系统提示",
            "explorer subagent",
            "reviewer subagent",
            "main agent",
            "agent workflow",
            "必须经过reviewer",
            "必须调用reviewer",
        )
        normalized_task = " ".join(user_task.casefold().split())
        accepted: list[ExtractedMemory] = []
        seen: set[tuple[MemoryKind, str]] = set()
        for item in sorted(report.items, key=lambda candidate: candidate.importance, reverse=True):
            if not item.evidence.strip():
                continue
            combined = f"{item.topic} {item.content} {item.evidence}".casefold()
            if any(marker in combined for marker in workflow_markers):
                continue
            if item.kind == MemoryKind.USER_PREFERENCE:
                evidence = " ".join(item.evidence.casefold().split())
                if (
                    item.source != "user_task"
                    or len(evidence) < 4
                    or evidence not in normalized_task
                ):
                    continue
            elif item.kind == MemoryKind.PROJECT_CONSTRAINT:
                if item.source != "project_evidence":
                    continue
            elif item.kind == MemoryKind.REUSABLE_EXPERIENCE:
                if not completed or item.source != "validated_result":
                    continue
            elif (item.source == "validated_result" and not completed) or item.source not in {
                "project_evidence",
                "validated_result",
            }:
                continue

            key = (item.kind, " ".join(item.content.casefold().split()))
            if key in seen:
                continue
            seen.add(key)
            accepted.append(item)
            if len(accepted) == 3:
                break
        return MemoryExtractionReport(items=accepted)

    def _record_usage(self, tracer: LocalTracer, responses: list[object]) -> None:
        input_tokens = sum(
            int(getattr(getattr(response, "usage", None), "input_tokens", 0) or 0)
            for response in responses
        )
        output_tokens = sum(
            int(getattr(getattr(response, "usage", None), "output_tokens", 0) or 0)
            for response in responses
        )
        with tracer.span(
            "llm",
            "usage",
            calls=len(responses),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ):
            pass


def workspace_settings(workspace: Path, **overrides: object) -> Settings:
    return Settings(workspace=workspace, **overrides)
