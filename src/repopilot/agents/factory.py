from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from agents import Agent, RunResult, function_tool
from agents.mcp import MCPServer
from agents.models.interface import Model

from repopilot.agents.prompts import EXPLORER_PROMPT, MAIN_PROMPT, REVIEWER_PROMPT
from repopilot.core.config import Settings
from repopilot.core.models import (
    ContextBudgetReport,
    ContextItem,
    ExplorationReport,
    MemoryItem,
    ReviewReport,
    TaskState,
    TestResult,
)
from repopilot.infrastructure.postgres import CheckpointStore, MemoryStore
from repopilot.infrastructure.tracing import LocalTracer
from repopilot.knowledge.memory import MemoryLifecycle
from repopilot.repository.index import RepositoryIndex
from repopilot.repository.tools import WorkspaceTools


@dataclass
class RuntimeState:
    task: TaskState
    memory_scope: str
    initial_plan: str = ""
    recalled_memories: list[MemoryItem] = field(default_factory=list)
    tests: list[TestResult] = field(default_factory=list)
    exploration: ExplorationReport | None = None
    review: ReviewReport | None = None
    context_items: list[ContextItem] = field(default_factory=list)
    context_reports: list[ContextBudgetReport] = field(default_factory=list)
    tracer: LocalTracer | None = None


def _json(value: Any) -> str:
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    return str(value)


class AgentFactory:
    """Composes SDK agents while deterministic behavior remains in injected services."""

    def __init__(
        self,
        settings: Settings,
        tools: WorkspaceTools,
        repository_index: RepositoryIndex,
        memory: MemoryStore,
        checkpoints: CheckpointStore,
        memory_lifecycle: MemoryLifecycle | None = None,
        model: str | Model | None = None,
    ) -> None:
        self.settings = settings
        self.tools = tools
        self.repository_index = repository_index
        self.memory_lifecycle = memory_lifecycle or MemoryLifecycle(
            memory, repository_index.embeddings
        )
        self.checkpoints = checkpoints
        self.model = model or settings.model

    def build_explorer(self, runtime: RuntimeState) -> Agent:
        return Agent(
            name="Explorer",
            instructions=EXPLORER_PROMPT,
            model=self.model,
            tools=self._read_tools(runtime),
            output_type=ExplorationReport,
        )

    def build_reviewer(self, runtime: RuntimeState) -> Agent:
        return Agent(
            name="Reviewer",
            instructions=REVIEWER_PROMPT,
            model=self.model,
            tools=self._read_tools(runtime) + self._review_tools(runtime),
            output_type=ReviewReport,
        )

    def build_main(
        self, runtime: RuntimeState, mcp_servers: list[MCPServer] | None = None
    ) -> Agent:
        read_tools = self._read_tools(runtime)
        return Agent(
            name="Main Coding Agent",
            instructions=MAIN_PROMPT,
            model=self.model,
            tools=[
                *read_tools,
                *self._write_tools(runtime),
                *self._review_tools(runtime),
            ],
            mcp_servers=mcp_servers or [],
        )

    def build(self, runtime: RuntimeState, mcp_servers: list[MCPServer] | None = None) -> Agent:
        """Compatibility alias for callers that only need the Main execution agent."""

        return self.build_main(runtime, mcp_servers)

    def record_exploration(self, runtime: RuntimeState, result: RunResult) -> ExplorationReport:
        report = result.final_output_as(ExplorationReport)
        with self._span(runtime, "agent", "explorer", files=len(report.relevant_files)):
            pass
        self._record_usage(runtime, "explorer_usage", result)
        runtime.exploration = report
        runtime.task.explored_files = list(
            dict.fromkeys(runtime.task.explored_files + report.relevant_files)
        )
        runtime.task.current_step = "exploration_complete"
        self.checkpoints.save(runtime.task)
        return report

    def record_review(self, runtime: RuntimeState, result: RunResult) -> ReviewReport:
        report = result.final_output_as(ReviewReport)
        with self._span(
            runtime,
            "agent",
            "reviewer",
            approved=report.approved,
            findings=len(report.findings),
        ):
            pass
        self._record_usage(runtime, "reviewer_usage", result)
        runtime.review = report
        runtime.task.review_status = "approved" if report.approved else "changes_requested"
        runtime.task.current_step = "review_complete"
        self.checkpoints.save(runtime.task)
        return report

    def _span(self, runtime: RuntimeState, category: str, name: str, **details: Any):
        if runtime.tracer is None:
            return nullcontext({})
        return runtime.tracer.span(category, name, **details)

    def _record_usage(self, runtime: RuntimeState, name: str, result: RunResult) -> None:
        input_tokens = sum(response.usage.input_tokens for response in result.raw_responses)
        output_tokens = sum(response.usage.output_tokens for response in result.raw_responses)
        with self._span(
            runtime,
            "llm",
            name,
            calls=len(result.raw_responses),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ):
            pass

    def _read_tools(self, runtime: RuntimeState) -> list[Any]:
        local = self.tools
        repository_index = self.repository_index

        @function_tool
        def list_files(path: str = ".", limit: int = 500) -> str:
            """List repository files below a workspace-relative directory."""

            with self._span(runtime, "tool", "list_files", path=path) as details:
                result = local.list_files(path, limit)
                details.update(ok=result.ok, truncated=result.truncated)
                return _json(result)

        @function_tool
        def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
            """Read a UTF-8 file with line numbers from the bounded workspace."""

            with self._span(runtime, "tool", "read_file", path=path) as details:
                result = local.read_file(path, start_line, end_line)
                details.update(ok=result.ok, truncated=result.truncated)
                return _json(result)

        @function_tool
        def search_code(query: str, path: str = ".", regex: bool = False, limit: int = 100) -> str:
            """Search code with ripgrep; literal search is the safe default."""

            with self._span(runtime, "tool", "search_code", path=path) as details:
                result = local.search_code(query, path, regex=regex, limit=limit)
                details.update(ok=result.ok, truncated=result.truncated)
                return _json(result)

        @function_tool
        async def retrieve_repository(query: str, top_k: int = 8) -> str:
            """Run hybrid semantic/vector and FTS keyword retrieval over the repository index."""

            with self._span(runtime, "tool", "retrieve_repository", top_k=top_k) as details:
                results = await repository_index.search(query, top_k)
                details["results"] = len(results)
                rendered = "[" + ",".join(result.model_dump_json() for result in results) + "]"
                return rendered

        return [list_files, read_file, search_code, retrieve_repository]

    def _write_tools(self, runtime: RuntimeState) -> list[Any]:
        local = self.tools
        checkpoints = self.checkpoints

        def record_change(path: str) -> None:
            runtime.task.changed_files = list(dict.fromkeys([*runtime.task.changed_files, path]))
            runtime.task.current_step = "editing"
            runtime.task.test_status = "not_run"
            runtime.task.review_status = "not_run"
            runtime.review = None
            checkpoints.save(runtime.task)

        async def refresh_index(path: str) -> None:
            with self._span(runtime, "rag", "incremental_index", path=path) as details:
                details.update(await self.repository_index.sync([path]))

        @function_tool
        async def create_file(path: str, content: str) -> str:
            """Create a new UTF-8 file. Existing files are never overwritten."""

            with self._span(runtime, "tool", "create_file", path=path) as details:
                result = local.write_file(path, content)
                details["ok"] = result.ok
                if result.ok and result.data:
                    record_change(result.data)
                    await refresh_index(result.data)
                return _json(result)

        @function_tool
        async def replace_text(path: str, old: str, new: str, expected_occurrences: int = 1) -> str:
            """Replace an exact text block with stale-edit occurrence checking."""

            with self._span(runtime, "tool", "replace_text", path=path) as details:
                result = local.replace_text(
                    path, old, new, expected_occurrences=expected_occurrences
                )
                details["ok"] = result.ok
                if result.ok and result.data:
                    record_change(result.data)
                    await refresh_index(result.data)
                return _json(result)

        @function_tool
        def run_shell(command: str, timeout_seconds: int | None = None) -> str:
            """Run one policy-checked shell command in the workspace with a hard timeout."""

            with self._span(runtime, "tool", "run_shell") as details:
                result = local.run_command(command, timeout_seconds)
                details.update(ok=result.ok, exit_code=result.metadata.get("exit_code"))
                return _json(result)

        @function_tool
        def run_tests(command: str | None = None) -> str:
            """Run tests and return structured status, bounded output, and duration."""

            with self._span(runtime, "test", "run_tests", command=command) as details:
                result = local.run_tests(command)
                details.update(
                    passed=result.passed,
                    exit_code=result.exit_code,
                    duration_seconds=result.duration_seconds,
                )
                runtime.tests.append(result)
                runtime.task.test_status = "passed" if result.passed else "failed"
                runtime.task.current_step = "tests_passed" if result.passed else "repairing_tests"
                if not result.passed:
                    runtime.task.retry_count += 1
                checkpoints.save(runtime.task)
                return result.model_dump_json()

        return [create_file, replace_text, run_shell, run_tests]

    def _review_tools(self, runtime: RuntimeState) -> list[Any]:
        local = self.tools

        @function_tool
        def git_diff() -> str:
            """Return the current uncommitted Git diff."""

            with self._span(runtime, "tool", "git_diff") as details:
                result = local.git_diff()
                details.update(ok=result.ok, truncated=result.truncated)
                return _json(result)

        @function_tool
        def git_status() -> str:
            """Return concise Git workspace status."""

            with self._span(runtime, "tool", "git_status") as details:
                result = local.git_status()
                details.update(ok=result.ok, truncated=result.truncated)
                return _json(result)

        return [git_diff, git_status]
