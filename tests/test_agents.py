from pathlib import Path
from typing import Any

import pytest
from agents import Runner
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from repopilot.agents.factory import AgentFactory, RuntimeState
from repopilot.application.workflow import RepoPilot
from repopilot.core.config import Settings
from repopilot.core.models import TaskState, TaskStatus
from repopilot.infrastructure.postgres import CheckpointStore, MemoryStore, TraceStore
from repopilot.infrastructure.tracing import LocalTracer
from repopilot.repository.index import RepositoryIndex, repository_scope
from repopilot.repository.tools import WorkspaceTools


def _message(text: str, identifier: str) -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseOutputMessage(
                id=identifier,
                content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
        response_id=identifier,
    )


def _tool(name: str, arguments: str, identifier: str) -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseFunctionToolCall(
                arguments=arguments,
                call_id=f"call_{identifier}",
                name=name,
                type="function_call",
                id=identifier,
                status="completed",
            )
        ],
        usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
        response_id=identifier,
    )


class WorkflowModel(Model):
    """Deterministic SDK model used to exercise the real nested agent/tool loop."""

    def __init__(self) -> None:
        self.main_turn = 0
        self.review_turn = 0

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: Any,
        output_schema: Any,
        handoffs: Any,
        tracing: Any,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> ModelResponse:
        instructions = system_instructions or ""
        if "Compress the supplied long-running coding task context" in instructions:
            return _message(
                "Preserve the task, repository constraints, failing diagnostics, "
                "current edits, and latest test/review state.",
                "context_compaction",
            )
        if "Memory Extractor" in instructions:
            return _message('{"items":[]}', "memory_extraction")
        if "initial planning phase" in instructions:
            return _message(
                "Inspect the reported root cause, make the smallest fix, and run pytest.",
                "main_plan",
            )
        if "Explorer SubAgent" in instructions:
            return _message(
                '{"relevant_files":["app.py"],"call_chain":[],"findings":'
                '["return value is wrong"],"potential_root_causes":["bad literal"],'
                '"related_tests":["test_app.py"],"suggested_next_steps":["replace literal"]}',
                "explorer_message",
            )
        if "Reviewer SubAgent" in instructions:
            self.review_turn += 1
            if self.review_turn == 1:
                return _message(
                    '{"summary":"one boundary issue remains","approved":false,'
                    '"findings":[{"severity":"low","file":"app.py","line":1,'
                    '"problem":"public function lacks a return type",'
                    '"recommendation":"add -> str"}],"missing_tests":[]}',
                    "reviewer_changes",
                )
            return _message(
                '{"summary":"correct and tested","approved":true,"findings":[],"missing_tests":[]}',
                "reviewer_approved",
            )
        self.main_turn += 1
        if self.main_turn == 1:
            return _tool("run_tests", "{}", "tests_fail")
        if self.main_turn == 2:
            return _tool(
                "replace_text",
                '{"path":"app.py","old":"broken","new":"fixed","expected_occurrences":1}',
                "edit",
            )
        if self.main_turn == 3:
            return _tool("run_tests", "{}", "tests_pass")
        if self.main_turn == 4:
            return _message("Initial fix implemented and tested.", "main_initial_done")
        if self.main_turn == 5:
            return _tool(
                "replace_text",
                '{"path":"app.py","old":"def value():","new":"def value() -> str:",'
                '"expected_occurrences":1}',
                "review_fix",
            )
        if self.main_turn == 6:
            return _tool("run_tests", "{}", "final_tests")
        return _message("Implemented, tested, and independently reviewed.", "main_done")

    async def stream_response(self, *args: Any, **kwargs: Any):
        if False:
            yield None


class ConstraintAwareWorkflowModel(WorkflowModel):
    def __init__(self, *, extract_constraint: bool, expect_recall: bool) -> None:
        super().__init__()
        self.extract_constraint = extract_constraint
        self.expect_recall = expect_recall
        self.recalled_constraint = False

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        instructions = kwargs.get("system_instructions")
        if instructions is None and args:
            instructions = args[0]
        input_value = kwargs.get("input")
        if input_value is None and len(args) > 1:
            input_value = args[1]
        instructions = instructions or ""
        if "Memory Extractor" in instructions:
            if self.extract_constraint:
                return _message(
                    '{"items":[{"kind":"user_preference",'
                    '"topic":"third-party dependencies",'
                    '"content":"不要增加新的第三方依赖",'
                    '"polarity":"forbid","importance":1.0,"source":"user_task",'
                    '"evidence":"不要增加新的第三方依赖"}]}',
                    "constraint_extraction_with_provenance",
                )
            return _message('{"items":[]}', "empty_extraction_with_provenance")
        if "Memory Extractor" in instructions:
            if self.extract_constraint:
                return _message(
                    '{"items":[{"kind":"project_constraint","topic":"第三方依赖",'
                    '"content":"不要增加新的第三方依赖","polarity":"forbid",'
                    '"importance":1.0}]}',
                    "constraint_extraction",
                )
            return _message('{"items":[]}', "empty_extraction")
        if "Main Coding Agent" in instructions and self.main_turn == 0:
            self.recalled_constraint = "不要增加新的第三方依赖" in str(input_value)
            self.recalled_constraint = "不要增加新的第三方依赖" in str(input_value)
            if self.expect_recall and not self.recalled_constraint:
                return _tool(
                    "create_file",
                    '{"path":"requirements.txt","content":"new-dependency"}',
                    "constraint_missed",
                )
        return await super().get_response(*args, **kwargs)


class LongWorkflowModel(WorkflowModel):
    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        instructions = kwargs.get("system_instructions")
        if instructions is None and args:
            instructions = args[0]
        instructions = instructions or ""
        if any(
            role in instructions
            for role in (
                "Compress the supplied long-running coding task context",
                "initial planning phase",
                "Explorer SubAgent",
                "Reviewer SubAgent",
                "Memory Extractor",
            )
        ):
            return await super().get_response(*args, **kwargs)
        self.main_turn += 1
        responses = {
            1: _tool("search_code", '{"query":"value","path":"."}', "long_search_one"),
            2: _tool("retrieve_repository", '{"query":"wrong value implementation"}', "long_rag"),
            3: _tool("search_code", '{"query":"return","path":"."}', "long_search_two"),
            4: _tool("run_tests", "{}", "long_tests_fail"),
            5: _tool(
                "replace_text",
                '{"path":"app.py","old":"broken","new":"fixed","expected_occurrences":1}',
                "long_edit",
            ),
            6: _tool("run_tests", "{}", "long_tests_pass"),
            7: _message("Initial fix implemented and tested.", "long_initial_done"),
            8: _tool(
                "replace_text",
                '{"path":"app.py","old":"def value():","new":"def value() -> str:",'
                '"expected_occurrences":1}',
                "long_review_fix",
            ),
            9: _tool("run_tests", "{}", "long_final_tests"),
        }
        return responses.get(
            self.main_turn,
            _message("Implemented, tested, and independently reviewed.", "long_done"),
        )


class McpWorkflowModel(WorkflowModel):
    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        instructions = kwargs.get("system_instructions")
        if instructions is None and args:
            instructions = args[0]
        instructions = instructions or ""
        if any(
            role in instructions
            for role in (
                "Compress the supplied long-running coding task context",
                "initial planning phase",
                "Explorer SubAgent",
                "Reviewer SubAgent",
                "Memory Extractor",
            )
        ):
            return await super().get_response(*args, **kwargs)
        self.main_turn += 1
        responses = {
            1: _tool("get_issue", '{"issue_number":101}', "mcp_issue"),
            2: _tool("run_tests", "{}", "mcp_tests_fail"),
            3: _tool(
                "replace_text",
                '{"path":"app.py","old":"broken","new":"fixed","expected_occurrences":1}',
                "mcp_edit",
            ),
            4: _tool("run_tests", "{}", "mcp_tests_pass"),
            5: _message("Initial MCP fix is tested.", "mcp_initial_done"),
            6: _tool(
                "replace_text",
                '{"path":"app.py","old":"def value():","new":"def value() -> str:",'
                '"expected_occurrences":1}',
                "mcp_review_fix",
            ),
            7: _tool("run_tests", "{}", "mcp_final_tests"),
        }
        return responses.get(
            self.main_turn,
            _message("Read issue through MCP, fixed, tested, and reviewed.", "mcp_done"),
        )


class UnverifiedWorkflowModel(WorkflowModel):
    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        instructions = kwargs.get("system_instructions")
        if instructions is None and args:
            instructions = args[0]
        instructions = instructions or ""
        if any(
            role in instructions
            for role in (
                "Compress the supplied long-running coding task context",
                "initial planning phase",
                "Explorer SubAgent",
                "Reviewer SubAgent",
                "Memory Extractor",
            )
        ):
            return await super().get_response(*args, **kwargs)
        self.main_turn += 1
        return _message("No verified change was produced.", f"unverified_{self.main_turn}")


def test_agent_factory_enforces_three_roles(
    tmp_path: Path, database, vector_store, fake_embeddings, fake_reranker
) -> None:
    settings = Settings(
        workspace=tmp_path,
        embedding_dimensions=4,
        chunk_lines=20,
        chunk_overlap_lines=5,
        _env_file=None,
    )
    tools = WorkspaceTools(tmp_path)
    index = RepositoryIndex(
        vector_store,
        tools,
        fake_embeddings,
        fake_reranker,
        chunk_lines=20,
        overlap_lines=5,
    )
    runtime = RuntimeState(TaskState(task_id="task", goal="fix login"), repository_scope(tmp_path))
    factory = AgentFactory(
        settings,
        tools,
        index,
        MemoryStore(database),
        CheckpointStore(database, repository_scope(tmp_path)),
    )

    main = factory.build_main(runtime)
    explorer = factory.build_explorer(runtime)
    reviewer = factory.build_reviewer(runtime)
    tool_names = {tool.name for tool in main.tools}
    explorer_tool_names = {tool.name for tool in explorer.tools}
    reviewer_tool_names = {tool.name for tool in reviewer.tools}

    assert main.name == "Main Coding Agent"
    assert {"create_file", "replace_text", "run_tests"} <= tool_names
    assert {"explore_repository", "review_changes", "save_memory"}.isdisjoint(tool_names)
    assert {"list_files", "read_file", "search_code", "retrieve_repository"} <= (
        explorer_tool_names & reviewer_tool_names
    )
    assert {"create_file", "replace_text", "run_tests"}.isdisjoint(explorer_tool_names)
    assert {"create_file", "replace_text", "run_tests"}.isdisjoint(reviewer_tool_names)
    assert "write_file" not in tool_names


@pytest.mark.asyncio
async def test_main_tool_history_is_not_duplicated_into_live_context(
    tmp_path: Path, database, vector_store, fake_embeddings, fake_reranker
) -> None:
    (tmp_path / "app.py").write_text("def value():\n    return 'broken'\n", encoding="utf-8")
    (tmp_path / "test_app.py").write_text(
        "from app import value\n\ndef test_value():\n    assert value() == 'fixed'\n",
        encoding="utf-8",
    )
    settings = Settings(
        workspace=tmp_path,
        embedding_dimensions=4,
        chunk_lines=20,
        chunk_overlap_lines=5,
        tracing_enabled=False,
        context_budget_tokens=1_000,
        compaction_threshold_tokens=1_000,
        _env_file=None,
    )
    tools = WorkspaceTools(tmp_path)
    index = RepositoryIndex(
        vector_store,
        tools,
        fake_embeddings,
        fake_reranker,
        chunk_lines=20,
        overlap_lines=5,
    )
    await index.rebuild()
    runtime = RuntimeState(
        TaskState(task_id="loop", goal="fix the value"),
        repository_scope(tmp_path),
        tracer=LocalTracer(TraceStore(database, repository_scope(tmp_path)), "trace_test", "loop"),
    )
    factory = AgentFactory(
        settings,
        tools,
        index,
        MemoryStore(database),
        CheckpointStore(database, repository_scope(tmp_path)),
        model=WorkflowModel(),
    )

    result = await Runner.run(factory.build_main(runtime), "fix the value", max_turns=10)

    assert result.final_output == "Initial fix implemented and tested."
    assert (tmp_path / "app.py").read_text(encoding="utf-8").endswith("'fixed'\n")
    assert [test.passed for test in runtime.tests] == [False, True]
    assert runtime.task.retry_count == 1
    assert runtime.task.changed_files == ["app.py"]
    assert runtime.context_items == []
    categories = {
        event.category
        for event in TraceStore(database, repository_scope(tmp_path)).list_for_task("loop")
    }
    assert {"tool", "test", "rag"} <= categories


@pytest.mark.asyncio
async def test_application_workflow_completes_after_fail_repair_review(
    tmp_path: Path,
    database,
    vector_store,
    fake_embeddings,
    fake_reranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-only")
    (tmp_path / "app.py").write_text("def value():\n    return 'broken'\n", encoding="utf-8")
    (tmp_path / "test_app.py").write_text(
        "from app import value\n\ndef test_value():\n    assert value() == 'fixed'\n",
        encoding="utf-8",
    )
    settings = Settings(
        workspace=tmp_path,
        embedding_dimensions=4,
        chunk_lines=20,
        chunk_overlap_lines=5,
        tracing_enabled=False,
        context_budget_tokens=1_000,
        compaction_threshold_tokens=1_000,
        _env_file=None,
    )
    index = RepositoryIndex(vector_store, WorkspaceTools(tmp_path), fake_embeddings, fake_reranker)
    pilot = RepoPilot(settings, repository_index=index, model=WorkflowModel())

    outcome = await pilot.run("fix the value", task_id="workflow", ensure_index=True)

    assert outcome.status == TaskStatus.COMPLETED
    assert [test.passed for test in outcome.tests] == [False, True, True]
    assert outcome.review is not None and outcome.review.approved
    checkpoint = pilot.checkpoints.load("workflow")
    assert checkpoint is not None and checkpoint.status == TaskStatus.COMPLETED
    events = pilot.trace_store.list_for_task("workflow")
    assert any(event.category == "agent" and event.name == "main_planning" for event in events)
    assert any(event.category == "agent" and event.name == "explorer" for event in events)
    assert any(event.category == "agent" and event.name == "reviewer" for event in events)
    assert any(event.category == "agent" and event.name == "main_execution" for event in events)
    agent_names = [event.name for event in events if event.category == "agent"]
    assert agent_names.index("main_planning") < agent_names.index("explorer")
    assert agent_names.index("explorer") < agent_names.index("main_execution")
    assert agent_names.index("main_execution") < agent_names.index("reviewer")
    assert any(
        event.category == "application" and event.details["status"] == "completed"
        for event in events
    )
    assert any(event.category == "llm" and event.details["total_tokens"] > 0 for event in events)
    assert any(
        event.category == "rag"
        and event.name == "incremental_index"
        and event.details.get("path") == "app.py"
        for event in events
    )
    assert any(
        event.category == "context"
        and event.name == "model_input_pruned"
        and event.details["final_tokens"] <= 1_000
        for event in events
    )


@pytest.mark.asyncio
async def test_reviewer_is_mandatory_even_when_tests_never_pass(
    tmp_path: Path,
    vector_store,
    fake_embeddings,
    fake_reranker,
) -> None:
    settings = Settings(
        workspace=tmp_path,
        embedding_dimensions=4,
        tracing_enabled=False,
        _env_file=None,
    )
    index = RepositoryIndex(vector_store, WorkspaceTools(tmp_path), fake_embeddings, fake_reranker)
    model = UnverifiedWorkflowModel()
    pilot = RepoPilot(settings, repository_index=index, model=model)

    outcome = await pilot.run("make an unverified change", task_id="unverified")

    assert outcome.status == TaskStatus.FAILED
    assert outcome.review is not None and not outcome.review.approved
    assert model.review_turn == 1
    assert any(
        event.category == "agent" and event.name == "reviewer"
        for event in pilot.trace_store.list_for_task("unverified")
    )


@pytest.mark.asyncio
async def test_agent_calls_real_stdio_mcp_tool_and_records_local_trace(
    tmp_path: Path,
    vector_store,
    fake_embeddings,
    fake_reranker,
) -> None:
    (tmp_path / "app.py").write_text("def value():\n    return 'broken'\n", encoding="utf-8")
    (tmp_path / "test_app.py").write_text(
        "from app import value\n\ndef test_value():\n    assert value() == 'fixed'\n",
        encoding="utf-8",
    )
    settings = Settings(
        workspace=tmp_path,
        embedding_dimensions=4,
        chunk_lines=20,
        chunk_overlap_lines=5,
        issue_mcp_enabled=True,
        tracing_enabled=False,
        _env_file=None,
    )
    index = RepositoryIndex(vector_store, WorkspaceTools(tmp_path), fake_embeddings, fake_reranker)
    pilot = RepoPilot(settings, repository_index=index, model=McpWorkflowModel())

    outcome = await pilot.run(
        "Read issue 101 through MCP, then fix and test it.",
        task_id="mcp-workflow",
    )

    assert outcome.status == TaskStatus.COMPLETED
    mcp_events = [
        event
        for event in pilot.trace_store.list_for_task("mcp-workflow")
        if event.category == "mcp" and event.name == "get_issue"
    ]
    assert len(mcp_events) == 1
    assert mcp_events[0].details["server"] == "Issue MCP"
    assert "Unknown login user becomes HTTP 500" in mcp_events[0].details["result_excerpt"]


@pytest.mark.asyncio
async def test_automatic_memory_is_recalled_in_a_new_session_and_affects_execution(
    tmp_path: Path,
    vector_store,
    fake_embeddings,
    fake_reranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-only")
    app_path = tmp_path / "app.py"
    test_path = tmp_path / "test_app.py"
    app_path.write_text("def value():\n    return 'broken'\n", encoding="utf-8")
    test_path.write_text(
        "from app import value\n\ndef test_value():\n    assert value() == 'fixed'\n",
        encoding="utf-8",
    )
    settings = Settings(
        workspace=tmp_path,
        embedding_dimensions=4,
        chunk_lines=20,
        chunk_overlap_lines=5,
        tracing_enabled=False,
        _env_file=None,
    )
    index = RepositoryIndex(vector_store, WorkspaceTools(tmp_path), fake_embeddings, fake_reranker)
    first_model = ConstraintAwareWorkflowModel(extract_constraint=True, expect_recall=False)
    first_pilot = RepoPilot(settings, repository_index=index, model=first_model)

    first = await first_pilot.run(
        "修复 value 并且这个项目以后都不要增加新的第三方依赖。",
        task_id="memory-first",
    )

    assert first.status == TaskStatus.COMPLETED
    first_events = first_pilot.trace_store.list_for_task("memory-first")
    assert any(
        event.category == "memory"
        and event.name == "extract"
        and event.details["actions"] == ["created"]
        for event in first_events
    )
    recalled = first_pilot.memory.retrieve(repository_scope(tmp_path), "帮我实现X")
    assert [item.content for item in recalled] == ["不要增加新的第三方依赖"]

    app_path.write_text("def value():\n    return 'broken'\n", encoding="utf-8")
    second_model = ConstraintAwareWorkflowModel(extract_constraint=False, expect_recall=True)
    second_pilot = RepoPilot(settings, repository_index=index, model=second_model)
    second = await second_pilot.run("帮我实现X。", task_id="memory-second")

    assert second.status == TaskStatus.COMPLETED
    assert second_model.recalled_constraint
    assert not (tmp_path / "requirements.txt").exists()
    assert second.changed_files == ["app.py"]


@pytest.mark.asyncio
async def test_long_task_prunes_context_and_still_completes(
    tmp_path: Path,
    vector_store,
    fake_embeddings,
    fake_reranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-only")
    (tmp_path / "app.py").write_text("def value():\n    return 'broken'\n", encoding="utf-8")
    (tmp_path / "test_app.py").write_text(
        "from app import value\n\ndef test_value():\n    assert value() == 'fixed'\n",
        encoding="utf-8",
    )
    (tmp_path / "noise.txt").write_text(
        "\n".join(f"value search noise {index}" for index in range(1_000)),
        encoding="utf-8",
    )
    settings = Settings(
        workspace=tmp_path,
        embedding_dimensions=4,
        chunk_lines=20,
        chunk_overlap_lines=5,
        max_tool_output_tokens=128,
        context_budget_tokens=1_000,
        compaction_threshold_tokens=1_000,
        tracing_enabled=False,
        _env_file=None,
    )
    index = RepositoryIndex(vector_store, WorkspaceTools(tmp_path), fake_embeddings, fake_reranker)
    pilot = RepoPilot(settings, repository_index=index, model=LongWorkflowModel())

    outcome = await pilot.run("fix the value through a long investigation", task_id="long-task")

    assert outcome.status == TaskStatus.COMPLETED
    assert [test.passed for test in outcome.tests] == [False, True, True]
    events = pilot.trace_store.list_for_task("long-task")
    assert sum(event.category == "tool" and event.name == "search_code" for event in events) == 2
    assert any(
        event.category == "tool" and event.name == "search_code" and event.details["truncated"]
        for event in events
    )
    pruning_events = [
        event
        for event in events
        if event.category == "context" and event.name == "model_input_pruned"
    ]
    assert pruning_events
    assert all(event.details["final_tokens"] <= 1_000 for event in pruning_events)
    assert all(set(event.details["category_tokens"]) <= {"state"} for event in pruning_events)
