import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repopilot.application.workflow import RepoPilot
from repopilot.core.config import Settings
from repopilot.core.models import TaskOutcome, TaskState, TaskStatus, TraceEvent
from repopilot.infrastructure.mcp import build_mcp_servers, github_read_only_filter
from repopilot.interfaces.evaluation import EvalRunner, Evaluator
from repopilot.repository.index import RepositoryIndex
from repopilot.repository.tools import WorkspaceTools


@pytest.mark.asyncio
async def test_live_run_requires_api_key(
    tmp_path: Path, database, vector_store, fake_embeddings, fake_reranker
) -> None:
    settings = Settings(
        workspace=tmp_path,
        embedding_dimensions=4,
        chunk_lines=20,
        chunk_overlap_lines=5,
        _env_file=None,
    )
    index = RepositoryIndex(vector_store, WorkspaceTools(tmp_path), fake_embeddings, fake_reranker)
    pilot = RepoPilot(settings, repository_index=index)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await pilot.run("fix login")


def test_github_mcp_is_opt_in(tmp_path: Path) -> None:
    disabled = Settings(workspace=tmp_path, _env_file=None)
    enabled = Settings(
        workspace=tmp_path,
        github_mcp_url="https://example.test/mcp",
        github_token="secret",  # type: ignore[arg-type]
        _env_file=None,
    )

    assert build_mcp_servers(disabled) == []
    servers = build_mcp_servers(enabled)
    assert len(servers) == 1
    assert servers[0].name == "GitHub MCP"


def test_local_issue_mcp_is_opt_in(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path, issue_mcp_enabled=True, _env_file=None)

    servers = build_mcp_servers(settings)

    assert len(servers) == 1
    assert servers[0].name == "Issue MCP"


@pytest.mark.parametrize(
    ("name", "allowed"),
    [
        ("get_issue", True),
        ("search_pull_requests", True),
        ("create_issue", False),
        ("merge_pull_request", False),
    ],
)
def test_github_mcp_filter_is_read_only(name: str, allowed: bool) -> None:
    tool = type("McpTool", (), {"name": name})()
    assert github_read_only_filter(None, tool) is allowed


def test_evaluation_uses_persisted_state_and_trace(
    tmp_path: Path, database, vector_store, fake_embeddings, fake_reranker
) -> None:
    settings = Settings(
        workspace=tmp_path,
        embedding_dimensions=4,
        chunk_lines=20,
        chunk_overlap_lines=5,
        _env_file=None,
    )
    index = RepositoryIndex(vector_store, WorkspaceTools(tmp_path), fake_embeddings, fake_reranker)
    pilot = RepoPilot(settings, repository_index=index)
    state = TaskState(
        task_id="done",
        goal="fix login",
        status=TaskStatus.COMPLETED,
        test_status="passed",
    )
    pilot.checkpoints.save(state)

    summary = Evaluator(pilot.checkpoints, pilot.trace_store).summarize()

    assert summary.tasks == 1
    assert summary.task_success_rate == 1
    assert summary.test_pass_rate == 1


def test_fixed_evaluation_tasks_are_well_formed() -> None:
    tasks_path = Path(__file__).parents[1] / "evals" / "tasks.json"
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))

    assert len(tasks) >= 3
    assert len({task["id"] for task in tasks}) == len(tasks)
    assert all(
        task["task"]
        and task["baseline_repo"]
        and task["success_command"]
        and task["expected_changed_files"]
        for task in tasks
    )


@pytest.mark.asyncio
async def test_eval_runner_executes_isolated_cases_and_writes_reports(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "app.py").write_text('VALUE = "before"\n', encoding="utf-8")
    (baseline / "test_app.py").write_text(
        'from app import VALUE\n\ndef test_value():\n    assert VALUE == "after"\n',
        encoding="utf-8",
    )
    dataset = tmp_path / "tasks.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": f"case-{index}",
                    "task": "change the value",
                    "baseline_repo": str(baseline),
                    "success_command": "{python} -m pytest -q",
                    "expected_changed_files": ["app.py"],
                    "forbidden_changes": ["test_app.py"],
                }
                for index in range(3)
            ]
        ),
        encoding="utf-8",
    )

    class FakeTraceStore:
        def __init__(self) -> None:
            self.events: list[TraceEvent] = []

        def list_for_task(self, task_id: str) -> list[TraceEvent]:
            return [event for event in self.events if event.task_id == task_id]

    class FakePilot:
        def __init__(self, workspace: Path) -> None:
            self.workspace = workspace
            self.trace_store = FakeTraceStore()

        async def run(self, goal: str, *, task_id: str) -> TaskOutcome:
            assert goal == "change the value"
            (self.workspace / "app.py").write_text('VALUE = "after"\n', encoding="utf-8")
            now = datetime.now(UTC)
            self.trace_store.events.extend(
                [
                    TraceEvent(
                        trace_id="trace-eval",
                        task_id=task_id,
                        category="agent",
                        name="main",
                        started_at=now,
                        duration_ms=10,
                        ok=True,
                        details={"model_responses": 3},
                    ),
                    TraceEvent(
                        trace_id="trace-eval",
                        task_id=task_id,
                        category="tool",
                        name="replace_text",
                        started_at=now,
                        duration_ms=2,
                        ok=True,
                    ),
                    TraceEvent(
                        trace_id="trace-eval",
                        task_id=task_id,
                        category="llm",
                        name="usage",
                        started_at=now,
                        duration_ms=0,
                        ok=True,
                        details={
                            "input_tokens": 20,
                            "output_tokens": 5,
                            "total_tokens": 25,
                        },
                    ),
                ]
            )
            return TaskOutcome(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                summary="done",
                changed_files=["app.py"],
                trace_id="trace-eval",
            )

    runner = EvalRunner(
        dataset,
        tmp_path / "eval-output",
        lambda workspace: FakePilot(workspace),  # type: ignore[arg-type,return-value]
    )

    summary = await runner.run()

    assert summary.tasks == 3
    assert summary.task_success_rate == 1
    assert summary.test_pass_rate == 1
    assert summary.average_turns == 3
    assert summary.average_tool_calls == 1
    assert summary.total_tokens == 75
    assert (baseline / "app.py").read_text(encoding="utf-8") == 'VALUE = "before"\n'
    assert (Path(summary.output_dir) / "summary.json").is_file()
    assert all((Path(result.artifact_dir) / "trace.json").is_file() for result in summary.results)
