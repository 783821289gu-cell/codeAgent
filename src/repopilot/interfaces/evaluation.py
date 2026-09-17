from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, Field, field_validator

from repopilot.core.models import TaskOutcome, TaskStatus, TraceEvent
from repopilot.infrastructure.postgres import CheckpointStore, TraceStore
from repopilot.interfaces.workspace import prepare_workspace
from repopilot.repository.tools import WorkspaceTools

if TYPE_CHECKING:
    from repopilot.application.workflow import RepoPilot


IGNORED_EVAL_PARTS = {
    ".git",
    ".repopilot",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


class EvaluationSummary(BaseModel):
    tasks: int
    task_success_rate: float
    test_pass_rate: float
    average_agent_steps: float
    average_tool_calls: float
    total_tokens: int
    average_duration_seconds: float


class EvalCase(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    task: str = Field(min_length=1, validation_alias=AliasChoices("task", "goal"))
    baseline_repo: Path = Field(validation_alias=AliasChoices("baseline_repo", "workspace"))
    success_command: str = Field(min_length=1)
    expected_changed_files: list[str] = Field(min_length=1)
    forbidden_changes: list[str] = Field(default_factory=list)

    @field_validator("expected_changed_files", "forbidden_changes")
    @classmethod
    def normalize_paths(cls, values: list[str]) -> list[str]:
        normalized = [value.replace("\\", "/").strip() for value in values]
        if any(not value or value.startswith(("/", "../")) for value in normalized):
            raise ValueError("eval file paths must be non-empty workspace-relative paths")
        return list(dict.fromkeys(normalized))


class EvalCaseResult(BaseModel):
    id: str
    task_id: str
    success: bool
    outcome_status: TaskStatus
    success_command_passed: bool
    success_command_output: str
    changed_files: list[str] = Field(default_factory=list)
    expected_changed_files_match: bool
    forbidden_changes: list[str] = Field(default_factory=list)
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0
    trace_id: str | None = None
    artifact_dir: str
    error: str | None = None


class EvalRunSummary(BaseModel):
    run_id: str
    dataset: str
    output_dir: str
    tasks: int
    task_success_rate: float
    test_pass_rate: float
    average_turns: float
    average_tool_calls: float
    total_tokens: int
    average_latency_seconds: float
    results: list[EvalCaseResult] = Field(default_factory=list)


class Evaluator:
    """Computes small, reproducible metrics from persisted checkpoints and local traces."""

    def __init__(self, checkpoints: CheckpointStore, traces: TraceStore) -> None:
        self.checkpoints = checkpoints
        self.traces = traces

    def summarize(self) -> EvaluationSummary:
        states = self.checkpoints.list_all()
        if not states:
            return EvaluationSummary(
                tasks=0,
                task_success_rate=0,
                test_pass_rate=0,
                average_agent_steps=0,
                average_tool_calls=0,
                total_tokens=0,
                average_duration_seconds=0,
            )
        successes = 0
        tests_passed = 0
        agent_steps = 0
        tool_calls = 0
        total_tokens = 0
        durations: list[float] = []
        for state in states:
            successes += state.status == TaskStatus.COMPLETED
            tests_passed += state.test_status == "passed"
            events = self.traces.list_for_task(state.task_id)
            agent_steps += sum(event.category == "agent" for event in events)
            tool_calls += sum(event.category in {"tool", "test", "memory"} for event in events)
            total_tokens += sum(
                int(event.details.get("total_tokens", 0))
                for event in events
                if event.category == "llm"
            )
            durations.extend(
                event.duration_ms / 1_000
                for event in events
                if event.category == "application" and event.name == "task"
            )
        count = len(states)
        return EvaluationSummary(
            tasks=count,
            task_success_rate=successes / count,
            test_pass_rate=tests_passed / count,
            average_agent_steps=agent_steps / count,
            average_tool_calls=tool_calls / count,
            total_tokens=total_tokens,
            average_duration_seconds=sum(durations) / len(durations) if durations else 0,
        )


class EvalRunner:
    """Execute a small coding benchmark in isolated repository copies."""

    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        pilot_factory: Callable[[Path], RepoPilot],
        *,
        command_timeout_seconds: int = 180,
    ) -> None:
        self.dataset_path = dataset_path.resolve()
        self.output_dir = output_dir.resolve()
        self.pilot_factory = pilot_factory
        self.command_timeout_seconds = command_timeout_seconds

    async def run(self) -> EvalRunSummary:
        cases = self._load_cases()
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        results: list[EvalCaseResult] = []
        for case in cases:
            results.append(await self._run_case(case, run_dir))
        summary = self._summarize(run_id, run_dir, results)
        (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        return summary

    def _load_cases(self) -> list[EvalCase]:
        raw = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not 3 <= len(raw) <= 5:
            raise ValueError("eval dataset must contain 3 to 5 cases")
        cases = [EvalCase.model_validate(item) for item in raw]
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("eval case ids must be unique")
        return cases

    async def _run_case(self, case: EvalCase, run_dir: Path) -> EvalCaseResult:
        case_dir = run_dir / case.id
        workspace = case_dir / "workspace"
        baseline = self._resolve_baseline(case.baseline_repo)
        prepare_workspace(baseline, workspace)
        before = _repository_snapshot(workspace)
        pilot = self.pilot_factory(workspace)
        task_id = f"eval-{case.id}-{uuid4().hex[:8]}"
        started = time.perf_counter()
        error: str | None = None
        outcome: TaskOutcome
        try:
            outcome = await pilot.run(case.task, task_id=task_id)
        except Exception as exc:  # Keep the remaining benchmark cases runnable.
            error = f"{exc.__class__.__name__}: {exc}"
            outcome = TaskOutcome(
                task_id=task_id,
                status=TaskStatus.FAILED,
                summary=error,
            )
        latency = time.perf_counter() - started
        command = case.success_command.replace(
            "{python}", subprocess.list2cmdline([sys.executable])
        )
        command_result = WorkspaceTools(
            workspace,
            timeout_seconds=self.command_timeout_seconds,
        ).run_command(
            command,
            timeout_seconds=self.command_timeout_seconds,
            _allowed_absolute_paths=(sys.executable,),
        )
        after = _repository_snapshot(workspace)
        changed_files = sorted(
            path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
        )
        expected_match = set(changed_files) == set(case.expected_changed_files)
        forbidden = sorted(
            path
            for path in changed_files
            if any(fnmatch(path, pattern) for pattern in case.forbidden_changes)
        )
        events = pilot.trace_store.list_for_task(task_id)
        metrics = _trace_metrics(events)
        command_output = command_result.data or command_result.error or ""
        command_passed = command_result.ok
        success = (
            outcome.status == TaskStatus.COMPLETED
            and command_passed
            and expected_match
            and not forbidden
        )
        result = EvalCaseResult(
            id=case.id,
            task_id=task_id,
            success=success,
            outcome_status=outcome.status,
            success_command_passed=command_passed,
            success_command_output=command_output,
            changed_files=changed_files,
            expected_changed_files_match=expected_match,
            forbidden_changes=forbidden,
            latency_seconds=latency,
            trace_id=outcome.trace_id,
            artifact_dir=str(case_dir),
            error=error,
            **metrics,
        )
        (case_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        (case_dir / "trace.json").write_text(
            json.dumps(
                [event.model_dump(mode="json") for event in events],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    def _resolve_baseline(self, value: Path) -> Path:
        candidates = [
            value,
            self.dataset_path.parent / value,
            self.dataset_path.parent.parent / value,
            Path.cwd() / value,
        ]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_dir():
                return resolved
        raise ValueError(f"eval baseline repository does not exist: {value}")

    def _summarize(
        self, run_id: str, run_dir: Path, results: list[EvalCaseResult]
    ) -> EvalRunSummary:
        count = len(results)
        return EvalRunSummary(
            run_id=run_id,
            dataset=str(self.dataset_path),
            output_dir=str(run_dir),
            tasks=count,
            task_success_rate=sum(result.success for result in results) / count,
            test_pass_rate=sum(result.success_command_passed for result in results) / count,
            average_turns=sum(result.turns for result in results) / count,
            average_tool_calls=sum(result.tool_calls for result in results) / count,
            total_tokens=sum(result.total_tokens for result in results),
            average_latency_seconds=sum(result.latency_seconds for result in results) / count,
            results=results,
        )


def _repository_snapshot(workspace: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in workspace.rglob("*"):
        if not path.is_file() or any(part in IGNORED_EVAL_PARTS for part in path.parts):
            continue
        relative = path.relative_to(workspace).as_posix()
        snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _trace_metrics(events: list[TraceEvent]) -> dict[str, int]:
    llm_events = [event for event in events if event.category == "llm"]
    return {
        "turns": sum(
            int(event.details.get("model_responses", 0))
            for event in events
            if event.category == "agent" and event.name == "main"
        ),
        "tool_calls": sum(event.category in {"tool", "test", "memory"} for event in events),
        "input_tokens": sum(int(event.details.get("input_tokens", 0)) for event in llm_events),
        "output_tokens": sum(int(event.details.get("output_tokens", 0)) for event in llm_events),
        "total_tokens": sum(int(event.details.get("total_tokens", 0)) for event in llm_events),
    }
