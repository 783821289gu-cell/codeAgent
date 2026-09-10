from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from repopilot.application.workflow import RepoPilot
from repopilot.core.config import Settings
from repopilot.core.models import TaskOutcome, TaskStatus, TestResult
from repopilot.repository.tools import WorkspaceTools


class DemoReport(BaseModel):
    run_id: str
    success: bool
    workspace: str
    task_id: str
    outcome: TaskOutcome
    verification_test: TestResult
    mcp_tools_called: list[str] = Field(default_factory=list)
    report_file: str
    trace_file: str
    diff_file: str
    test_output_file: str


class DemoRunner:
    """Run one reproducible coding scenario and retain its evidence."""

    def __init__(
        self,
        baseline_repo: Path,
        output_dir: Path,
        settings_factory: Callable[[Path], Settings],
    ) -> None:
        self.baseline_repo = baseline_repo.resolve()
        self.output_dir = output_dir.resolve()
        self.settings_factory = settings_factory

    async def run(
        self,
        goal: str,
        *,
        name: str,
        required_mcp_tool: str | None = None,
    ) -> DemoReport:
        if not self.baseline_repo.is_dir():
            raise ValueError(f"demo baseline does not exist: {self.baseline_repo}")
        run_id = name + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
        run_dir = self.output_dir / run_id
        workspace = run_dir / "workspace"
        shutil.copytree(
            self.baseline_repo,
            workspace,
            ignore=shutil.ignore_patterns(".repopilot", ".pytest_cache", "__pycache__"),
        )
        settings = self.settings_factory(workspace)
        pilot = RepoPilot(settings)
        task_id = run_id
        outcome = await pilot.run(goal, task_id=task_id)
        verification = WorkspaceTools(workspace).run_tests(f'"{sys.executable}" -m pytest -q')
        diff_result = WorkspaceTools(workspace).git_diff()
        events = pilot.trace_store.list_for_task(task_id)
        mcp_tools = [event.name for event in events if event.category == "mcp"]
        success = (
            outcome.status == TaskStatus.COMPLETED
            and verification.passed
            and outcome.review is not None
            and outcome.review.approved
            and (required_mcp_tool is None or required_mcp_tool in mcp_tools)
        )
        report_file = run_dir / "report.json"
        trace_file = run_dir / "trace.json"
        diff_file = run_dir / "git.diff"
        test_output_file = run_dir / "test-output.txt"
        report = DemoReport(
            run_id=run_id,
            success=success,
            workspace=str(workspace),
            task_id=task_id,
            outcome=outcome,
            verification_test=verification,
            mcp_tools_called=mcp_tools,
            report_file=str(report_file),
            trace_file=str(trace_file),
            diff_file=str(diff_file),
            test_output_file=str(test_output_file),
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        trace_file.write_text(
            json.dumps(
                [event.model_dump(mode="json") for event in events],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        diff_file.write_text(diff_result.data or diff_result.error or "", encoding="utf-8")
        test_output_file.write_text(verification.output, encoding="utf-8")
        return report
