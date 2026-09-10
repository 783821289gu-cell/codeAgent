from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from repopilot.application.workflow import RepoPilot
from repopilot.core.config import Settings
from repopilot.core.models import TaskOutcome
from repopilot.interfaces.demo import DemoRunner
from repopilot.interfaces.evaluation import EvalRunner, Evaluator
from repopilot.repository.embeddings import EmbeddingProvider
from repopilot.repository.reranker import Reranker

app = typer.Typer(no_args_is_help=True, help="RepoPilot multi-agent coding agent")
console = Console()
DEFAULT_WORKSPACE = Path.cwd()


def _settings(workspace: Path, allow_dangerous: bool = False) -> Settings:
    return Settings(
        workspace=workspace,
        allow_dangerous_commands=allow_dangerous,
    )


def _print_outcome(outcome: TaskOutcome) -> None:
    console.print_json(outcome.model_dump_json(indent=2))


@app.command()
def doctor(
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)
    ] = DEFAULT_WORKSPACE,
) -> None:
    """Inspect runtime prerequisites without changing the repository."""

    settings = _settings(workspace)
    table = Table(title="RepoPilot environment")
    table.add_column("Capability")
    table.add_column("Status")
    table.add_row("Workspace", str(settings.workspace))
    table.add_row("Python", sys.executable)
    table.add_row("ripgrep", shutil.which("rg") or "missing")
    table.add_row("Git", shutil.which("git") or "missing")
    table.add_row("OPENAI_API_KEY", "set" if settings.openai_api_key else "missing")
    table.add_row(
        "LLM provider",
        settings.llm_base_url or "OpenAI Responses API",
    )
    table.add_row("LLM model", settings.model)
    table.add_row(
        "LLM credential",
        "set" if settings.agent_api_key else "missing",
    )
    table.add_row("Embedding", f"{settings.embedding_provider}: {settings.embedding_model}")
    table.add_row("Storage", "PostgreSQL + pgvector")
    table.add_row(
        "PostgreSQL URL",
        "set" if settings.postgres_url else "missing",
    )
    table.add_row("GitHub MCP", settings.github_mcp_url or "disabled")
    console.print(table)


@app.command("index")
def index_repository(
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)
    ] = DEFAULT_WORKSPACE,
) -> None:
    """Incrementally synchronize the repository FTS and vector index."""

    settings = _settings(workspace)
    if settings.embedding_provider == "openai" and not settings.openai_api_key:
        raise typer.BadParameter("OPENAI_API_KEY is required to create embeddings")
    stats = asyncio.run(RepoPilot(settings).index())
    console.print_json(json.dumps(stats))


@app.command("run")
def run_task(
    goal: Annotated[str, typer.Argument(help="Coding task to complete")],
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)
    ] = DEFAULT_WORKSPACE,
    task_id: Annotated[str | None, typer.Option("--task-id")] = None,
    skip_index: Annotated[bool, typer.Option("--skip-index")] = False,
    allow_dangerous: Annotated[
        bool,
        typer.Option(
            "--allow-dangerous",
            help="Approve policy-blocked shell syntax for this run; workspace limits remain.",
        ),
    ] = False,
) -> None:
    """Run an autonomous coding task and persist its checkpoint and trace."""

    settings = _settings(workspace, allow_dangerous)
    try:
        outcome = asyncio.run(
            RepoPilot(settings).run(goal, task_id=task_id, ensure_index=not skip_index)
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_outcome(outcome)


@app.command("resume")
def resume_task(
    task_id: Annotated[str, typer.Argument(help="Checkpoint task id")],
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)
    ] = DEFAULT_WORKSPACE,
) -> None:
    """Resume an interrupted or failed task from its persisted task/session state."""

    settings = _settings(workspace)
    try:
        outcome = asyncio.run(RepoPilot(settings).resume(task_id))
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_outcome(outcome)


@app.command("trace")
def show_trace(
    task_id: Annotated[str, typer.Argument(help="Task id")],
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)
    ] = DEFAULT_WORKSPACE,
) -> None:
    """Show the persisted local trace for one task."""

    pilot = RepoPilot(_settings(workspace))
    events = pilot.trace_store.list_for_task(task_id)
    table = Table(title=f"Trace {task_id}")
    for column in ("Category", "Name", "OK", "Duration ms", "Details"):
        table.add_column(column)
    for event in events:
        table.add_row(
            event.category,
            event.name,
            "yes" if event.ok else "no",
            f"{event.duration_ms:.1f}",
            json.dumps(event.details, ensure_ascii=False),
        )
    console.print(table)


@app.command("eval-report")
def eval_report(
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)
    ] = DEFAULT_WORKSPACE,
) -> None:
    """Compute task/test/step/tool/token/latency metrics from persisted runs."""

    pilot = RepoPilot(_settings(workspace))
    summary = Evaluator(pilot.checkpoints, pilot.trace_store).summarize()
    console.print_json(summary.model_dump_json(indent=2))


@app.command("eval")
def run_evaluation(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", exists=True, dir_okay=False, readable=True),
    ] = Path("evals/tasks.json"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory that receives isolated runs and reports."),
    ] = Path("artifacts/evals"),
    allow_dangerous: Annotated[
        bool,
        typer.Option(
            "--allow-dangerous",
            help="Apply the normal opt-in shell policy to the agent inside each eval case.",
        ),
    ] = False,
) -> None:
    """Run the 3-5 case coding benchmark in fresh repository copies."""

    shared_embeddings: EmbeddingProvider | None = None
    shared_reranker: Reranker | None = None

    def build_eval_pilot(workspace: Path) -> RepoPilot:
        nonlocal shared_embeddings, shared_reranker
        pilot = RepoPilot(
            _settings(workspace, allow_dangerous),
            embeddings=shared_embeddings,
            reranker=shared_reranker,
        )
        shared_embeddings = pilot.repository_index.embeddings
        shared_reranker = pilot.repository_index.reranker
        return pilot

    runner = EvalRunner(
        dataset,
        output_dir,
        build_eval_pilot,
    )
    try:
        summary = asyncio.run(runner.run())
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Eval error:[/red] {exc}")
        raise typer.Exit(1) from exc
    table = Table(title=f"RepoPilot eval {summary.run_id}")
    for column in ("Case", "Success", "Tests", "Turns", "Tools", "Tokens", "Seconds"):
        table.add_column(column)
    for result in summary.results:
        table.add_row(
            result.id,
            "yes" if result.success else "no",
            "pass" if result.success_command_passed else "fail",
            str(result.turns),
            str(result.tool_calls),
            str(result.total_tokens),
            f"{result.latency_seconds:.2f}",
        )
    console.print(table)
    console.print_json(summary.model_dump_json(indent=2))
    if summary.task_success_rate < 1:
        raise typer.Exit(1)


@app.command("mcp-demo")
def run_mcp_demo(
    baseline: Annotated[
        Path,
        typer.Option("--baseline", exists=True, file_okay=False, readable=True),
    ] = Path("demo/bug_repo"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory that receives the complete demo evidence."),
    ] = Path("artifacts/mcp-demo"),
) -> None:
    """Read Issue 101 through a real stdio MCP call, then fix and verify it."""

    runner = DemoRunner(
        baseline,
        output_dir,
        lambda workspace: Settings(workspace=workspace, issue_mcp_enabled=True),
    )
    goal = (
        "First call the get_issue MCP tool with issue_number 101 and use that issue as the "
        "source of truth. Then explore the repository, implement the focused fix and regression "
        "tests, run tests, and obtain independent Reviewer approval."
    )
    try:
        report = asyncio.run(runner.run(goal, name="mcp-demo", required_mcp_tool="get_issue"))
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]MCP demo error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print_json(report.model_dump_json(indent=2))
    if not report.success:
        raise typer.Exit(1)


@app.command("demo")
def run_demo(
    baseline: Annotated[
        Path,
        typer.Option("--baseline", exists=True, file_okay=False, readable=True),
    ] = Path("demo/bug_repo"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory that receives the complete demo evidence."),
    ] = Path("artifacts/demo"),
) -> None:
    """Run the real-model login bug E2E and retain trace, diff, and tests."""

    runner = DemoRunner(
        baseline,
        output_dir,
        lambda workspace: Settings(workspace=workspace),
    )
    goal = (
        "Fix the login endpoint returning HTTP 500 for an unknown user. Preserve the existing-user "
        "HTTP 200 contract, add focused regression tests, run tests, and obtain independent "
        "Reviewer approval."
    )
    try:
        report = asyncio.run(runner.run(goal, name="demo"))
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Demo error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print_json(report.model_dump_json(indent=2))
    if not report.success:
        raise typer.Exit(1)


def main() -> None:
    # Support both `python -m repopilot.interfaces.cli` and the console script.
    app()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
