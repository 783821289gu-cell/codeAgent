"""Create self-contained Git workspaces for reproducible demos and evaluations."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def prepare_workspace(baseline: Path, workspace: Path) -> None:
    """Copy a trusted fixture and commit its starting state, even after a fresh clone."""
    shutil.copytree(
        baseline,
        workspace,
        ignore=shutil.ignore_patterns(
            ".git", ".repopilot", ".pytest_cache", ".ruff_cache", "__pycache__",
            ".venv", "node_modules", ".env", ".env.*",
        ),
    )
    commands = (
        ["git", "init", "-b", "main"],
        ["git", "add", "--all"],
        [
            "git", "-c", "user.name=RepoPilot Demo",
            "-c", "user.email=repopilot@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "--allow-empty",
            "-m", "Baseline snapshot for isolated run",
        ],
    )
    for command in commands:
        subprocess.run(
            command, cwd=workspace, capture_output=True, text=True,
            check=True, timeout=30,
        )
