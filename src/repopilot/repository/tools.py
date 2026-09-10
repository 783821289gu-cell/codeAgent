from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from repopilot.core.models import TestResult, ToolResult
from repopilot.knowledge.context import prune_test_output
from repopilot.knowledge.tokens import TokenCounter


class WorkspaceViolation(ValueError):
    pass


class CommandRejected(ValueError):
    pass


class WorkspaceBoundary:
    """Central path policy used by every filesystem operation."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def resolve(self, path: str | Path, *, must_exist: bool = False) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve(strict=False)
        try:
            common = Path(os.path.commonpath([self.workspace, resolved]))
        except ValueError as exc:
            raise WorkspaceViolation(f"path is outside workspace: {path}") from exc
        if common != self.workspace:
            raise WorkspaceViolation(f"path is outside workspace: {path}")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        return resolved

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.workspace).as_posix()


_DANGEROUS_PATTERNS = (
    r"\brm\s+-[^\r\n]*r[^\r\n]*f\b",
    r"\b(remove-item|del|erase|rmdir|rd)\b[^\r\n]*(/s|/q|-recurse|-force)",
    r"\bgit\s+(reset\s+--hard|clean\s+-[^\r\n]*f|push\s+--force)\b",
    r"\b(format|diskpart|shutdown|reboot|restart-computer|stop-computer)\b",
    r"\b(drop\s+(database|table)|truncate\s+table)\b",
    r"(?:^|\s)(?:[A-Za-z]:\\|/)(?:\s|$)",
)
_SHELL_META = re.compile(r"(?:\r|\n|&&|\|\||[;<>`]|\$\()")
_ABSOLUTE_WINDOWS_PATH = re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/]|\\\\)")
_ENVIRONMENT_REFERENCE = re.compile(r"(?:\$env:|%[A-Za-z_][A-Za-z0-9_]*%)", re.IGNORECASE)
_RG_EXCLUDED_DIRS = (
    ".git",
    ".venv",
    ".repopilot",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
)


def _rg_exclude_args() -> list[str]:
    return [argument for name in _RG_EXCLUDED_DIRS for argument in ("-g", f"!{name}")]


class CommandPolicy:
    def __init__(self, allow_dangerous: bool = False) -> None:
        self.allow_dangerous = allow_dangerous

    def validate(self, command: str, *, allowed_absolute_paths: tuple[str, ...] = ()) -> None:
        normalized = command.strip()
        if not normalized:
            raise CommandRejected("command cannot be empty")
        if len(normalized) > 2_000:
            raise CommandRejected("command is too long")
        if ".." in normalized or "~" in normalized:
            raise CommandRejected("parent and home-directory references are not allowed")
        path_checked = normalized
        for allowed in allowed_absolute_paths:
            path_checked = path_checked.replace(allowed, "")
        if _ABSOLUTE_WINDOWS_PATH.search(path_checked) or re.search(
            r"(?:^|\s)\\(?!\\)", path_checked
        ):
            raise CommandRejected("absolute paths are not allowed in shell commands")
        if _ENVIRONMENT_REFERENCE.search(normalized):
            raise CommandRejected("environment-variable path references are not allowed")
        if not self.allow_dangerous and _SHELL_META.search(normalized):
            raise CommandRejected("shell chaining, redirection, and substitution require approval")
        if not self.allow_dangerous:
            for pattern in _DANGEROUS_PATTERNS:
                if re.search(pattern, normalized, flags=re.IGNORECASE):
                    raise CommandRejected("dangerous command requires explicit CLI approval")


def _trim(value: str, maximum: int, counter: TokenCounter) -> tuple[str, bool]:
    if counter.count(value) <= maximum:
        return value, False
    return counter.truncate_middle(value, maximum), True


class WorkspaceTools:
    """Deterministic local repository operations; no planning or LLM calls live here."""

    def __init__(
        self,
        workspace: Path,
        *,
        timeout_seconds: int = 120,
        max_output_tokens: int = 3_000,
        token_counter: TokenCounter | None = None,
        allow_dangerous: bool = False,
    ) -> None:
        self.boundary = WorkspaceBoundary(workspace)
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.token_counter = token_counter or TokenCounter("gpt-5.6-terra")
        self.command_policy = CommandPolicy(allow_dangerous)

    @property
    def workspace(self) -> Path:
        return self.boundary.workspace

    def read_file(
        self, path: str, start_line: int = 1, end_line: int | None = None
    ) -> ToolResult[str]:
        try:
            target = self.boundary.resolve(path, must_exist=True)
            if not target.is_file():
                raise ValueError(f"not a file: {path}")
            if start_line < 1 or (end_line is not None and end_line < start_line):
                raise ValueError("invalid line range")
            lines = target.read_text(encoding="utf-8").splitlines()
            chosen = lines[start_line - 1 : end_line]
            numbered = "\n".join(
                f"{number}: {line}" for number, line in enumerate(chosen, start=start_line)
            )
            output, truncated = _trim(numbered, self.max_output_tokens, self.token_counter)
            return ToolResult(ok=True, data=output, truncated=truncated)
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolResult(ok=False, error=str(exc))

    def list_files(self, path: str = ".", limit: int = 500) -> ToolResult[list[str]]:
        try:
            target = self.boundary.resolve(path, must_exist=True)
            if not target.is_dir():
                raise ValueError(f"not a directory: {path}")
            completed = subprocess.run(
                ["rg", "--files", "--hidden", *_rg_exclude_args()],
                cwd=target,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
            files = [
                self.boundary.relative(target / value)
                for value in completed.stdout.splitlines()
                if value.strip()
            ]
            return ToolResult(ok=True, data=files[:limit], truncated=len(files) > limit)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return ToolResult(ok=False, error=str(exc))

    def search_code(
        self, query: str, path: str = ".", *, regex: bool = False, limit: int = 100
    ) -> ToolResult[str]:
        try:
            if not query:
                raise ValueError("query cannot be empty")
            target = self.boundary.resolve(path, must_exist=True)
            command = [
                "rg",
                "-n",
                "--no-heading",
                "--color",
                "never",
                "--hidden",
                *_rg_exclude_args(),
            ]
            if not regex:
                command.append("--fixed-strings")
            command.extend([query, str(target)])
            completed = subprocess.run(
                command,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode not in {0, 1}:
                raise OSError(completed.stderr.strip() or f"ripgrep exited {completed.returncode}")
            matches = completed.stdout.splitlines()
            normalized: list[str] = []
            for match in matches[:limit]:
                prefix, separator, rest = match.partition(":")
                with suppress(ValueError, WorkspaceViolation):
                    prefix = self.boundary.relative(Path(prefix))
                normalized.append(prefix + separator + rest)
            output, truncated_chars = _trim(
                "\n".join(normalized), self.max_output_tokens, self.token_counter
            )
            return ToolResult(
                ok=True,
                data=output,
                truncated=truncated_chars or len(matches) > limit,
                metadata={"match_count": len(matches)},
            )
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return ToolResult(ok=False, error=str(exc))

    def write_file(self, path: str, content: str, *, overwrite: bool = False) -> ToolResult[str]:
        try:
            target = self.boundary.resolve(path)
            if target.exists() and not overwrite:
                raise FileExistsError(f"file already exists: {path}; use replace_text")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
            return ToolResult(ok=True, data=self.boundary.relative(target))
        except (OSError, ValueError) as exc:
            return ToolResult(ok=False, error=str(exc))

    def replace_text(
        self, path: str, old: str, new: str, *, expected_occurrences: int = 1
    ) -> ToolResult[str]:
        try:
            if not old:
                raise ValueError("old text cannot be empty")
            if expected_occurrences < 1:
                raise ValueError("expected_occurrences must be positive")
            target = self.boundary.resolve(path, must_exist=True)
            content = target.read_text(encoding="utf-8")
            actual = content.count(old)
            if actual != expected_occurrences:
                raise ValueError(
                    f"stale edit: expected {expected_occurrences} occurrence(s), found {actual}"
                )
            target.write_text(content.replace(old, new), encoding="utf-8", newline="\n")
            return ToolResult(ok=True, data=self.boundary.relative(target))
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolResult(ok=False, error=str(exc))

    def run_command(
        self,
        command: str,
        timeout_seconds: int | None = None,
        *,
        _allowed_absolute_paths: tuple[str, ...] = (),
        _test_output: bool = False,
    ) -> ToolResult[str]:
        try:
            self.command_policy.validate(command, allowed_absolute_paths=_allowed_absolute_paths)
            timeout = min(timeout_seconds or self.timeout_seconds, self.timeout_seconds)
            completed = subprocess.run(
                command,
                cwd=self.workspace,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            combined = completed.stdout
            if completed.stderr:
                combined += ("\n" if combined else "") + "STDERR:\n" + completed.stderr
            raw_output = combined.strip()
            if _test_output:
                output = prune_test_output(
                    raw_output,
                    passed=completed.returncode == 0,
                    max_tokens=self.max_output_tokens,
                    counter=self.token_counter,
                )
                truncated = self.token_counter.count(output) < self.token_counter.count(raw_output)
            else:
                output, truncated = _trim(raw_output, self.max_output_tokens, self.token_counter)
            return ToolResult(
                ok=completed.returncode == 0,
                data=output,
                error=None if completed.returncode == 0 else f"exit code {completed.returncode}",
                truncated=truncated,
                metadata={"exit_code": completed.returncode},
            )
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return ToolResult(ok=False, error=str(exc))

    def run_tests(self, command: str | None = None) -> TestResult:
        command = command or subprocess.list2cmdline([sys.executable, "-m", "pytest", "-q"])
        started = time.perf_counter()
        allowed = (sys.executable,) if sys.executable in command else ()
        result = self.run_command(command, _allowed_absolute_paths=allowed, _test_output=True)
        return TestResult(
            command=command,
            passed=result.ok,
            exit_code=int(result.metadata.get("exit_code", -1)),
            output=result.data or result.error or "",
            duration_seconds=time.perf_counter() - started,
        )

    def git_status(self) -> ToolResult[str]:
        return self.run_command("git status --short")

    def git_diff(self) -> ToolResult[str]:
        return self.run_command("git diff --no-ext-diff")

    def as_dict(self, result: ToolResult[Any] | TestResult) -> dict[str, Any]:
        return result.model_dump(mode="json")
