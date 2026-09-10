from pathlib import Path

import pytest

from repopilot.repository.tools import (
    CommandPolicy,
    CommandRejected,
    WorkspaceBoundary,
    WorkspaceTools,
)


def test_workspace_boundary_blocks_escape(tmp_path: Path) -> None:
    boundary = WorkspaceBoundary(tmp_path)
    with pytest.raises(ValueError, match="outside workspace"):
        boundary.resolve("../secret.txt")


def test_read_replace_search_and_list(tmp_path: Path) -> None:
    source = tmp_path / "app.py"
    source.write_text("def login():\n    return 'broken'\n", encoding="utf-8")
    tools = WorkspaceTools(tmp_path)

    assert tools.read_file("app.py").ok
    search = tools.search_code("login")
    assert search.ok and "app.py:1:def login" in (search.data or "")
    assert tools.replace_text("app.py", "'broken'", "'fixed'").ok
    assert source.read_text(encoding="utf-8").endswith("'fixed'\n")
    assert tools.list_files().data == ["app.py"]


def test_replace_is_stale_safe(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("same same", encoding="utf-8")
    result = WorkspaceTools(tmp_path).replace_text("a.txt", "same", "new")
    assert not result.ok
    assert "expected 1 occurrence" in (result.error or "")


@pytest.mark.parametrize(
    "command",
    [
        "git reset --hard",
        "rm -rf .",
        "pytest && shutdown",
        "cd ..",
        "type C:\\Windows\\win.ini",
        "type $env:SystemRoot\\win.ini",
        "python script.py > output.txt",
    ],
)
def test_command_policy_rejects_dangerous_commands(command: str) -> None:
    with pytest.raises(CommandRejected):
        CommandPolicy().validate(command)


def test_command_output_is_bounded(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path, max_output_tokens=100)
    result = tools.run_command("python -c \"print('x'*2000)\"")
    assert result.ok and result.truncated
    assert tools.token_counter.count(result.data or "") <= 100
