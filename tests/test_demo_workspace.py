import subprocess
from pathlib import Path

from repopilot.interfaces.workspace import prepare_workspace


def git(workspace: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(workspace), *args], text=True).strip()


def test_fresh_clone_fixture_gets_independent_diff(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    baseline = tmp_path / "demo" / "fixture"
    baseline.mkdir(parents=True)
    (baseline / "app.py").write_text("value = 1\n", encoding="utf-8")
    workspace = tmp_path / "artifacts" / "workspace"

    prepare_workspace(baseline, workspace)

    assert Path(git(workspace, "rev-parse", "--show-toplevel")) == workspace
    assert git(workspace, "status", "--porcelain") == ""
    (workspace / "app.py").write_text("value = 2\n", encoding="utf-8")
    diff = git(workspace, "diff")
    assert "-value = 1" in diff and "+value = 2" in diff
    assert (baseline / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def test_fixture_git_metadata_and_secrets_are_not_copied(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / ".git").write_text("gitdir: /unavailable/worktree\n", encoding="utf-8")
    (baseline / ".env").write_text("PRIVATE_SETTING=test-only\n", encoding="utf-8")
    (baseline / ".env.local").write_text("PRIVATE_SETTING=test-only\n", encoding="utf-8")
    (baseline / ".repopilot").mkdir()
    (baseline / ".repopilot" / "state.db").write_bytes(b"private runtime")
    (baseline / "app.py").write_text("value = 1\n", encoding="utf-8")
    workspace = tmp_path / "workspace"

    prepare_workspace(baseline, workspace)

    assert (workspace / ".git").is_dir()
    assert not (workspace / ".env").exists()
    assert not (workspace / ".env.local").exists()
    assert not (workspace / ".repopilot").exists()
    assert git(workspace, "ls-files") == "app.py"
