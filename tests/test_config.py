from pathlib import Path

import pytest
from pydantic import ValidationError

from repopilot.core.config import Settings


def test_settings_resolve_workspace(tmp_path: Path) -> None:
    settings = Settings(
        workspace=tmp_path,
        model="direct-model",
        _env_file=None,
    )
    assert settings.workspace == tmp_path.resolve()
    assert settings.model == "direct-model"
    assert settings.pgvector_schema == "repopilot"


def test_settings_reject_bad_context_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(
            workspace=tmp_path,
            context_budget_tokens=1_000,
            compaction_threshold_tokens=1_200,
            _env_file=None,
        )


def test_database_url_environment_configures_postgres(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:password@db/repopilot")

    settings = Settings(workspace=tmp_path, _env_file=None)

    assert settings.postgres_url is not None
    assert settings.postgres_url.get_secret_value().startswith("postgresql+psycopg://")


def test_review_agent_openai_compatible_environment_is_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REVIEW_AGENT_LLM_BASE_URL", "https://llm.example.test")
    monkeypatch.setenv("REVIEW_AGENT_LLM_API_KEY", "secret")
    monkeypatch.setenv("REVIEW_AGENT_LLM_MODEL", "compatible-model")
    monkeypatch.setenv("REVIEW_AGENT_LLM_TIMEOUT_SECONDS", "45")

    settings = Settings(workspace=tmp_path, _env_file=None)

    assert settings.llm_base_url == "https://llm.example.test"
    assert settings.agent_api_key is not None
    assert settings.agent_api_key.get_secret_value() == "secret"
    assert settings.model == "compatible-model"
    assert settings.llm_timeout_seconds == 45
