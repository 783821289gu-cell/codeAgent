from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_bge_cache_dir() -> Path:
    configured = os.getenv("REVIEW_AGENT_BGE_CACHE_DIR", "").strip()
    if configured:
        return Path(configured)
    runtime_root = os.getenv("REVIEW_AGENT_RUNTIME_ROOT", "").strip()
    if runtime_root:
        return Path(runtime_root) / "models" / "huggingface"
    return Path.home() / ".cache" / "huggingface" / "hub"


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables and optional `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="REPOPILOT_",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    workspace: Path = Field(default_factory=Path.cwd)
    model: str = Field(
        default="gpt-5.6-terra",
        validation_alias=AliasChoices("REPOPILOT_MODEL", "REVIEW_AGENT_LLM_MODEL"),
    )
    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("REPOPILOT_LLM_BASE_URL", "REVIEW_AGENT_LLM_BASE_URL"),
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("REPOPILOT_LLM_API_KEY", "REVIEW_AGENT_LLM_API_KEY"),
    )
    llm_timeout_seconds: int = Field(
        default=120,
        ge=1,
        le=900,
        validation_alias=AliasChoices(
            "REPOPILOT_LLM_TIMEOUT_SECONDS", "REVIEW_AGENT_LLM_TIMEOUT_SECONDS"
        ),
    )
    embedding_provider: str = "bge"
    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    embedding_dimensions: int = Field(default=1024, ge=1)
    embedding_cache_dir: Path = Field(default_factory=_default_bge_cache_dir)
    embedding_device: str = "cpu"
    embedding_max_length: int = Field(default=1024, ge=32, le=8192)
    embedding_batch_size: int = Field(default=8, ge=1, le=256)
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_revision: str = "2cfc18c9415c912f9d8155881c133215df768a70"
    reranker_cache_dir: Path = Field(default_factory=_default_bge_cache_dir)
    reranker_device: str = "cpu"
    reranker_max_length: int = Field(default=512, ge=32, le=8192)
    reranker_batch_size: int = Field(default=8, ge=1, le=256)
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    max_turns: int = Field(default=30, ge=1, le=100)
    shell_timeout_seconds: int = Field(default=120, ge=1, le=900)
    max_tool_output_tokens: int = Field(default=3_000, ge=128)
    context_budget_tokens: int = Field(default=12_000, ge=1_000)
    compaction_threshold_tokens: int = Field(default=9_000, ge=512)
    sdk_compaction_threshold_tokens: int = Field(default=100_000, ge=4_000)
    rag_top_k: int = Field(default=8, ge=1, le=30)
    chunk_lines: int = Field(default=80, ge=20, le=300)
    chunk_overlap_lines: int = Field(default=15, ge=0, le=100)
    allow_dangerous_commands: bool = False
    github_mcp_url: str | None = None
    github_token: SecretStr | None = None
    issue_mcp_enabled: bool = False
    postgres_url: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "DATABASE_URL", "REPOPILOT_DATABASE_URL", "REVIEW_AGENT_DATABASE_URL"
        ),
    )
    pgvector_schema: str = "repopilot"
    memory_duplicate_threshold: float = Field(default=0.94, ge=0, le=1)
    memory_topic_threshold: float = Field(default=0.80, ge=0, le=1)
    tracing_enabled: bool = True

    @field_validator("workspace", mode="after")
    @classmethod
    def resolve_workspace(cls, value: Path) -> Path:
        path = value.expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"workspace does not exist or is not a directory: {path}")
        return path

    @field_validator("chunk_overlap_lines")
    @classmethod
    def overlap_smaller_than_chunk(cls, value: int, info: object) -> int:
        data = getattr(info, "data", {})
        chunk_lines = data.get("chunk_lines", 80)
        if value >= chunk_lines:
            raise ValueError("chunk_overlap_lines must be smaller than chunk_lines")
        return value

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, value: str) -> str:
        normalized = value.casefold().strip()
        if normalized not in {"bge", "openai"}:
            raise ValueError("embedding_provider must be 'bge' or 'openai'")
        return normalized

    @model_validator(mode="after")
    def validate_context_budget(self) -> Settings:
        if self.compaction_threshold_tokens > self.context_budget_tokens:
            raise ValueError("compaction threshold cannot exceed total context budget")
        return self

    @property
    def agent_api_key(self) -> SecretStr | None:
        return self.llm_api_key or self.openai_api_key
