from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class ToolResult[T](BaseModel):
    ok: bool
    data: T | None = None
    error: str | None = None
    truncated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    NEEDS_REPAIR = "needs_repair"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskState(BaseModel):
    task_id: str
    goal: str
    plan: list[str] = Field(default_factory=list)
    current_step: str = "initialize"
    explored_files: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    test_status: str = "not_run"
    retry_count: int = 0
    review_status: str = "not_run"
    status: TaskStatus = TaskStatus.PENDING
    compact_summary: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ExplorationReport(BaseModel):
    relevant_files: list[str] = Field(default_factory=list)
    call_chain: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    potential_root_causes: list[str] = Field(default_factory=list)
    related_tests: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)


class ReviewSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReviewFinding(BaseModel):
    severity: ReviewSeverity
    file: str | None = None
    line: int | None = None
    problem: str
    recommendation: str


class ReviewReport(BaseModel):
    summary: str
    approved: bool
    findings: list[ReviewFinding] = Field(default_factory=list)
    missing_tests: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    path: str
    start_line: int
    end_line: int
    content: str
    score: float
    rrf_score: float | None = None
    rerank_score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None


class MemoryKind(StrEnum):
    PROJECT_CONSTRAINT = "project_constraint"
    USER_PREFERENCE = "user_preference"
    ARCHITECTURE_DECISION = "architecture_decision"
    CODING_CONVENTION = "coding_convention"
    REUSABLE_EXPERIENCE = "reusable_experience"

    CONSTRAINT = "project_constraint"
    PREFERENCE = "user_preference"
    DECISION = "architecture_decision"
    EXPERIENCE = "reusable_experience"


class MemoryPolarity(StrEnum):
    REQUIRE = "require"
    FORBID = "forbid"
    PREFER = "prefer"
    AVOID = "avoid"
    FACT = "fact"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class MemoryItem(BaseModel):
    id: int | None = None
    scope: str
    kind: MemoryKind
    topic: str = ""
    content: str
    polarity: MemoryPolarity = MemoryPolarity.FACT
    status: MemoryStatus = MemoryStatus.ACTIVE
    importance: float = Field(default=0.5, ge=0, le=1)
    supersedes_id: int | None = None
    conflict_with_id: int | None = None
    embedding: list[float] = Field(default_factory=list, exclude=True, repr=False)
    topic_embedding: list[float] = Field(default_factory=list, exclude=True, repr=False)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ExtractedMemory(BaseModel):
    kind: MemoryKind
    topic: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=2_000)
    polarity: MemoryPolarity = MemoryPolarity.FACT
    importance: float = Field(default=0.7, ge=0, le=1)
    source: Literal["user_task", "project_evidence", "validated_result"] = "validated_result"
    evidence: str = Field(default="", max_length=500)


class MemoryExtractionReport(BaseModel):
    items: list[ExtractedMemory] = Field(default_factory=list, max_length=12)


class TestResult(BaseModel):
    command: str
    passed: bool
    exit_code: int
    output: str
    duration_seconds: float


class ContextSource(StrEnum):
    TASK = "task"
    STATE = "state"
    MEMORY = "memory"
    RAG = "rag"
    HISTORY = "history"
    TOOL = "tool"
    TEST = "test"
    EXPLORER = "explorer"
    REVIEWER = "reviewer"


class ContextItem(BaseModel):
    source: ContextSource
    content: str
    priority: int = Field(default=50, ge=0, le=100)
    critical: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class ContextBudgetReport(BaseModel):
    initial_tokens: int
    final_tokens: int
    budget_tokens: int
    category_tokens: dict[ContextSource, int] = Field(default_factory=dict)
    reduction_trace: list[str] = Field(default_factory=list)
    pruned_items: int = 0


class TaskOutcome(BaseModel):
    task_id: str
    status: TaskStatus
    summary: str
    changed_files: list[str] = Field(default_factory=list)
    tests: list[TestResult] = Field(default_factory=list)
    review: ReviewReport | None = None
    trace_id: str | None = None


class TraceEvent(BaseModel):
    trace_id: str
    task_id: str
    category: str
    name: str
    started_at: datetime
    duration_ms: float
    ok: bool
    details: dict[str, Any] = Field(default_factory=dict)
