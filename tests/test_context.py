import pytest
from agents.run_config import ModelInputData

from repopilot.core.models import ContextItem, ContextSource, TaskState
from repopilot.knowledge.context import ContextManager, diagnostic_excerpt, prune_test_output
from repopilot.knowledge.tokens import TokenCounter


@pytest.fixture
def counter() -> TokenCounter:
    return TokenCounter("gpt-5.6-terra")


def test_diagnostic_excerpt_keeps_failure_evidence(counter: TokenCounter) -> None:
    text = "\n".join(["noise"] * 100 + ["AssertionError: expected 404"] + ["tail"] * 100)
    result = diagnostic_excerpt(text, 50, counter)
    assert "AssertionError" in result
    assert counter.count(result) <= 50


def test_success_and_failure_test_outputs_are_semantically_pruned(
    counter: TokenCounter,
) -> None:
    success = "\n".join(["collecting noise"] * 500 + ["42 passed in 1.25s"])
    failure = "\n".join(
        ["noise"] * 500
        + ["Traceback (most recent call last):", "AssertionError: expected 404"]
        + ["tail"] * 500
        + ["1 failed, 41 passed in 3.2s"]
    )

    success_result = prune_test_output(success, passed=True, max_tokens=40, counter=counter)
    failure_result = prune_test_output(failure, passed=False, max_tokens=80, counter=counter)

    assert success_result == "42 passed in 1.25s"
    assert "AssertionError" in failure_result
    assert "1 failed" in failure_result
    assert counter.count(failure_result) <= 80


@pytest.mark.asyncio
async def test_context_compaction_preserves_state_and_uses_summarizer(
    counter: TokenCounter,
) -> None:
    manager = ContextManager(1_000, 500, counter)
    state = TaskState(task_id="1", goal="fix login error")
    state.changed_files = ["auth.py"]
    calls: list[str] = []

    async def summarize(payload: str) -> str:
        calls.append(payload)
        return "Goal preserved: fix login error; changed auth.py; unresolved AssertionError"

    items = [
        ContextItem(source=ContextSource.TASK, content="fix login error", critical=True),
        ContextItem(
            source=ContextSource.TEST,
            content="noise\n" * 2_000 + "AssertionError: 500 != 404",
            priority=80,
        ),
    ]
    result = await manager.build(items, state, summarizer=summarize)

    assert len(calls) == 1
    assert "Goal preserved" in result
    assert state.compact_summary.startswith("Goal preserved")
    assert manager.last_report.final_tokens <= 1_000


@pytest.mark.asyncio
async def test_context_render_never_exceeds_token_budget(counter: TokenCounter) -> None:
    manager = ContextManager(300, 300, counter)
    items = [
        ContextItem(source=ContextSource.TASK, content="a " * 450, critical=True),
        ContextItem(source=ContextSource.RAG, content="b " * 450, priority=80),
    ]

    result = await manager.build(items, TaskState(task_id="budget", goal="bounded"))

    assert counter.count(result) <= 300


def test_model_input_filter_prunes_old_tool_history_and_tracks_live_sources(
    counter: TokenCounter,
) -> None:
    manager = ContextManager(450, 400, counter)
    model_items: list[dict[str, str]] = [
        {"type": "message", "role": "user", "content": "original task"}
    ]
    for index in range(12):
        model_items.extend(
            [
                {
                    "type": "function_call",
                    "name": "run_tests" if index % 3 == 0 else "search_code",
                    "call_id": f"call-{index}",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": f"call-{index}",
                    "output": "large output " * 80,
                },
            ]
        )
    model_items.append({"type": "message", "role": "user", "content": "continue"})
    live_items = [
        ContextItem(source=ContextSource.TOOL, content="search output", priority=30),
        ContextItem(source=ContextSource.TEST, content="1 failed", priority=90),
        ContextItem(source=ContextSource.EXPLORER, content="root cause", priority=85),
        ContextItem(source=ContextSource.REVIEWER, content="missing case", priority=65),
    ]

    filtered, report = manager.filter_model_input(
        ModelInputData(input=model_items, instructions="system instructions"), live_items
    )

    assert report.pruned_items > 0
    assert report.final_tokens <= report.budget_tokens
    assert len(filtered.input) < len(model_items)
    assert {
        ContextSource.TOOL,
        ContextSource.TEST,
        ContextSource.EXPLORER,
        ContextSource.REVIEWER,
    } <= set(report.category_tokens)
