from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence

from agents.run_config import ModelInputData

from repopilot.core.models import (
    ContextBudgetReport,
    ContextItem,
    ContextSource,
    TaskState,
)
from repopilot.knowledge.tokens import TokenCounter

Summarizer = Callable[[str], Awaitable[str]]


SOURCE_WEIGHTS: dict[ContextSource, float] = {
    ContextSource.TASK: 1.0,
    ContextSource.STATE: 1.0,
    ContextSource.EXPLORER: 0.8,
    ContextSource.RAG: 0.7,
    ContextSource.MEMORY: 0.55,
    ContextSource.REVIEWER: 0.6,
    ContextSource.TEST: 0.8,
    ContextSource.TOOL: 0.35,
    ContextSource.HISTORY: 0.3,
}


def diagnostic_excerpt(text: str, max_tokens: int, counter: TokenCounter) -> str:
    """Retain failure evidence and boundaries under a token budget."""

    if counter.count(text) <= max_tokens:
        return text
    lines = text.splitlines()
    diagnostic_indexes = {
        index
        for index, line in enumerate(lines)
        if re.search(
            r"error|fail|exception|traceback|assert|warning|short test summary",
            line,
            re.IGNORECASE,
        )
    }
    selected_indexes = set(range(min(8, len(lines))))
    selected_indexes.update(range(max(0, len(lines) - 10), len(lines)))
    for index in diagnostic_indexes:
        selected_indexes.update(range(max(0, index - 2), min(len(lines), index + 3)))
    selected = [lines[index] for index in sorted(selected_indexes)]
    return counter.truncate_middle("\n".join(selected), max_tokens)


def prune_test_output(text: str, *, passed: bool, max_tokens: int, counter: TokenCounter) -> str:
    if not text.strip():
        return text
    if passed:
        summaries = [
            line
            for line in text.splitlines()
            if re.search(r"\b\d+\s+passed\b|\btests? passed\b|\bok\b", line, re.IGNORECASE)
        ]
        result = summaries[-1] if summaries else text.splitlines()[-1]
        return counter.truncate_prefix(result, max_tokens)
    return diagnostic_excerpt(text, max_tokens, counter)


class ContextManager:
    """Token-budgeted turn context; SDK Compaction separately owns session history."""

    def __init__(
        self,
        budget_tokens: int,
        compaction_threshold_tokens: int,
        counter: TokenCounter,
    ) -> None:
        if compaction_threshold_tokens > budget_tokens:
            raise ValueError("compaction threshold cannot exceed total context budget")
        self.budget_tokens = budget_tokens
        self.compaction_threshold_tokens = compaction_threshold_tokens
        self.counter = counter
        self.last_report = ContextBudgetReport(
            initial_tokens=0,
            final_tokens=0,
            budget_tokens=budget_tokens,
        )

    async def build(
        self,
        items: Sequence[ContextItem],
        state: TaskState,
        *,
        summarizer: Summarizer | None = None,
    ) -> str:
        ordered = self._ordered_unique(items)
        initial_tokens = sum(self.counter.count(item.content) for item in ordered)
        reduction_trace: list[str] = []
        if initial_tokens >= self.compaction_threshold_tokens:
            summary = await self.compact(ordered, state, summarizer=summarizer)
            recent = self._render_budgeted(ordered[:6], max(256, self.budget_tokens // 3))
            rendered = f"# Compact summary\n{summary}\n\n# Recent critical context\n{recent}"
            rendered = self.counter.truncate_prefix(rendered, self.budget_tokens)
            reduction_trace.append("summarized_initial_context")
        else:
            rendered = self._render_budgeted(ordered, self.budget_tokens)
            if self.counter.count(rendered) < initial_tokens:
                reduction_trace.append("applied_source_token_budgets")
        self.last_report = self._report(
            ordered,
            initial_tokens,
            self.counter.count(rendered),
            reduction_trace,
        )
        return rendered

    async def compact(
        self,
        items: Sequence[ContextItem],
        state: TaskState,
        *,
        summarizer: Summarizer | None = None,
    ) -> str:
        grouped: dict[ContextSource, list[str]] = defaultdict(list)
        for item in items:
            per_item_limit = 750 if item.critical else 375
            content = (
                diagnostic_excerpt(item.content, per_item_limit, self.counter)
                if item.source in {ContextSource.TEST, ContextSource.TOOL}
                else self.counter.truncate_prefix(item.content, per_item_limit)
            )
            if content not in grouped[item.source]:
                grouped[item.source].append(content)
        state_block = (
            f"Goal: {state.goal}\nCurrent step: {state.current_step}\n"
            f"Plan: {state.plan}\nExplored: {state.explored_files}\n"
            f"Changed: {state.changed_files}\nTest: {state.test_status}\n"
            f"Review: {state.review_status}\nRetries: {state.retry_count}"
        )
        source_blocks = [
            f"## {source.value}\n" + "\n---\n".join(values) for source, values in grouped.items()
        ]
        payload = state_block + "\n\n" + "\n\n".join(source_blocks)
        result = await summarizer(payload) if summarizer is not None else payload
        state.compact_summary = self.counter.truncate_prefix(
            result, max(256, self.budget_tokens // 2)
        )
        return state.compact_summary

    def filter_model_input(
        self,
        model_data: ModelInputData,
        live_items: Sequence[ContextItem],
    ) -> tuple[ModelInputData, ContextBudgetReport]:
        ordered_live_items = self._ordered_unique(live_items)
        requested_live_budget = max(64, self.budget_tokens // 4)
        live_context = self._render_budgeted(ordered_live_items, requested_live_budget)
        base_instructions = model_data.instructions or ""
        instructions = base_instructions
        if live_context:
            instructions += "\n\n# Live structured task context (token-budgeted)\n" + live_context
        retained = list(model_data.input)
        initial_tokens = self._model_input_tokens(instructions, retained)
        reduction_trace: list[str] = []
        pruned_items = 0
        if initial_tokens > self.budget_tokens:
            for indices, category, _priority in self._removable_groups(retained):
                if self._model_input_tokens(instructions, retained) <= self.budget_tokens:
                    break
                removed = 0
                for index in indices:
                    if 0 <= index < len(retained) and retained[index] is not None:
                        retained[index] = None  # type: ignore[assignment]
                        removed += 1
                if removed:
                    pruned_items += removed
                    reduction_trace.append(f"pruned_old_{category.value}_items")
            retained = [item for item in retained if item is not None]
        if self._model_input_tokens(instructions, retained) > self.budget_tokens:
            base_tokens = self.counter.count(base_instructions)
            reserved_live_tokens = min(128, max(0, self.budget_tokens - base_tokens - 64))
            input_budget = max(32, self.budget_tokens - base_tokens - reserved_live_tokens)
            retained, compacted_items = self._compact_textual_items(retained, input_budget)
            if compacted_items:
                reduction_trace.append("compacted_retained_text_items")
        input_tokens = self._model_input_tokens("", retained)
        marker = "\n\n# Live structured task context (token-budgeted)\n"
        available_live_tokens = max(
            0,
            self.budget_tokens
            - self.counter.count(base_instructions)
            - input_tokens
            - self.counter.count(marker),
        )
        live_context = self._render_budgeted(
            ordered_live_items,
            min(requested_live_budget, available_live_tokens),
        )
        instructions = base_instructions + (marker + live_context if live_context else "")
        if self._model_input_tokens(instructions, retained) > self.budget_tokens:
            excess = self._model_input_tokens(instructions, retained) - self.budget_tokens
            retained, compacted_items = self._compact_textual_items(
                retained, max(16, input_tokens - excess - 8)
            )
            if compacted_items:
                reduction_trace.append("enforced_strict_model_input_budget")
        final_tokens = self._model_input_tokens(instructions, retained)
        report = self._report(
            live_items,
            initial_tokens,
            final_tokens,
            list(dict.fromkeys(reduction_trace)),
            pruned_items=pruned_items,
        )
        self.last_report = report
        return ModelInputData(input=retained, instructions=instructions), report

    def _compact_textual_items(
        self, items: Sequence[object], budget: int
    ) -> tuple[list[object], int]:
        compacted = list(items)
        changed = 0
        while self._model_input_tokens("", compacted) > budget:
            candidates = [
                (index, self.counter.count(text), field, text)
                for index, item in enumerate(compacted)
                for field, text in _text_fields(item)
                if self.counter.count(text) > 24
            ]
            if not candidates:
                break
            index, text_tokens, field, text = max(candidates, key=lambda candidate: candidate[1])
            excess = self._model_input_tokens("", compacted) - budget
            target_tokens = max(16, text_tokens - excess - 8)
            replacement = (
                diagnostic_excerpt(text, target_tokens, self.counter)
                if field == "output"
                else self.counter.truncate_middle(text, target_tokens)
            )
            compacted[index] = _replace_text_field(compacted[index], field, replacement)
            changed += 1
        return compacted, changed

    def _render_budgeted(self, items: Sequence[ContextItem], budget: int) -> str:
        sections: list[str] = []
        used_by_source: dict[ContextSource, int] = defaultdict(int)
        used = 0
        for item in items:
            header = f"## {item.source.value}\n"
            header_tokens = self.counter.count(header)
            remaining = budget - used - header_tokens
            source_cap = max(64, int(budget * SOURCE_WEIGHTS[item.source]))
            source_remaining = source_cap - used_by_source[item.source]
            allowance = min(remaining, source_remaining)
            if allowance <= 0:
                continue
            content = (
                diagnostic_excerpt(item.content, allowance, self.counter)
                if item.source in {ContextSource.TEST, ContextSource.TOOL}
                else self.counter.truncate_prefix(item.content, allowance)
            )
            content_tokens = self.counter.count(content)
            sections.append(header + content)
            used += header_tokens + content_tokens
            used_by_source[item.source] += content_tokens
            if used >= budget:
                break
        return self.counter.truncate_prefix("\n\n".join(sections), budget)

    @staticmethod
    def _ordered_unique(items: Sequence[ContextItem]) -> list[ContextItem]:
        unique: dict[tuple[ContextSource, str], ContextItem] = {}
        for item in items:
            normalized = " ".join(item.content.casefold().split())
            key = (item.source, normalized)
            previous = unique.get(key)
            if previous is None or item.priority > previous.priority:
                unique[key] = item
        return sorted(
            unique.values(),
            key=lambda item: (item.critical, item.priority, item.created_at),
            reverse=True,
        )

    def _model_input_tokens(self, instructions: str, items: Sequence[object]) -> int:
        payload = json.dumps(
            [_item_payload(item) for item in items],
            ensure_ascii=False,
            default=str,
        )
        return self.counter.count(instructions) + self.counter.count(payload)

    def _removable_groups(
        self, items: Sequence[object]
    ) -> list[tuple[tuple[int, ...], ContextSource, int]]:
        payloads = [_item_payload(item) for item in items]
        call_names: dict[str, tuple[str, int]] = {}
        output_indexes: dict[str, int] = {}
        for index, payload in enumerate(payloads):
            item_type = str(payload.get("type", ""))
            call_id = str(payload.get("call_id", ""))
            if item_type == "function_call" and call_id:
                call_names[call_id] = (str(payload.get("name", "")), index)
            elif item_type == "function_call_output" and call_id:
                output_indexes[call_id] = index
        groups: list[tuple[tuple[int, ...], ContextSource, int]] = []
        grouped_indexes: set[int] = set()
        last_index = len(items) - 1
        for call_id, (name, call_index) in call_names.items():
            output_index = output_indexes.get(call_id)
            indices = (call_index,) if output_index is None else (call_index, output_index)
            grouped_indexes.update(indices)
            if 0 in indices or last_index in indices:
                continue
            category, priority = _tool_category(name)
            groups.append((indices, category, priority))
        for index, payload in enumerate(payloads):
            if index in grouped_indexes or index in {0, last_index}:
                continue
            item_type = str(payload.get("type", ""))
            category = ContextSource.HISTORY
            priority = 20 if item_type == "reasoning" else 40
            groups.append(((index,), category, priority))
        return sorted(groups, key=lambda group: (group[2], max(group[0])))

    def _report(
        self,
        items: Sequence[ContextItem],
        initial_tokens: int,
        final_tokens: int,
        reduction_trace: list[str],
        *,
        pruned_items: int = 0,
    ) -> ContextBudgetReport:
        category_tokens: dict[ContextSource, int] = defaultdict(int)
        for item in items:
            category_tokens[item.source] += self.counter.count(item.content)
        return ContextBudgetReport(
            initial_tokens=initial_tokens,
            final_tokens=final_tokens,
            budget_tokens=self.budget_tokens,
            category_tokens=dict(category_tokens),
            reduction_trace=reduction_trace,
            pruned_items=pruned_items,
        )


def _item_payload(item: object) -> dict[str, object]:
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")  # type: ignore[no-any-return, union-attr]
    return {"type": item.__class__.__name__, "value": str(item)}


def _text_fields(item: object) -> list[tuple[str, str]]:
    payload = _item_payload(item)
    return [
        (field, value)
        for field in ("content", "output")
        if isinstance((value := payload.get(field)), str)
    ]


def _replace_text_field(item: object, field: str, value: str) -> object:
    if isinstance(item, dict):
        replacement = dict(item)
        replacement[field] = value
        return replacement
    model_copy = getattr(item, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={field: value})
    replacement = _item_payload(item)
    replacement[field] = value
    return replacement


def _tool_category(name: str) -> tuple[ContextSource, int]:
    if name == "run_tests":
        return ContextSource.TEST, 80
    if name == "explore_repository":
        return ContextSource.EXPLORER, 75
    if name == "review_changes":
        return ContextSource.REVIEWER, 65
    return ContextSource.TOOL, 30
