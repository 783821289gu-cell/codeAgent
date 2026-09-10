from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from agents import Agent
from agents.lifecycle import RunHooksBase
from agents.run_context import RunContextWrapper
from agents.tool import (
    FunctionTool,
    Tool,
    ToolOriginType,
    get_function_tool_origin,
)

from repopilot.core.models import TraceEvent
from repopilot.infrastructure.postgres import TraceStore, json_details
from repopilot.knowledge.tokens import TokenCounter


class LocalTracer:
    """Persists application/tool/test events alongside the SDK's model-level trace."""

    def __init__(self, store: TraceStore, trace_id: str, task_id: str) -> None:
        self.store = store
        self.trace_id = trace_id
        self.task_id = task_id

    @contextmanager
    def span(self, category: str, name: str, **details: Any) -> Iterator[dict[str, Any]]:
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        mutable_details = json_details(**details)
        ok = True
        try:
            yield mutable_details
        except Exception as exc:
            ok = False
            mutable_details["error"] = str(exc)
            raise
        finally:
            self.store.append(
                TraceEvent(
                    trace_id=self.trace_id,
                    task_id=self.task_id,
                    category=category,
                    name=name,
                    started_at=started_at,
                    duration_ms=(time.perf_counter() - started) * 1_000,
                    ok=ok,
                    details=mutable_details,
                )
            )


class LocalRunHooks(RunHooksBase[Any, Agent]):
    """Add SDK-originated MCP calls to the persisted local trace."""

    def __init__(self, tracer: LocalTracer, counter: TokenCounter) -> None:
        self.tracer = tracer
        self.counter = counter

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        tool: Tool,
        result: object,
    ) -> None:
        del agent
        if not isinstance(tool, FunctionTool):
            return
        origin = get_function_tool_origin(tool)
        if origin is None or origin.type != ToolOriginType.MCP:
            return
        rendered = str(result)
        with self.tracer.span(
            "mcp",
            tool.name,
            server=origin.mcp_server_name,
            call_id=str(getattr(context, "tool_call_id", "")),
            result_tokens=self.counter.count(rendered),
            result_excerpt=self.counter.truncate_prefix(rendered, 256),
        ):
            pass
