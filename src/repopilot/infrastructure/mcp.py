from __future__ import annotations

import re
import sys
from typing import Any

from agents.mcp import MCPServer, MCPServerStdio, MCPServerStreamableHttp

from repopilot.core.config import Settings

_MUTATING_TOOL = re.compile(
    r"(?:^|_)(?:add|assign|close|comment|create|delete|edit|merge|remove|reopen|"
    r"resolve|submit|update|write)(?:_|$)",
    re.IGNORECASE,
)


def github_read_only_filter(context: Any, tool: Any) -> bool:
    """Expose GitHub discovery tools while rejecting names that imply mutation."""

    del context
    name = str(getattr(tool, "name", ""))
    return bool(name) and _MUTATING_TOOL.search(name) is None


def build_mcp_servers(settings: Settings) -> list[MCPServer]:
    """Build opt-in external MCP servers; local repository tools do not use MCP."""

    servers: list[MCPServer] = []
    if settings.issue_mcp_enabled:
        servers.append(
            MCPServerStdio(
                params={
                    "command": sys.executable,
                    "args": ["-m", "repopilot.infrastructure.issue_mcp_server"],
                },
                name="Issue MCP",
                cache_tools_list=True,
                require_approval="never",
                max_retry_attempts=1,
            )
        )
    if settings.github_mcp_url:
        headers: dict[str, str] = {}
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token.get_secret_value()}"
        servers.append(
            MCPServerStreamableHttp(
                params={"url": settings.github_mcp_url, "headers": headers, "timeout": 15},
                name="GitHub MCP",
                cache_tools_list=True,
                tool_filter=github_read_only_filter,
                require_approval="never",
                max_retry_attempts=2,
            )
        )
    return servers
