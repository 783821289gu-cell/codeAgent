from __future__ import annotations

from mcp.server.mcpserver import MCPServer

server = MCPServer(
    name="RepoPilot issue demo",
    instructions="Read-only issue tracker used by the RepoPilot MCP integration demo.",
)


@server.tool(description="Read one issue by its numeric identifier.")
def get_issue(issue_number: int) -> dict[str, object]:
    if issue_number != 101:
        raise ValueError(f"issue not found: {issue_number}")
    return {
        "number": 101,
        "title": "Unknown login user becomes HTTP 500",
        "body": (
            "Calling login_endpoint with a user id absent from USERS dereferences None. "
            "Return a non-500 client response, preserve the existing-user HTTP 200 contract, "
            "and add regression coverage."
        ),
        "labels": ["bug", "login", "regression-test"],
        "state": "open",
    }


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
