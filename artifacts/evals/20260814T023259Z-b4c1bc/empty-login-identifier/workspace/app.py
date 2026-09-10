from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from wsgiref.simple_server import make_server


@dataclass(frozen=True)
class Response:
    status: int
    body: dict[str, Any]


USERS = {
    "alice": {"name": "Alice"},
}


def login_endpoint(user_id: str) -> Response:
    """Return public login profile data for an existing user."""

    user = USERS.get(user_id)
    # Deliberate demo bug: an unknown user is dereferenced and becomes HTTP 500 in wsgi_app.
    return Response(status=HTTPStatus.OK, body={"id": user_id, "name": user["name"]})


def wsgi_app(environ: dict[str, Any], start_response: Any) -> list[bytes]:
    user_id = environ.get("PATH_INFO", "").removeprefix("/login/")
    try:
        response = login_endpoint(user_id)
    except Exception:
        response = Response(
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            body={"error": "internal server error"},
        )
    phrase = HTTPStatus(response.status).phrase
    start_response(f"{response.status} {phrase}", [("Content-Type", "application/json")])
    return [json.dumps(response.body).encode("utf-8")]


if __name__ == "__main__":
    with make_server("127.0.0.1", 8000, wsgi_app) as server:
        print("Demo API listening on http://127.0.0.1:8000/login/alice")
        server.serve_forever()

