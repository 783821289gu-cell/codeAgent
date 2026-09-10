from http import HTTPStatus

from app import login_endpoint, wsgi_app


def test_login_returns_existing_user() -> None:
    response = login_endpoint("alice")

    assert response.status == HTTPStatus.OK
    assert response.body == {"id": "alice", "name": "Alice"}


def test_login_missing_user_returns_client_error() -> None:
    response = login_endpoint("unknown")

    assert response.status == HTTPStatus.NOT_FOUND
    assert response.body == {"error": "user not found"}


def test_wsgi_missing_user_is_not_500() -> None:
    captured = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = headers

    body = wsgi_app({"PATH_INFO": "/login/unknown"}, start_response)

    assert captured["status"] == "404 Not Found"
    assert b"user not found" in body[0]

