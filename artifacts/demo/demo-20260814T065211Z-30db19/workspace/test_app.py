from http import HTTPStatus

from app import login_endpoint, wsgi_app


def test_login_returns_existing_user() -> None:
    response = login_endpoint("alice")

    assert response.status == HTTPStatus.OK
    assert response.body == {"id": "alice", "name": "Alice"}


def test_login_unknown_user_returns_not_found() -> None:
    response = login_endpoint("unknown")

    assert response.status == HTTPStatus.NOT_FOUND
    assert response.body == {"error": "user not found"}


def test_wsgi_app_unknown_user_does_not_return_500() -> None:
    captured: dict[str, object] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = headers

    body = wsgi_app({"PATH_INFO": "/login/unknown"}, start_response)

    assert captured["status"] == "404 Not Found"
    assert body == [b'{"error": "user not found"}']
    assert captured["headers"] == [("Content-Type", "application/json")]

