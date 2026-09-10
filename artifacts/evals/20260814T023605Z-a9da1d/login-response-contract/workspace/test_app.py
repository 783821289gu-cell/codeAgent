from http import HTTPStatus

from app import login_endpoint


def test_login_returns_existing_user() -> None:
    response = login_endpoint("alice")

    assert response.status == HTTPStatus.OK
    assert response.body == {"id": "alice", "name": "Alice"}


def test_login_returns_client_error_for_missing_user() -> None:
    response = login_endpoint("unknown")

    assert response.status == HTTPStatus.NOT_FOUND
    assert response.body == {"error": "user not found"}

