from http import HTTPStatus

from app import login_endpoint


def test_login_returns_existing_user() -> None:
    response = login_endpoint("alice")

    assert response.status == HTTPStatus.OK
    assert response.body == {"id": "alice", "name": "Alice"}


def test_login_returns_client_error_for_empty_identifier() -> None:
    response = login_endpoint("")

    assert response.status == HTTPStatus.BAD_REQUEST
    assert response.body == {"error": "login identifier is required"}

