from http import HTTPStatus

from app import login_endpoint


def test_login_returns_existing_user() -> None:
    response = login_endpoint("alice")

    assert response.status == HTTPStatus.OK
    assert response.body == {"id": "alice", "name": "Alice"}

