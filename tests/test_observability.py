"""HW2 authentication tests. Offline: no Langfuse, Docker, or model key."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import app as server_app


@pytest.fixture
def client(world):
    """In-process HTTP client against the seeded temp database.

    Not used as a context manager, so the lifespan hook (tracing setup) does
    not run and the tests stay offline.
    """
    server_app._SESSIONS.clear()
    return TestClient(server_app.app)


def test_create_session_rejects_role_mismatch(client):
    # User 1 is a shopper in the database; claiming "support" must be refused.
    response = client.post("/sessions", json={"user_id": 1, "role": "support"})

    assert response.status_code == 403
    assert server_app._SESSIONS == {}  # no session was created


def test_token_for_one_session_cannot_authorize_another(client, monkeypatch):
    async def agent_must_not_run(*args, **kwargs):
        raise AssertionError("agent ran despite a token for another session")

    monkeypatch.setattr(server_app.Runner, "run", agent_must_not_run)

    a = client.post("/sessions", json={"user_id": 1, "role": "shopper"}).json()
    b = client.post("/sessions", json={"user_id": 9002, "role": "merchant"}).json()

    response = client.post(
        f"/sessions/{b['session_id']}/messages",
        headers={"Authorization": f"Bearer {a['token']}"},
        json={"message": "Show my recent orders."},
    )

    assert response.status_code == 403
