from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from nova.server import RunRequest, app


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_run_request_model_defaults():
    req = RunRequest(task="hello")
    assert req.task == "hello"
    assert req.max_iterations == 8
    assert req.temperature == 0.0
