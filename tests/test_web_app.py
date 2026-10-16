"""Check public HTTP input bounds and the demo asset allowlist.
The tests use Flask loopback requests and the real simulator.
No external integrations or saved results are used.
"""
import pytest
from app import app


@pytest.fixture
def client():
    return app.test_client()


def test_http_runs_engine_and_exposes_security_headers(client):
    response = client.post("/api/simulate", json={"particles": 8, "steps": 20})
    assert response.status_code == 200
    assert response.json["mode"] == "live_cpu_simulation"
    assert response.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


@pytest.mark.parametrize("raw", [b"{", b"null", b'{"seed":1,"seed":2}',
    b'{"noise":NaN}', b'{"seed":' + b'9'*1000 + b'}', b'['*1500 + b']'*1500])
def test_bad_json_and_giant_numbers_return_400(client, raw):
    assert client.post("/api/simulate", data=raw, content_type="application/json").status_code == 400


def test_size_type_and_method_bounds(client):
    assert client.post("/api/simulate", data=b"{}", content_type="text/plain").status_code == 415
    assert client.post("/api/simulate", data=b" "*4097, content_type="application/json").status_code == 413
    assert client.get("/api/simulate").status_code == 405


def test_serves_only_public_assets(client):
    for path in ["/", "/study", "/app.js", "/live.js", "/style.css", "/live.css", "/data.json"]:
        assert client.get(path).status_code == 200
    for path in ["/simulate.py", "/app.py", "/.env", "/../simulate.py", "/tests/test_live_demo.py"]:
        assert client.get(path).status_code == 404
