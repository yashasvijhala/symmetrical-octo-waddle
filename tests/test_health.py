from fastapi.testclient import TestClient

from forecasting_service.main import app


client = TestClient(app)


def test_liveness() -> None:
    response = client.get("/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness() -> None:
    response = client.get("/v1/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
