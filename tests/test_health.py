from fastapi.testclient import TestClient

from forecasting_service.config import Settings
from forecasting_service.main import create_app


def test_liveness(tmp_path, database_url: str) -> None:
    app = create_app(Settings(environment="test", state_dir=tmp_path, database_url=database_url))
    with TestClient(app) as client:
        response = client.get("/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness(tmp_path, database_url: str) -> None:
    app = create_app(Settings(environment="test", state_dir=tmp_path, database_url=database_url))
    with TestClient(app) as client:
        response = client.get("/v1/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
