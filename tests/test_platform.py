import csv
import io
import time
from datetime import date, timedelta

from fastapi.testclient import TestClient

from forecasting_service.config import Settings
from forecasting_service.main import create_app

HEADERS = {"X-Tenant-ID": "tenant-a"}


def dataset_csv() -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=["sku", "date", "sales", "category", "promo", "importance"]
    )
    writer.writeheader()
    start = date(2026, 1, 1)
    for sku, base in (("a", 10), ("b", 20)):
        for index in range(42):
            writer.writerow(
                {
                    "sku": sku,
                    "date": start + timedelta(days=index),
                    "sales": base + index % 7 + index * 0.1,
                    "category": "core",
                    "promo": int(index % 10 == 0),
                    "importance": 2 if sku == "b" else 1,
                }
            )
    return output.getvalue().encode()


def wait_for(client: TestClient, path: str) -> dict:
    for _ in range(200):
        payload = client.get(path, headers=HEADERS).json()
        if payload["state"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def test_full_lightgbm_workflow(tmp_path, database_url: str) -> None:
    app = create_app(
        Settings(
            environment="test",
            state_dir=tmp_path,
            database_url=database_url,
            test_max_workers=1,
        )
    )
    with TestClient(app) as client:
        created = client.post("/v1/datasets", headers=HEADERS, json={"name": "sales"})
        assert created.status_code == 201
        dataset_id = created.json()["id"]

        uploaded = client.post(
            f"/v1/datasets/{dataset_id}/upload",
            headers=HEADERS,
            files={"file": ("sales.csv", dataset_csv(), "text/csv")},
        )
        assert uploaded.status_code == 200

        profile = client.post(f"/v1/datasets/{dataset_id}/profile", headers=HEADERS)
        assert profile.status_code == 201
        assert profile.json()["proposal"]["target_column"] == "sales"

        finalized = client.post(
            f"/v1/datasets/{dataset_id}/finalize",
            headers=HEADERS,
            json={
                "timestamp_column": "date",
                "target_column": "sales",
                "item_id_column": "sku",
                "horizon": 3,
                "non_negative": True,
                "column_roles": {
                    "category": "hierarchy",
                    "promo": "known_future",
                    "importance": "weight",
                },
            },
        )
        assert finalized.status_code == 201, finalized.text
        assert finalized.json()["items"] == 2

        experiment = client.post(
            "/v1/experiments",
            headers=HEADERS,
            json={
                "dataset_id": dataset_id,
                "horizon": 3,
                "validation_windows": 1,
                "model_policy": "fast",
            },
        )
        assert experiment.status_code == 202
        result = wait_for(client, f"/v1/experiments/{experiment.json()['id']}")
        assert result["state"] == "succeeded", result.get("error", result)
        model_id = result["model_id"]

        invalid_forecast = client.post(
            "/v1/forecasts", headers=HEADERS, json={"model_id": model_id}
        )
        invalid_result = wait_for(client, f"/v1/forecasts/{invalid_forecast.json()['id']}")
        assert invalid_result["state"] == "failed"
        assert "future covariates are incomplete" in invalid_result["error"]

        future_covariates = [
            {
                "item_id": sku,
                "timestamp": (date(2026, 2, 12) + timedelta(days=step)).isoformat() + "T00:00:00",
                "promo": 0,
            }
            for sku in ("a", "b")
            for step in range(3)
        ]

        forecast = client.post(
            "/v1/forecasts",
            headers=HEADERS,
            json={
                "model_id": model_id,
                "new_items": [
                    {
                        "item_id": "new-sku",
                        "level_prior": 15,
                        "metadata": {"category": "core"},
                    }
                ],
                "future_covariates": future_covariates,
            },
        )
        assert forecast.status_code == 202
        forecast_result = wait_for(client, f"/v1/forecasts/{forecast.json()['id']}")
        assert forecast_result["state"] == "succeeded", forecast_result
        assert forecast_result["summary"] == {"rows": 12, "items": 4}

        downloaded = client.get(f"/v1/forecasts/{forecast.json()['id']}/download", headers=HEADERS)
        assert downloaded.status_code == 200
        rows = downloaded.json()
        assert any(row["method"] == "cold_start_metadata_neighbors" for row in rows)
        assert any(row["method"] == "bottom_up_reconciled" for row in rows)

        promoted = client.post(
            f"/v1/models/{model_id}/promote",
            headers=HEADERS,
            json={"force": True, "justification": "integration test override"},
        )
        assert promoted.status_code == 200
        assert promoted.json()["stage"] == "champion"

        observed = rows[0]
        actuals = client.post(
            "/v1/actuals",
            headers=HEADERS,
            json={
                "model_id": model_id,
                "points": [
                    {
                        "item_id": observed["item_id"],
                        "timestamp": observed["timestamp"],
                        "value": observed["mean"],
                    }
                ],
            },
        )
        assert actuals.status_code == 201
        monitoring = client.get(f"/v1/models/{model_id}/monitoring", headers=HEADERS)
        assert monitoring.json()["matched_points"] == 1
        assert monitoring.json()["mae"] == 0


def test_tenant_isolation(tmp_path, database_url: str) -> None:
    app = create_app(Settings(environment="test", state_dir=tmp_path, database_url=database_url))
    with TestClient(app) as client:
        created = client.post("/v1/datasets", headers=HEADERS, json={"name": "private"})
        dataset_id = created.json()["id"]
        hidden = client.get(f"/v1/datasets/{dataset_id}", headers={"X-Tenant-ID": "tenant-b"})
        assert hidden.status_code == 404


def test_production_authentication_and_idempotency(tmp_path, database_url: str) -> None:
    app = create_app(
        Settings(
            environment="production",
            state_dir=tmp_path,
            database_url=database_url,
            api_keys={"tenant-a": "secret"},
        )
    )
    with TestClient(app) as client:
        assert client.post("/v1/datasets", json={"name": "sales"}).status_code == 401
        assert (
            client.post(
                "/v1/datasets",
                headers={"X-Tenant-ID": "tenant-a", "X-API-Key": "wrong"},
                json={"name": "sales"},
            ).status_code
            == 401
        )
        headers = {
            "X-Tenant-ID": "tenant-a",
            "X-API-Key": "secret",
            "Idempotency-Key": "create-sales-once",
        }
        first = client.post("/v1/datasets", headers=headers, json={"name": "sales"})
        second = client.post("/v1/datasets", headers=headers, json={"name": "sales"})
        conflict = client.post("/v1/datasets", headers=headers, json={"name": "other"})
        assert first.status_code == 201
        assert second.status_code == 201
        assert first.json()["id"] == second.json()["id"]
        assert conflict.status_code == 409
