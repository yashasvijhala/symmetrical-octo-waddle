from __future__ import annotations

import argparse
import csv
import io
import json
import time
from datetime import date, timedelta
from typing import Any

import httpx


def _sample_csv() -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["sku", "date", "sales", "category", "promo", "importance"],
    )
    writer.writeheader()
    start = date(2026, 1, 1)
    for sku, category, base in (
        ("coffee", "beverages", 24),
        ("tea", "beverages", 18),
        ("cookies", "snacks", 31),
    ):
        for index in range(70):
            weekly = (index % 7) * 0.8
            promotion = int(index % 14 in {0, 1})
            writer.writerow(
                {
                    "sku": sku,
                    "date": start + timedelta(days=index),
                    "sales": round(base + weekly + promotion * 7 + index * 0.05, 2),
                    "category": category,
                    "promo": promotion,
                    "importance": 2 if sku == "cookies" else 1,
                }
            )
    return output.getvalue().encode()


class DemoSeeder:
    def __init__(self, api_url: str, tenant_id: str, timeout_seconds: int) -> None:
        self.base_url = api_url.rstrip("/") + "/v1"
        self.tenant_id = tenant_id
        self.timeout_seconds = timeout_seconds
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"X-Tenant-ID": tenant_id},
            timeout=30,
        )

    def close(self) -> None:
        self.client.close()

    def seed(self) -> dict[str, Any]:
        dataset = self._create_dataset()
        dataset_id = dataset["id"]
        if dataset["state"] != "ready":
            self._request(
                "POST",
                f"/datasets/{dataset_id}/upload",
                files={"file": ("demo-sales.csv", _sample_csv(), "text/csv")},
            )
            self._request("POST", f"/datasets/{dataset_id}/profile")
            self._request(
                "POST",
                f"/datasets/{dataset_id}/finalize",
                json={
                    "timestamp_column": "date",
                    "target_column": "sales",
                    "item_id_column": "sku",
                    "horizon": 7,
                    "non_negative": True,
                    "column_roles": {
                        "category": "hierarchy",
                        "promo": "known_future",
                        "importance": "weight",
                    },
                },
            )

        experiment = self._request(
            "POST",
            "/experiments",
            headers={"Idempotency-Key": "demo-lightgbm-experiment-v1"},
            json={
                "dataset_id": dataset_id,
                "dataset_version": 1,
                "horizon": 7,
                "validation_windows": 2,
                "model_policy": "fast",
                "time_limit_seconds": 60,
                "num_threads": 2,
            },
        )
        experiment = self._wait(f"/experiments/{experiment['id']}")
        model_id = experiment["model_id"]
        self._request(
            "POST",
            f"/models/{model_id}/promote",
            json={"force": True, "justification": "local demo fixture"},
        )

        first_future_day = date(2026, 3, 12)
        future_covariates = [
            {
                "item_id": sku,
                "timestamp": (first_future_day + timedelta(days=step)).isoformat() + "T00:00:00",
                "promo": int(step in {5, 6}),
            }
            for sku in ("coffee", "tea", "cookies")
            for step in range(7)
        ]
        forecast = self._request(
            "POST",
            "/forecasts",
            headers={"Idempotency-Key": "demo-lightgbm-forecast-v1"},
            json={
                "model_id": model_id,
                "future_covariates": future_covariates,
                "new_items": [
                    {
                        "item_id": "granola",
                        "level_prior": 20,
                        "metadata": {"category": "snacks"},
                    }
                ],
            },
        )
        forecast = self._wait(f"/forecasts/{forecast['id']}")
        rows = self._request("GET", f"/forecasts/{forecast['id']}/download")

        monitoring = self._request("GET", f"/models/{model_id}/monitoring")
        if monitoring["matched_points"] == 0:
            observed = rows[0]
            self._request(
                "POST",
                "/actuals",
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

        return {
            "tenant_id": self.tenant_id,
            "dataset_id": dataset_id,
            "experiment_id": experiment["id"],
            "job_id": experiment["job_id"],
            "model_id": model_id,
            "forecast_id": forecast["id"],
            "forecast_rows": forecast["summary"]["rows"],
        }

    def _create_dataset(self) -> dict[str, Any]:
        created = self._request(
            "POST",
            "/datasets",
            headers={"Idempotency-Key": "demo-sales-dataset-v1"},
            json={
                "name": "Demo retail demand",
                "description": "Generated daily demand for local development",
            },
        )
        return self._request("GET", f"/datasets/{created['id']}")

    def _wait(self, path: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            resource = self._request("GET", path)
            if resource["state"] == "succeeded":
                return resource
            if resource["state"] in {"failed", "cancelled"}:
                raise RuntimeError(
                    f"{path} ended as {resource['state']}: {resource.get('error', 'no error')}"
                )
            time.sleep(0.25)
        raise TimeoutError(f"timed out after {self.timeout_seconds}s waiting for {path}")

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.client.request(method, path, **kwargs)
        if response.is_error:
            raise RuntimeError(f"{method} {path} returned {response.status_code}: {response.text}")
        return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed an idempotent local forecasting demo")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tenant", default="demo")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    seeder = DemoSeeder(args.api_url, args.tenant, args.timeout)
    try:
        print(json.dumps(seeder.seed(), indent=2))
    finally:
        seeder.close()


if __name__ == "__main__":
    main()
