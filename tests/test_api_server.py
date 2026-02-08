from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

from logic.market_data_client import FinMindApiError, PricePoint

try:
    from fastapi.testclient import TestClient
    from logic.api_server import app
    HAS_FASTAPI = True
except ModuleNotFoundError:
    HAS_FASTAPI = False


@dataclass
class FakeClient:
    stock_info: list[dict[str, str]]
    history: list[PricePoint]
    raise_api_error: bool = False

    def get_taiwan_stock_info(self) -> list[dict[str, str]]:
        if self.raise_api_error:
            raise FinMindApiError("upstream error")
        return self.stock_info

    def get_price_history(self, **_) -> list[PricePoint]:
        if self.raise_api_error:
            raise FinMindApiError("upstream error")
        return self.history


class ApiServerTests(unittest.TestCase):
    def setUp(self) -> None:
        if not HAS_FASTAPI:
            self.skipTest("fastapi is not installed")
        self.client = TestClient(app)

    def test_api_annual_return_success(self) -> None:
        fake = FakeClient(
            stock_info=[{"stock_id": "2330", "stock_name": "台積電"}],
            history=[
                PricePoint(data_id="2330", date="2025-01-01", close=500.0),
                PricePoint(data_id="2330", date="2026-02-07", close=600.0),
            ],
        )
        with patch("logic.api_server.FinMindClient.from_env", return_value=fake):
            response = self.client.post("/api/v1/tw-stock/annual-return", json={"query": "台積電"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["resolved_stock_id"], "2330")
        self.assertEqual(payload["resolved_stock_name"], "台積電")
        self.assertAlmostEqual(payload["annual_return"], 0.2, places=6)

    def test_api_invalid_query_returns_422_shape(self) -> None:
        fake = FakeClient(stock_info=[{"stock_id": "2330", "stock_name": "台積電"}], history=[])
        with patch("logic.api_server.FinMindClient.from_env", return_value=fake):
            response = self.client.post("/api/v1/tw-stock/annual-return", json={"query": "123"})
        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertEqual(payload["error_code"], "INVALID_QUERY")
        self.assertIn("message", payload)
        self.assertIn("details", payload)

    def test_api_upstream_error_returns_502_shape(self) -> None:
        fake = FakeClient(stock_info=[], history=[], raise_api_error=True)
        with patch("logic.api_server.FinMindClient.from_env", return_value=fake):
            response = self.client.post("/api/v1/tw-stock/annual-return", json={"query": "2330"})
        self.assertEqual(response.status_code, 502)
        payload = response.json()
        self.assertEqual(payload["error_code"], "UPSTREAM_ERROR")
        self.assertIn("message", payload)
        self.assertIn("details", payload)

    def test_api_missing_token_returns_500(self) -> None:
        with patch("logic.api_server.FinMindClient.from_env", return_value=None):
            response = self.client.post("/api/v1/tw-stock/annual-return", json={"query": "2330"})
        self.assertEqual(response.status_code, 500)
        payload = response.json()
        self.assertEqual(payload["error_code"], "CONFIG_ERROR")


if __name__ == "__main__":
    unittest.main()
