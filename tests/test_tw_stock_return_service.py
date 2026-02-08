from __future__ import annotations

import unittest
from dataclasses import dataclass

from logic.market_data_client import PricePoint
from logic.tw_stock_return_service import StockQueryError, compute_annual_return, resolve_stock


@dataclass
class FakeClient:
    stock_info: list[dict[str, str]]
    history: list[PricePoint]

    def get_taiwan_stock_info(self) -> list[dict[str, str]]:
        return self.stock_info

    def get_price_history(self, **_) -> list[PricePoint]:
        return self.history


class TwStockReturnServiceTests(unittest.TestCase):
    def test_resolve_stock_by_id_success(self) -> None:
        client = FakeClient(stock_info=[{"stock_id": "2330", "stock_name": "台積電"}], history=[])
        resolved = resolve_stock("2330", client)
        self.assertEqual(resolved.stock_id, "2330")
        self.assertEqual(resolved.stock_name, "台積電")

    def test_resolve_stock_invalid_numeric_length(self) -> None:
        client = FakeClient(stock_info=[{"stock_id": "2330", "stock_name": "台積電"}], history=[])
        with self.assertRaises(StockQueryError) as ctx:
            resolve_stock("123", client)
        self.assertEqual(ctx.exception.error_code, "INVALID_QUERY")

    def test_resolve_stock_name_not_found(self) -> None:
        client = FakeClient(stock_info=[{"stock_id": "2330", "stock_name": "台積電"}], history=[])
        with self.assertRaises(StockQueryError) as ctx:
            resolve_stock("不存在公司", client)
        self.assertEqual(ctx.exception.error_code, "STOCK_NOT_FOUND")

    def test_compute_annual_return_success(self) -> None:
        client = FakeClient(
            stock_info=[],
            history=[
                PricePoint(data_id="2330", date="2025-01-01", close=500.0),
                PricePoint(data_id="2330", date="2026-02-07", close=600.0),
            ],
        )
        result = compute_annual_return("2330", client)
        self.assertEqual(result.price_base, 500.0)
        self.assertEqual(result.price_latest, 600.0)
        self.assertAlmostEqual(result.annual_return, 0.2, places=6)

    def test_compute_annual_return_no_base_price(self) -> None:
        client = FakeClient(
            stock_info=[],
            history=[PricePoint(data_id="2330", date="2026-01-30", close=600.0)],
        )
        with self.assertRaises(StockQueryError) as ctx:
            compute_annual_return("2330", client)
        self.assertEqual(ctx.exception.error_code, "NO_PRICE_DATA")


if __name__ == "__main__":
    unittest.main()
