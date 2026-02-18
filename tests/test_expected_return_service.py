from __future__ import annotations

import unittest

from logic.expected_return_service import (
    classify_asset_class,
    compute_historical_weight,
    estimate_expected_return,
)


class ExpectedReturnServiceTests(unittest.TestCase):
    def _profile(self, years: int = 30) -> dict:
        return {
            "age": 35,
            "riskLevel": "balanced",
            "taxRegion": "TW",
            "horizonYears": years,
        }

    def test_30y_high_historical_signal_is_capped(self) -> None:
        payload = estimate_expected_return(
            profile=self._profile(years=30),
            positions=[
                {
                    "ticker": "AAPL",
                    "market": "US",
                    "weight": 1.0,
                    "expectedReturn": 0.50,
                    "isCash": False,
                }
            ],
            coverage=1.0,
        )
        self.assertLessEqual(payload["expected_return"], 0.12)

    def test_historical_weight_decays_with_horizon(self) -> None:
        w_1y = compute_historical_weight(1)
        w_30y = compute_historical_weight(30)
        self.assertGreater(w_1y, w_30y)

    def test_high_concentration_penalty_applies(self) -> None:
        payload = estimate_expected_return(
            profile=self._profile(years=10),
            positions=[
                {"ticker": "AAPL", "market": "US", "weight": 0.9, "expectedReturn": 0.15, "isCash": False},
                {"ticker": "BND", "market": "US", "weight": 0.1, "expectedReturn": 0.03, "isCash": False},
            ],
            coverage=1.0,
        )
        self.assertGreater(payload["components"]["concentration_penalty"], 0.0)

    def test_low_coverage_penalty_applies(self) -> None:
        payload = estimate_expected_return(
            profile=self._profile(years=10),
            positions=[
                {"ticker": "2330", "market": "TW", "weight": 1.0, "expectedReturn": 0.12, "isCash": False},
            ],
            coverage=0.5,
        )
        self.assertGreater(payload["components"]["coverage_penalty"], 0.0)

    def test_unknown_ticker_defaults_to_equity(self) -> None:
        self.assertEqual(classify_asset_class("MYSTERY", market="TW", is_cash=False), "equity")


if __name__ == "__main__":
    unittest.main()
