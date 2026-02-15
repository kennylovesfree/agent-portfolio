from __future__ import annotations

import unittest

from logic.portfolio_health_service import PortfolioInputError, evaluate_portfolio_health


class PortfolioHealthServiceTests(unittest.TestCase):
    def test_health_check_flags_concentration_and_bounds_score(self) -> None:
        result = evaluate_portfolio_health(
            positions=[
                {"ticker": "AAPL", "weight": 0.62},
                {"ticker": "BND", "weight": 0.2},
                {"ticker": "CASH", "weight": 0.18},
            ],
            portfolio={"volatility": 0.29, "maxDrawdown": 0.36},
        )
        self.assertIn("OVER_CONCENTRATION", result.flags)
        self.assertTrue(0 <= result.health_score <= 100)
        self.assertEqual(result.risk_band, "medium")

    def test_health_check_invalid_weight_sum_raises(self) -> None:
        with self.assertRaises(PortfolioInputError):
            evaluate_portfolio_health(
                positions=[{"ticker": "AAPL", "weight": 0.0}],
                portfolio={"volatility": 0.15, "maxDrawdown": 0.1},
            )


if __name__ == "__main__":
    unittest.main()
