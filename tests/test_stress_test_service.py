from __future__ import annotations

import unittest

from logic.stress_test_service import PortfolioInputError, run_stress_test


class StressTestServiceTests(unittest.TestCase):
    def test_stress_test_default_pack_and_worst_case(self) -> None:
        result = run_stress_test(
            positions=[
                {"ticker": "AAPL", "market": "US", "amountUsd": 600.0, "weight": 0.6},
                {"ticker": "BND", "market": "US", "amountUsd": 300.0, "weight": 0.3},
                {"ticker": "CASH", "market": "TW", "amountUsd": 100.0, "weight": 0.1},
            ]
        )
        self.assertEqual(len(result.scenario_results), 4)
        self.assertEqual(result.worst_case.scenario_id, "GLOBAL_EQUITY_-15")

    def test_stress_test_invalid_scenario_raises(self) -> None:
        with self.assertRaises(PortfolioInputError):
            run_stress_test(
                positions=[{"ticker": "AAPL", "market": "US", "amountUsd": 1000.0, "weight": 1.0}],
                scenarios=[{"scenario_id": "UNKNOWN_SCENARIO"}],
            )


if __name__ == "__main__":
    unittest.main()
