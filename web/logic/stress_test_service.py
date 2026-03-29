"""Stress testing helpers for deterministic scenario analysis."""
from __future__ import annotations

from dataclasses import dataclass


class PortfolioInputError(ValueError):
    """Raised when stress test payload is structurally invalid."""


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    portfolio_pnl_usd: float
    drawdown_est: float
    risk_label: str


@dataclass(frozen=True)
class StressTestResult:
    scenario_results: list[ScenarioResult]
    worst_case: ScenarioResult
    survival_days_est: int


_BOND_TICKERS = {"BND", "AGG", "TLT", "IEF", "00679B", "00865B", "00720B", "LQD"}
_CASH_TICKERS = {"CASH", "USD", "USDC", "MMF"}

_DEFAULT_SCENARIOS: dict[str, dict[str, float]] = {
    "GLOBAL_EQUITY_-15": {"stock": -0.15, "bond": 0.0, "cash": 0.0},
    "RATE_SHOCK_BOND_-8": {"stock": 0.0, "bond": -0.08, "cash": 0.0},
    "FX_USD_TWD_+5": {"stock": 0.0, "bond": 0.0, "cash": 0.0, "us_market": 0.05},
    "COMBINED_RISK_OFF": {"stock": -0.12, "bond": -0.04, "cash": 0.0},
}


def _classify_bucket(position: dict) -> str:
    ticker = str(position.get("ticker", "")).strip().upper()
    if not ticker or ticker in _CASH_TICKERS:
        return "cash"
    if ticker in _BOND_TICKERS:
        return "bond"
    return "stock"


def _risk_label(drawdown_est: float) -> str:
    if drawdown_est < 0.08:
        return "low"
    if drawdown_est < 0.16:
        return "medium"
    return "high"


def _survival_days_estimate(worst_drawdown: float) -> int:
    if worst_drawdown >= 0.25:
        return 30
    if worst_drawdown >= 0.18:
        return 60
    if worst_drawdown >= 0.12:
        return 120
    return 240


def _validate_positions(positions: list[dict]) -> None:
    if not positions:
        raise PortfolioInputError("positions 不可為空。")

    for idx, position in enumerate(positions):
        try:
            amount = float(position.get("amountUsd", 0))
            weight = float(position.get("weight", 0))
        except (TypeError, ValueError) as exc:
            raise PortfolioInputError(f"positions[{idx}] 欄位格式不正確。") from exc

        if amount < 0:
            raise PortfolioInputError(f"positions[{idx}].amountUsd 不可小於 0。")
        if weight < 0 or weight > 1:
            raise PortfolioInputError(f"positions[{idx}].weight 需介於 0 與 1。")


def _resolve_scenario_ids(scenarios: list[dict] | None) -> list[str]:
    if not scenarios:
        return list(_DEFAULT_SCENARIOS.keys())

    ids: list[str] = []
    for idx, scenario in enumerate(scenarios):
        scenario_id = str(scenario.get("scenario_id", "")).strip().upper()
        if not scenario_id:
            raise PortfolioInputError(f"scenarios[{idx}].scenario_id 不可為空。")
        if scenario_id not in _DEFAULT_SCENARIOS:
            raise PortfolioInputError(f"scenarios[{idx}].scenario_id 不支援：{scenario_id}")
        ids.append(scenario_id)

    # Keep stable order while deduplicating.
    return list(dict.fromkeys(ids))


def run_stress_test(positions: list[dict], scenarios: list[dict] | None = None) -> StressTestResult:
    """Run deterministic stress tests across predefined scenario pack."""
    _validate_positions(positions)
    scenario_ids = _resolve_scenario_ids(scenarios)

    total_usd = sum(float(position.get("amountUsd", 0)) for position in positions)
    if total_usd <= 0:
        raise PortfolioInputError("positions.amountUsd 總和需大於 0。")

    results: list[ScenarioResult] = []
    for scenario_id in scenario_ids:
        shocks = _DEFAULT_SCENARIOS[scenario_id]
        portfolio_pnl = 0.0

        for position in positions:
            amount_usd = float(position.get("amountUsd", 0))
            market = str(position.get("market", "TW")).strip().upper()
            bucket = _classify_bucket(position)
            shock = shocks.get(bucket, 0.0)
            if scenario_id == "FX_USD_TWD_+5" and market == "US":
                shock = shocks.get("us_market", 0.0)
            portfolio_pnl += amount_usd * shock

        drawdown_est = max(0.0, -portfolio_pnl / total_usd)
        results.append(
            ScenarioResult(
                scenario_id=scenario_id,
                portfolio_pnl_usd=round(portfolio_pnl, 2),
                drawdown_est=round(drawdown_est, 4),
                risk_label=_risk_label(drawdown_est),
            )
        )

    worst_case = min(results, key=lambda item: item.portfolio_pnl_usd)
    survival_days_est = _survival_days_estimate(worst_case.drawdown_est)
    return StressTestResult(
        scenario_results=results,
        worst_case=worst_case,
        survival_days_est=survival_days_est,
    )
