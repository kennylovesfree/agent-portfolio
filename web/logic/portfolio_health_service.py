"""Risk explainability helpers for portfolio health scoring."""
from __future__ import annotations

from dataclasses import dataclass


class PortfolioInputError(ValueError):
    """Raised when portfolio payload is structurally invalid."""


@dataclass(frozen=True)
class HealthComponents:
    concentration_score: int
    diversification_score: int
    volatility_score: int
    drawdown_score: int


@dataclass(frozen=True)
class HealthCheckResult:
    health_score: int
    risk_band: str
    components: HealthComponents
    flags: list[str]
    explanations: list[str]


def _bounded_score(base: float) -> int:
    return max(0, min(100, int(round(base))))


def _risk_band(score: int) -> str:
    if score >= 75:
        return "low"
    if score >= 45:
        return "medium"
    return "high"


def _flag_to_explanation(flag: str) -> str:
    mapping = {
        "OVER_CONCENTRATION": "單一標的比重超過 35%，集中風險偏高。",
        "LOW_DIVERSIFICATION": "前三大持倉合計超過 70%，分散度不足。",
        "HIGH_VOLATILITY": "組合年化波動率超過 22%，短期淨值震盪風險上升。",
        "DEEP_DRAWDOWN": "模擬最大回撤超過 30%，請檢查風險承受度與資金配置。",
    }
    return mapping.get(flag, "偵測到風險訊號，建議人工覆核。")


def evaluate_portfolio_health(positions: list[dict], portfolio: dict) -> HealthCheckResult:
    """Compute deterministic health score and explainable risk flags."""
    if not positions:
        raise PortfolioInputError("positions 不可為空。")

    weights: list[float] = []
    for idx, position in enumerate(positions):
        try:
            weight = float(position.get("weight", 0))
        except (TypeError, ValueError) as exc:
            raise PortfolioInputError(f"positions[{idx}].weight 格式不正確。") from exc
        if weight < 0 or weight > 1:
            raise PortfolioInputError(f"positions[{idx}].weight 需介於 0 與 1。")
        weights.append(weight)

    total_weight = sum(weights)
    if total_weight <= 0:
        raise PortfolioInputError("positions.weight 總和需大於 0。")

    try:
        volatility = float(portfolio.get("volatility", 0))
        max_drawdown = float(portfolio.get("maxDrawdown", 0))
    except (TypeError, ValueError) as exc:
        raise PortfolioInputError("portfolio.volatility 或 portfolio.maxDrawdown 格式不正確。") from exc

    if volatility < 0:
        raise PortfolioInputError("portfolio.volatility 不可小於 0。")
    if max_drawdown < 0:
        raise PortfolioInputError("portfolio.maxDrawdown 不可小於 0。")

    max_weight = max(weights)
    top3_weight = sum(sorted(weights, reverse=True)[:3])

    concentration_penalty = 0.0
    if max_weight > 0.35:
        concentration_penalty = min(100.0, ((max_weight - 0.35) / 0.35) * 100.0)

    diversification_penalty = 0.0
    if top3_weight > 0.7:
        diversification_penalty = min(100.0, ((top3_weight - 0.7) / 0.3) * 100.0)

    volatility_penalty = 0.0
    if volatility > 0.22:
        volatility_penalty = min(100.0, ((volatility - 0.22) / 0.28) * 100.0)

    drawdown_penalty = 0.0
    if max_drawdown > 0.30:
        drawdown_penalty = min(100.0, ((max_drawdown - 0.30) / 0.50) * 100.0)

    components = HealthComponents(
        concentration_score=_bounded_score(100.0 - concentration_penalty),
        diversification_score=_bounded_score(100.0 - diversification_penalty),
        volatility_score=_bounded_score(100.0 - volatility_penalty),
        drawdown_score=_bounded_score(100.0 - drawdown_penalty),
    )

    flags: list[str] = []
    if max_weight > 0.35:
        flags.append("OVER_CONCENTRATION")
    if top3_weight > 0.7:
        flags.append("LOW_DIVERSIFICATION")
    if volatility > 0.22:
        flags.append("HIGH_VOLATILITY")
    if max_drawdown > 0.30:
        flags.append("DEEP_DRAWDOWN")

    health_score = _bounded_score(
        (
            components.concentration_score
            + components.diversification_score
            + components.volatility_score
            + components.drawdown_score
        )
        / 4.0
    )

    explanations = [_flag_to_explanation(flag) for flag in flags]
    if not explanations:
        explanations = ["風險指標在可控範圍，仍建議定期檢視配置。"]

    return HealthCheckResult(
        health_score=health_score,
        risk_band=_risk_band(health_score),
        components=components,
        flags=flags,
        explanations=explanations,
    )
