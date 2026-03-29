"""CMA-convergent expected return model for long-horizon portfolio planning."""
from __future__ import annotations

import math
from typing import Any


class ExpectedReturnInputError(ValueError):
    """Raised when expected return input payload is invalid."""


CASH_TICKERS = {"CASH", "USD", "USDT", "USDC"}
BOND_TICKERS = {
    "BND",
    "AGG",
    "TLT",
    "IEF",
    "LQD",
    "SHY",
    "HYG",
    "TIP",
    "MUB",
    "00679B",
    "00865B",
}

ANCHOR_RETURNS = {
    "tw_equity": 0.07,
    "us_equity": 0.065,
    "bond": 0.03,
    "cash": 0.015,
}

HISTORICAL_BOUNDS = {
    "equity": (-0.20, 0.25),
    "bond": (-0.05, 0.08),
    "cash": (0.00, 0.03),
}

FEE_DRAG = 0.008
MAX_RETURN_CAP = 0.12
MIN_RETURN_FLOOR = -0.02
METHOD_NAME = "cma_convergent_v1"


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def classify_asset_class(ticker: str | None, market: str | None = "TW", is_cash: bool = False) -> str:
    normalized = (ticker or "").strip().upper()
    if is_cash or not normalized or normalized in CASH_TICKERS:
        return "cash"
    if normalized in BOND_TICKERS:
        return "bond"
    return "equity"


def compute_anchor_return(asset_class: str, market: str | None = "TW") -> float:
    normalized_market = (market or "TW").strip().upper()
    if asset_class == "cash":
        return ANCHOR_RETURNS["cash"]
    if asset_class == "bond":
        return ANCHOR_RETURNS["bond"]
    if normalized_market == "US":
        return ANCHOR_RETURNS["us_equity"]
    return ANCHOR_RETURNS["tw_equity"]


def compute_historical_weight(horizon_years: int) -> float:
    if horizon_years < 1 or horizon_years > 30:
        raise ExpectedReturnInputError("horizonYears must be within 1~30.")
    raw = 0.55 * math.exp(-(horizon_years - 1) / 8.0)
    return _clamp(raw, 0.05, 0.55)


def _historical_bounds(asset_class: str) -> tuple[float, float]:
    if asset_class == "cash":
        return HISTORICAL_BOUNDS["cash"]
    if asset_class == "bond":
        return HISTORICAL_BOUNDS["bond"]
    return HISTORICAL_BOUNDS["equity"]


def _normalize_positions(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    total_weight = 0.0
    for item in positions:
        weight = max(0.0, float(item.get("weight", 0.0) or 0.0))
        total_weight += weight
        cleaned.append({**item, "weight": weight})

    if total_weight <= 0:
        raise ExpectedReturnInputError("positions weight sum must be greater than 0.")

    for item in cleaned:
        item["weight"] = item["weight"] / total_weight
    return cleaned


def estimate_expected_return(profile: dict[str, Any], positions: list[dict[str, Any]], coverage: float = 1.0) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ExpectedReturnInputError("profile is required.")
    if not isinstance(positions, list) or not positions:
        raise ExpectedReturnInputError("positions is required.")

    horizon_years = int(profile.get("horizonYears", 10) or 10)
    hist_weight = compute_historical_weight(horizon_years)
    safe_coverage = _clamp(float(coverage or 0.0), 0.0, 1.0)

    normalized_positions = _normalize_positions(positions)
    per_position: list[dict[str, Any]] = []

    anchor_return = 0.0
    historical_signal_return = 0.0
    gross_return = 0.0
    max_weight = 0.0

    for item in normalized_positions:
        ticker = str(item.get("ticker", "") or "")
        market = str(item.get("market", "TW") or "TW")
        position_weight = float(item["weight"])
        max_weight = max(max_weight, position_weight)

        asset_class = classify_asset_class(ticker=ticker, market=market, is_cash=bool(item.get("isCash", False)))
        anchor = compute_anchor_return(asset_class=asset_class, market=market)

        hist_raw = float(item.get("expectedReturn", 0.0) or 0.0)
        lo, hi = _historical_bounds(asset_class)
        hist_clamped = _clamp(hist_raw, lo, hi)

        blended = (1.0 - hist_weight) * anchor + hist_weight * hist_clamped

        anchor_return += position_weight * anchor
        historical_signal_return += position_weight * hist_clamped
        gross_return += position_weight * blended

        per_position.append(
            {
                "ticker": ticker,
                "market": market,
                "asset_class": asset_class,
                "weight": position_weight,
                "anchor_return": anchor,
                "historical_return": hist_clamped,
                "blended_return": blended,
            }
        )

    concentration_penalty = _clamp((max_weight - 0.35) * 0.04, 0.0, 0.012)
    coverage_penalty = (0.8 - safe_coverage) * 0.02 if safe_coverage < 0.8 else 0.0
    total_drag = FEE_DRAG + concentration_penalty + coverage_penalty

    expected_return = _clamp(gross_return - total_drag, MIN_RETURN_FLOOR, MAX_RETURN_CAP)

    return {
        "expected_return": expected_return,
        "components": {
            "anchor_return": anchor_return,
            "historical_signal_return": historical_signal_return,
            "historical_weight": hist_weight,
            "fee_drag": FEE_DRAG,
            "concentration_penalty": concentration_penalty,
            "coverage_penalty": coverage_penalty,
            "total_drag": total_drag,
        },
        "per_position": per_position,
        "method": METHOD_NAME,
    }


__all__ = [
    "ExpectedReturnInputError",
    "classify_asset_class",
    "compute_anchor_return",
    "compute_historical_weight",
    "estimate_expected_return",
]
