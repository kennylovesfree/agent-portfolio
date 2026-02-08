"""Data ingestion helpers that support mock JSON and FinMind-backed portfolio valuation."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from market_data_client import FinMindApiError, FinMindClient


class DataIngestionError(RuntimeError):
    """Raised when ingestion config is invalid."""


def load_portfolio_data() -> dict[str, Any]:
    """Load portfolio from FinMind config when enabled, otherwise fallback to local mock file."""
    if _env_is_true("USE_FINMIND_API"):
        return _load_from_finmind()
    return _load_local_mock()


def _load_local_mock() -> dict[str, Any]:
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_portfolio.json"
    return json.loads(data_path.read_text(encoding="utf-8"))


def _load_from_finmind() -> dict[str, Any]:
    config_path = os.getenv("FINMIND_PORTFOLIO_PATH", "").strip()
    if not config_path:
        raise DataIngestionError("USE_FINMIND_API=1 時，需設定 FINMIND_PORTFOLIO_PATH。")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    positions = cfg.get("positions", [])
    target_allocation = cfg.get("target_allocation", [])
    constraints = cfg.get("constraints", {})
    if not positions:
        raise DataIngestionError("FINMIND portfolio config must include non-empty positions.")

    client = FinMindClient.from_env()
    if client is None:
        raise DataIngestionError("USE_FINMIND_API=1 時，需設定 FINMIND_API_TOKEN。")

    holdings = _build_holdings_from_positions(client, positions)
    return {
        "as_of": cfg.get("as_of", ""),
        "holdings": holdings,
        "target_allocation": target_allocation,
        "constraints": constraints,
        "source": "finmind_api",
    }


def _build_holdings_from_positions(client: FinMindClient, positions: list[dict[str, Any]]) -> list[dict[str, float]]:
    valuations: list[dict[str, Any]] = []
    for p in positions:
        data_id = str(p.get("data_id", "")).strip()
        asset = str(p.get("asset", data_id)).strip()
        shares = float(p.get("shares", 0))
        dataset = str(p.get("dataset", "TaiwanStockPrice"))
        if not data_id or shares <= 0:
            raise DataIngestionError(f"Invalid position entry: {p}")

        try:
            latest = client.get_latest_close(data_id=data_id, dataset=dataset)
        except FinMindApiError as exc:
            raise DataIngestionError(str(exc)) from exc

        valuations.append(
            {
                "asset": asset,
                "market_value": shares * latest.close,
                "data_id": data_id,
                "pricing_date": latest.date,
            }
        )

    total_value = sum(v["market_value"] for v in valuations)
    if total_value <= 0:
        raise DataIngestionError("Total market value must be positive.")

    return [{"asset": v["asset"], "weight": v["market_value"] / total_value} for v in valuations]


def _env_is_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
