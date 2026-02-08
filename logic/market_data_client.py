"""FinMind API client for pulling market data used by the demo ingestion flow."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class FinMindApiError(RuntimeError):
    """Raised when FinMind API returns an error payload or malformed response."""


@dataclass(frozen=True)
class PricePoint:
    data_id: str
    date: str
    close: float


class FinMindClient:
    """Minimal HTTP client for FinMind v4 dataset endpoint."""

    def __init__(self, token: str, base_url: str = "https://api.finmindtrade.com/api/v4/data") -> None:
        self.token = token
        self.base_url = base_url

    @classmethod
    def from_env(cls) -> "FinMindClient | None":
        token = os.getenv("FINMIND_API_TOKEN", "").strip()
        if not token:
            return None
        return cls(token=token)

    def get_latest_close(
        self,
        *,
        data_id: str,
        dataset: str = "TaiwanStockPrice",
        lookback_days: int = 30,
    ) -> PricePoint:
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        payload = self._get(
            dataset=dataset,
            data_id=data_id,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        records = payload.get("data", [])
        if not records:
            raise FinMindApiError(f"No price data found for data_id={data_id}, dataset={dataset}.")

        latest = sorted(records, key=lambda row: row.get("date", ""))[-1]
        close = latest.get("close")
        if close is None:
            raise FinMindApiError(f"Latest record missing close price for data_id={data_id}.")

        return PricePoint(data_id=data_id, date=str(latest.get("date", "")), close=float(close))

    def _get(self, **query: str) -> dict[str, Any]:
        url = f"{self.base_url}?{urlencode(query)}"
        request = Request(url, headers={"Authorization": f"Bearer {self.token}"})
        with urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")

        parsed: dict[str, Any] = json.loads(body)
        if parsed.get("msg") != "success":
            raise FinMindApiError(f"FinMind API error: {parsed}")
        return parsed
