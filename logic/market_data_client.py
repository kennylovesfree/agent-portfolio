"""FinMind API client for pulling market data used by the demo ingestion flow."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class FinMindApiError(RuntimeError):
    """Raised when FinMind API returns an error payload or malformed response."""


class FinMindUpstreamError(RuntimeError):
    """Raised when FinMind API is unavailable or response parsing fails."""


@dataclass(frozen=True)
class PricePoint:
    data_id: str
    date: str
    close: float


@dataclass(frozen=True)
class SplitEvent:
    data_id: str
    date: str
    before_price: float
    after_price: float


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
        records = self.get_price_history(
            data_id=data_id,
            dataset=dataset,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        if not records:
            raise FinMindApiError(f"No price data found for data_id={data_id}, dataset={dataset}.")

        return records[-1]

    def get_taiwan_stock_info(self, dataset: str = "TaiwanStockInfo") -> list[dict[str, str]]:
        payload = self._get(dataset=dataset)
        records = payload.get("data", [])
        parsed: list[dict[str, str]] = []
        for row in records:
            stock_id = str(row.get("stock_id", "")).strip()
            stock_name = str(row.get("stock_name", "")).strip()
            if not stock_id or not stock_name:
                continue
            parsed.append({"stock_id": stock_id, "stock_name": stock_name})
        return parsed

    def get_price_history(
        self,
        *,
        data_id: str,
        start_date: str,
        end_date: str,
        dataset: str = "TaiwanStockPrice",
    ) -> list[PricePoint]:
        payload = self._get(
            dataset=dataset,
            data_id=data_id,
            start_date=start_date,
            end_date=end_date,
        )
        records = payload.get("data", [])
        parsed: list[PricePoint] = []
        for row in records:
            close = row.get("close")
            if close is None:
                continue
            parsed.append(
                PricePoint(
                    data_id=data_id,
                    date=str(row.get("date", "")),
                    close=float(close),
                )
            )
        return sorted(parsed, key=lambda r: r.date)

    def get_split_events(
        self,
        *,
        data_id: str,
        start_date: str,
        end_date: str,
        dataset: str = "TaiwanStockSplitPrice",
    ) -> list[SplitEvent]:
        payload = self._get(
            dataset=dataset,
            data_id=data_id,
            start_date=start_date,
            end_date=end_date,
        )
        records = payload.get("data", [])
        parsed: list[SplitEvent] = []
        for row in records:
            before_price = row.get("before_price")
            after_price = row.get("after_price")
            if before_price is None or after_price is None:
                continue
            parsed.append(
                SplitEvent(
                    data_id=data_id,
                    date=str(row.get("date", "")),
                    before_price=float(before_price),
                    after_price=float(after_price),
                )
            )
        return sorted(parsed, key=lambda r: r.date)

    def _get(self, **query: str) -> dict[str, Any]:
        url = f"{self.base_url}?{urlencode(query)}"
        request = Request(url, headers={"Authorization": f"Bearer {self.token}"})
        try:
            with urlopen(request, timeout=20) as response:
                body = response.read().decode("utf-8")
        except URLError as exc:
            raise FinMindUpstreamError(f"Failed to reach FinMind API: {exc}") from exc

        try:
            parsed: dict[str, Any] = json.loads(body)
        except json.JSONDecodeError as exc:
            raise FinMindUpstreamError("FinMind API returned malformed JSON response.") from exc

        if parsed.get("msg") != "success":
            raise FinMindApiError(f"FinMind API error: {parsed}")
        return parsed
