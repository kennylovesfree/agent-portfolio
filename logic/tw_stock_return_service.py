"""Service helpers for resolving Taiwan stocks and computing annual return."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

try:
    from .market_data_client import FinMindApiError, FinMindClient, FinMindUpstreamError, PricePoint
except ImportError:  # pragma: no cover - support direct script-style imports
    from market_data_client import FinMindApiError, FinMindClient, FinMindUpstreamError, PricePoint


class StockQueryError(RuntimeError):
    """Raised when user input does not pass strict validation rules."""

    def __init__(self, error_code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details


class UpstreamServiceError(RuntimeError):
    """Raised when FinMind upstream fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class ResolvedStock:
    stock_id: str
    stock_name: str


@dataclass(frozen=True)
class AnnualReturnResult:
    price_date_latest: str
    price_latest: float
    price_date_base: str
    price_base: float
    annual_return: float


def resolve_stock(query: str, client: FinMindClient) -> ResolvedStock:
    cleaned = query.strip()
    if not cleaned:
        raise StockQueryError("INVALID_QUERY", "query 不可為空白。")

    try:
        universe = client.get_taiwan_stock_info()
    except FinMindUpstreamError as exc:
        raise UpstreamServiceError(str(exc)) from exc
    except FinMindApiError as exc:
        raise UpstreamServiceError(str(exc)) from exc

    if cleaned.isdigit():
        if len(cleaned) != 4:
            raise StockQueryError("INVALID_QUERY", "台股代碼需為 4 位數字。", {"query": cleaned})
        for row in universe:
            if row["stock_id"] == cleaned:
                return ResolvedStock(stock_id=row["stock_id"], stock_name=row["stock_name"])
        raise StockQueryError("STOCK_NOT_FOUND", "查無此台股代碼。", {"query": cleaned})

    matches = [row for row in universe if row["stock_name"] == cleaned]
    dedup_by_stock_id: dict[str, dict[str, str]] = {m["stock_id"]: m for m in matches}
    unique_matches = list(dedup_by_stock_id.values())
    if not unique_matches:
        raise StockQueryError("STOCK_NOT_FOUND", "查無此台股名稱。", {"query": cleaned})
    if len(unique_matches) > 1:
        raise StockQueryError(
            "AMBIGUOUS_NAME",
            "名稱對應到多筆股票，請改用台股代碼。",
            {"query": cleaned, "matched_stock_ids": [m["stock_id"] for m in unique_matches]},
        )
    return ResolvedStock(stock_id=unique_matches[0]["stock_id"], stock_name=unique_matches[0]["stock_name"])


def compute_annual_return(stock_id: str, client: FinMindClient) -> AnnualReturnResult:
    today = date.today()
    start_date = (today - timedelta(days=400)).isoformat()
    end_date = today.isoformat()
    target_base_date = today - timedelta(days=365)

    try:
        history = client.get_price_history(
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date,
            dataset="TaiwanStockPrice",
        )
    except FinMindUpstreamError as exc:
        raise UpstreamServiceError(str(exc)) from exc
    except FinMindApiError as exc:
        raise UpstreamServiceError(str(exc)) from exc

    if not history:
        raise StockQueryError("NO_PRICE_DATA", "查無可用價格資料。", {"stock_id": stock_id})

    latest = history[-1]
    base = _find_base_price(history, target_base_date)
    if base is None:
        raise StockQueryError(
            "NO_PRICE_DATA",
            "找不到距今滿一年之前的基準收盤價。",
            {"stock_id": stock_id, "required_before": target_base_date.isoformat()},
        )
    if base.close <= 0:
        raise StockQueryError("NO_PRICE_DATA", "基準收盤價異常，無法計算報酬率。", {"stock_id": stock_id})

    annual_return = (latest.close / base.close) - 1.0
    return AnnualReturnResult(
        price_date_latest=latest.date,
        price_latest=latest.close,
        price_date_base=base.date,
        price_base=base.close,
        annual_return=annual_return,
    )


def _find_base_price(history: list[PricePoint], target_date: date) -> PricePoint | None:
    candidates: list[PricePoint] = []
    for point in history:
        try:
            point_date = date.fromisoformat(point.date)
        except ValueError:
            continue
        if point_date <= target_date:
            candidates.append(point)
    if not candidates:
        return None
    return candidates[-1]
