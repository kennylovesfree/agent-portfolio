"""HTTP API for annual return, expected return, risk checks, and AI advice."""
from __future__ import annotations

import logging
from dataclasses import asdict
import os
from typing import Optional

from fastapi import FastAPI, Header, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from .market_data_client import FinMindClient
    from .advice_service import (
        AdviceConfigError,
        AdviceRateLimitError,
        AdviceRequest,
        AdviceUpstreamError,
        generate_advice,
    )
    from .portfolio_health_service import (
        PortfolioInputError as HealthPortfolioInputError,
        evaluate_portfolio_health,
    )
    from .expected_return_service import (
        ExpectedReturnInputError,
        estimate_expected_return,
    )
    from .stress_test_service import (
        PortfolioInputError as StressPortfolioInputError,
        run_stress_test,
    )
    from .tw_stock_return_service import (
        StockQueryError as TwStockQueryError,
        UpstreamServiceError as TwUpstreamServiceError,
        compute_annual_return as compute_tw_annual_return,
        resolve_stock,
    )
    from .us_stock_return_service import (
        StockQueryError as UsStockQueryError,
        UpstreamServiceError as UsUpstreamServiceError,
        compute_annual_return as compute_us_annual_return,
    )
    from .crypto_news_service import (
        CryptoNewsConfigError,
        CryptoNewsRateLimitError,
        CryptoNewsUpstreamError,
        get_news_digest,
        refresh_news_digest,
    )
    from .crypto_news_store import CryptoNewsStoreConfigError, CryptoNewsStoreError
except ImportError:  # pragma: no cover - support direct script-style imports
    from market_data_client import FinMindClient
    from advice_service import (
        AdviceConfigError,
        AdviceRateLimitError,
        AdviceRequest,
        AdviceUpstreamError,
        generate_advice,
    )
    from portfolio_health_service import (
        PortfolioInputError as HealthPortfolioInputError,
        evaluate_portfolio_health,
    )
    from expected_return_service import (
        ExpectedReturnInputError,
        estimate_expected_return,
    )
    from stress_test_service import (
        PortfolioInputError as StressPortfolioInputError,
        run_stress_test,
    )
    from tw_stock_return_service import (
        StockQueryError as TwStockQueryError,
        UpstreamServiceError as TwUpstreamServiceError,
        compute_annual_return as compute_tw_annual_return,
        resolve_stock,
    )
    from us_stock_return_service import (
        StockQueryError as UsStockQueryError,
        UpstreamServiceError as UsUpstreamServiceError,
        compute_annual_return as compute_us_annual_return,
    )
    from crypto_news_service import (
        CryptoNewsConfigError,
        CryptoNewsRateLimitError,
        CryptoNewsUpstreamError,
        get_news_digest,
        refresh_news_digest,
    )
    from crypto_news_store import CryptoNewsStoreConfigError, CryptoNewsStoreError


class AnnualReturnRequest(BaseModel):
    query: str


class PortfolioProfile(BaseModel):
    age: int = Field(ge=18, le=90)
    riskLevel: str
    taxRegion: str
    horizonYears: int = Field(ge=1, le=30)


class PortfolioPosition(BaseModel):
    ticker: str = ""
    market: str = "TW"
    amountUsd: float = Field(ge=0)
    expectedReturn: float
    volatility: float = Field(ge=0)
    weight: float = Field(ge=0, le=1)


class PortfolioMetrics(BaseModel):
    expectedReturn: float
    volatility: float = Field(ge=0)
    maxDrawdown: float = Field(ge=0)


class PortfolioHealthCheckRequest(BaseModel):
    profile: PortfolioProfile
    positions: list[PortfolioPosition]
    portfolio: PortfolioMetrics


class ExpectedReturnPosition(BaseModel):
    ticker: str = ""
    market: str = "TW"
    weight: float = Field(ge=0, le=1)
    expectedReturn: float
    isCash: bool = False


class PortfolioExpectedReturnRequest(BaseModel):
    profile: PortfolioProfile
    positions: list[ExpectedReturnPosition]
    coverage: float = Field(default=1.0, ge=0, le=1)


class StressScenarioRequest(BaseModel):
    scenario_id: str


class PortfolioStressTestRequest(BaseModel):
    positions: list[PortfolioPosition]
    scenarios: Optional[list[StressScenarioRequest]] = None


class CryptoNewsRefreshRequest(BaseModel):
    lang: str = "both"


class ApiError(RuntimeError):
    def __init__(self, *, status_code: int, error_code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.details = details


logger = logging.getLogger(__name__)
app = FastAPI(title="Stock Annual Return API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ApiError)
async def api_error_handler(_, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "INVALID_QUERY",
            "message": "請提供有效的 query 字串。",
            "details": {"errors": exc.errors()},
        },
    )


@app.post("/api/v1/tw-stock/annual-return")
def get_annual_return(payload: AnnualReturnRequest) -> dict:
    client = FinMindClient.from_env()
    if client is None:
        raise ApiError(
            status_code=500,
            error_code="CONFIG_ERROR",
            message="缺少 FINMIND_API_TOKEN，無法查詢資料。",
            details=None,
        )

    try:
        resolved = resolve_stock(payload.query, client)
        result = compute_tw_annual_return(resolved.stock_id, client)
    except TwStockQueryError as exc:
        raise ApiError(
            status_code=422,
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
        ) from exc
    except TwUpstreamServiceError as exc:
        raise ApiError(
            status_code=502,
            error_code="UPSTREAM_ERROR",
            message=exc.message,
            details=None,
        ) from exc

    logger.info("stock_resolve_success stock_id=%s query=%s", resolved.stock_id, payload.query)

    return {
        "query": payload.query,
        "resolved_stock_id": resolved.stock_id,
        "resolved_stock_name": resolved.stock_name,
        "price_date_latest": result.price_date_latest,
        "price_latest": result.price_latest,
        "price_date_base": result.price_date_base,
        "price_base": result.price_base,
        "annual_return": result.annual_return,
    }


@app.post("/api/v1/us-stock/annual-return")
def get_us_annual_return(payload: AnnualReturnRequest) -> dict:
    try:
        resolved_ticker, result = compute_us_annual_return(payload.query)
    except UsStockQueryError as exc:
        raise ApiError(
            status_code=422,
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
        ) from exc
    except UsUpstreamServiceError as exc:
        raise ApiError(
            status_code=502,
            error_code="UPSTREAM_ERROR",
            message=exc.message,
            details=None,
        ) from exc

    logger.info("us_stock_resolve_success ticker=%s query=%s", resolved_ticker, payload.query)

    return {
        "query": payload.query,
        "resolved_stock_id": resolved_ticker,
        "resolved_stock_name": resolved_ticker,
        "price_date_latest": result.price_date_latest,
        "price_latest": result.price_latest,
        "price_date_base": result.price_date_base,
        "price_base": result.price_base,
        "annual_return": result.annual_return,
    }


@app.post("/api/v1/portfolio/health-check")
def post_portfolio_health_check(payload: PortfolioHealthCheckRequest) -> dict:
    try:
        result = evaluate_portfolio_health(
            positions=[position.model_dump() for position in payload.positions],
            portfolio=payload.portfolio.model_dump(),
        )
    except HealthPortfolioInputError as exc:
        raise ApiError(
            status_code=422,
            error_code="INVALID_PORTFOLIO_INPUT",
            message=str(exc),
            details=None,
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive error mapping
        logger.exception("portfolio_health_check_failed")
        raise ApiError(
            status_code=500,
            error_code="RISK_ENGINE_ERROR",
            message="健康度引擎暫時無法使用。",
            details=None,
        ) from exc

    return {
        "health_score": result.health_score,
        "risk_band": result.risk_band,
        "components": asdict(result.components),
        "flags": result.flags,
        "explanations": result.explanations,
    }


@app.post("/api/v1/portfolio/expected-return")
def post_portfolio_expected_return(payload: PortfolioExpectedReturnRequest) -> dict:
    try:
        result = estimate_expected_return(
            profile=payload.profile.model_dump(),
            positions=[position.model_dump() for position in payload.positions],
            coverage=payload.coverage,
        )
    except ExpectedReturnInputError as exc:
        raise ApiError(
            status_code=422,
            error_code="EXPECTED_RETURN_INVALID_INPUT",
            message=str(exc),
            details=None,
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive error mapping
        logger.exception("portfolio_expected_return_failed")
        raise ApiError(
            status_code=500,
            error_code="EXPECTED_RETURN_ENGINE_ERROR",
            message="長期收益估算引擎暫時無法使用。",
            details=None,
        ) from exc

    return result


@app.post("/api/v1/portfolio/stress-test")
def post_portfolio_stress_test(payload: PortfolioStressTestRequest) -> dict:
    try:
        result = run_stress_test(
            positions=[position.model_dump() for position in payload.positions],
            scenarios=[scenario.model_dump() for scenario in payload.scenarios] if payload.scenarios else None,
        )
    except StressPortfolioInputError as exc:
        raise ApiError(
            status_code=422,
            error_code="INVALID_PORTFOLIO_INPUT",
            message=str(exc),
            details=None,
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive error mapping
        logger.exception("portfolio_stress_test_failed")
        raise ApiError(
            status_code=500,
            error_code="RISK_ENGINE_ERROR",
            message="壓力測試引擎暫時無法使用。",
            details=None,
        ) from exc

    return {
        "scenario_results": [asdict(item) for item in result.scenario_results],
        "worst_case": asdict(result.worst_case),
        "survival_days_est": result.survival_days_est,
    }


@app.post("/api/v1/advice/generate")
def post_generate_advice(payload: AdviceRequest) -> dict:
    try:
        advice = generate_advice(payload)
    except AdviceConfigError as exc:
        raise ApiError(
            status_code=503,
            error_code="AI_CONFIG_ERROR",
            message=str(exc),
            details=None,
        ) from exc
    except AdviceRateLimitError as exc:
        raise ApiError(
            status_code=429,
            error_code="AI_RATE_LIMITED",
            message=str(exc),
            details=None,
        ) from exc
    except AdviceUpstreamError as exc:
        raise ApiError(
            status_code=502,
            error_code="AI_UPSTREAM_ERROR",
            message=str(exc),
            details=None,
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive mapping for provider/schema drift
        logger.exception("advice_generate_unhandled_error")
        raise ApiError(
            status_code=502,
            error_code="AI_UPSTREAM_ERROR",
            message="AI provider response is not in the expected format.",
            details=None,
        ) from exc

    logger.info(
        "advice_success risk=%s model=%s latency_ms=%s",
        advice.risk_level,
        advice.model_meta.model,
        advice.model_meta.latency_ms,
    )
    return advice.model_dump()


@app.get("/api/v1/crypto/news-digest")
def get_crypto_news_digest(
    lang: str = Query(default="zh"),
    limit: int = Query(default=6, ge=3, le=12),
    force_refresh: int = Query(default=0),
) -> dict:
    try:
        return get_news_digest(lang=lang, limit=limit, force_refresh=force_refresh == 1)
    except (CryptoNewsConfigError, CryptoNewsStoreConfigError) as exc:
        raise ApiError(
            status_code=500,
            error_code="CRYPTO_NEWS_CONFIG_ERROR",
            message=str(exc),
            details=None,
        ) from exc
    except CryptoNewsRateLimitError as exc:
        raise ApiError(
            status_code=429,
            error_code="CRYPTO_NEWS_RATE_LIMITED",
            message=str(exc),
            details=None,
        ) from exc
    except (CryptoNewsUpstreamError, CryptoNewsStoreError) as exc:
        raise ApiError(
            status_code=502,
            error_code="CRYPTO_NEWS_UPSTREAM_ERROR",
            message=str(exc),
            details=None,
        ) from exc


@app.post("/api/v1/crypto/news-digest/refresh")
def post_crypto_news_digest_refresh(
    payload: CryptoNewsRefreshRequest,
    authorization: Optional[str] = Header(default=None),
) -> dict:
    expected = os.getenv("CRYPTO_NEWS_REFRESH_TOKEN", "").strip()
    token = (authorization or "").replace("Bearer ", "", 1).strip()
    if expected and token != expected:
        raise ApiError(
            status_code=401,
            error_code="CRYPTO_NEWS_UNAUTHORIZED",
            message="Invalid refresh token.",
            details=None,
        )

    lang = payload.lang if payload.lang in {"zh", "en", "both"} else "both"
    try:
        return refresh_news_digest(lang=lang)
    except (CryptoNewsConfigError, CryptoNewsStoreConfigError) as exc:
        raise ApiError(
            status_code=500,
            error_code="CRYPTO_NEWS_CONFIG_ERROR",
            message=str(exc),
            details=None,
        ) from exc
    except CryptoNewsRateLimitError as exc:
        raise ApiError(
            status_code=429,
            error_code="CRYPTO_NEWS_RATE_LIMITED",
            message=str(exc),
            details=None,
        ) from exc
    except (CryptoNewsUpstreamError, CryptoNewsStoreError) as exc:
        raise ApiError(
            status_code=502,
            error_code="CRYPTO_NEWS_UPSTREAM_ERROR",
            message=str(exc),
            details=None,
        ) from exc
