"""Daily crypto news aggregation, ranking, and summarization."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from pydantic import BaseModel, Field

from .crypto_news_store import CryptoNewsStore, CryptoNewsStoreConfigError, CryptoNewsStoreError

logger = logging.getLogger(__name__)


class CryptoNewsConfigError(RuntimeError):
    """Raised when crypto news configuration is missing."""


class CryptoNewsUpstreamError(RuntimeError):
    """Raised when upstream crypto news feeds fail."""


class CryptoNewsRateLimitError(RuntimeError):
    """Raised when upstream provider is rate-limited."""


class NewsItemSummary(BaseModel):
    summary: str
    risk_tags: list[str] = Field(default_factory=list)
    impact_level: str
    why_it_matters: str


SOURCE_WEIGHTS = {
    "cryptopanic": 1.0,
    "newsapi": 0.96,
    "thenewsapi": 0.95,
    "coindesk_rss": 0.92,
    "cointelegraph_rss": 0.88,
    "decrypt_rss": 0.86,
    "theblock_rss": 0.84,
}

RISK_KEYWORDS = {
    "regulation": [
        "sec",
        "etf",
        "regulation",
        "lawsuit",
        "ban",
        "compliance",
        "監管",
        "法規",
        "政策",
    ],
    "security": ["hack", "exploit", "breach", "stolen", "security", "漏洞", "被盜", "駭客"],
    "liquidity": ["liquidation", "funding", "outflow", "inflow", "liquidity", "清算", "流動性"],
    "market-structure": ["exchange", "derivatives", "open interest", "perp", "basis", "交易所", "衍生品"],
    "macro": ["fed", "rate", "cpi", "inflation", "usd", "yield", "聯準會", "利率", "通膨", "美元"],
}

DEFAULT_HTTP_HEADERS = {
    "User-Agent": "RebalanceLabsNewsBot/1.0 (+https://agent-portfolio.vercel.app)",
    "Accept": "application/json, application/xml, text/xml, text/plain;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class CryptoNewsConfig:
    default_limit: int
    cache_ttl_hours: int
    llm_max_items: int
    connect_timeout_seconds: float
    read_timeout_seconds: float
    http_retries: int
    cryptopanic_key: str
    newsapi_key: str
    thenewsapi_key: str
    ai_provider: str
    ai_model: str
    ai_key: str


@dataclass(frozen=True)
class FetchResult:
    items: list[dict[str, Any]]
    health_detail: dict[str, dict[str, str]]


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_dt(text: str | None) -> datetime:
    if not text:
        return _utcnow() - timedelta(days=3)
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        for fmt in (
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return _utcnow() - timedelta(days=3)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_optional_utc(text: str | None) -> Optional[datetime]:
    if not text:
        return None
    try:
        return _parse_dt(text)
    except Exception:
        return None


def _load_config() -> CryptoNewsConfig:
    default_limit = int(os.getenv("CRYPTO_NEWS_DEFAULT_LIMIT", "6"))
    cache_ttl_hours = int(os.getenv("CRYPTO_NEWS_CACHE_TTL_HOURS", "24"))
    llm_max_items = int(os.getenv("CRYPTO_NEWS_LLM_MAX_ITEMS", "3"))

    connect_timeout = float(os.getenv("CRYPTO_NEWS_HTTP_CONNECT_TIMEOUT_SEC", "3"))
    read_timeout = float(
        os.getenv(
            "CRYPTO_NEWS_HTTP_READ_TIMEOUT_SEC",
            os.getenv("CRYPTO_NEWS_HTTP_TIMEOUT_SEC", os.getenv("CRYPTO_NEWS_TIMEOUT_SEC", "8")),
        )
    )
    http_retries = int(os.getenv("CRYPTO_NEWS_HTTP_RETRIES", "3"))

    cp_key = os.getenv("CRYPTOPANIC_API_KEY", "").strip()
    newsapi_key = os.getenv("NEWSAPI_KEY", "").strip()
    thenewsapi_key = os.getenv("THENEWSAPI_KEY", "").strip()

    provider = os.getenv("AI_PROVIDER", "").strip().lower()
    if not provider:
        provider = "gemini" if os.getenv("GEMINI_API_KEY", "").strip() else "openai"

    if provider == "gemini":
        ai_key = os.getenv("GEMINI_API_KEY", "").strip()
        ai_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"
    else:
        provider = "openai"
        ai_key = os.getenv("OPENAI_API_KEY", "").strip()
        ai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

    return CryptoNewsConfig(
        default_limit=max(3, min(default_limit, 12)),
        cache_ttl_hours=max(1, cache_ttl_hours),
        llm_max_items=max(0, min(llm_max_items, 6)),
        connect_timeout_seconds=max(1.0, connect_timeout),
        read_timeout_seconds=max(1.0, read_timeout),
        http_retries=max(0, min(http_retries, 5)),
        cryptopanic_key=cp_key,
        newsapi_key=newsapi_key,
        thenewsapi_key=thenewsapi_key,
        ai_provider=provider,
        ai_model=ai_model,
        ai_key=ai_key,
    )


def _timeout_total(config: CryptoNewsConfig) -> float:
    return max(1.0, config.connect_timeout_seconds + config.read_timeout_seconds)


def _sleep_backoff(attempt: int) -> None:
    delay = min(2.0, 0.25 * (2**attempt)) + random.uniform(0, 0.2)
    time.sleep(delay)


def _error_code(exc: Exception) -> str:
    if isinstance(exc, CryptoNewsRateLimitError):
        return "RATE_LIMITED"
    text = str(exc)
    match = re.search(r"HTTP\s+(\d{3})", text)
    if match:
        return f"HTTP_{match.group(1)}"
    if "timed out" in text.lower():
        return "TIMEOUT"
    return "UPSTREAM_ERROR"


def _health(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _http_fetch(
    url: str,
    *,
    config: CryptoNewsConfig,
    headers: Optional[dict[str, str]] = None,
    expect_json: bool,
) -> Any:
    request_headers = dict(DEFAULT_HTTP_HEADERS)
    if headers:
        request_headers.update(headers)

    last_error: Optional[Exception] = None
    for attempt in range(config.http_retries + 1):
        request = Request(url, method="GET", headers=request_headers)
        try:
            with urlopen(request, timeout=_timeout_total(config)) as response:
                raw = response.read().decode("utf-8", errors="ignore")
            if expect_json:
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise CryptoNewsUpstreamError("News provider returned invalid JSON") from exc
            return raw
        except HTTPError as exc:
            if exc.code == 429:
                if attempt < config.http_retries:
                    _sleep_backoff(attempt)
                    last_error = exc
                    continue
                raise CryptoNewsRateLimitError("News provider rate limited") from exc

            if exc.code in RETRYABLE_HTTP_CODES and attempt < config.http_retries:
                _sleep_backoff(attempt)
                last_error = exc
                continue

            detail = exc.read().decode("utf-8", errors="ignore")
            raise CryptoNewsUpstreamError(f"News provider HTTP {exc.code}: {detail[:180]}") from exc
        except URLError as exc:
            if attempt < config.http_retries:
                _sleep_backoff(attempt)
                last_error = exc
                continue
            raise CryptoNewsUpstreamError(f"News provider unreachable: {exc}") from exc

    if isinstance(last_error, HTTPError):
        raise CryptoNewsUpstreamError(f"News provider HTTP {last_error.code}") from last_error
    raise CryptoNewsUpstreamError("News provider request failed") from last_error


def _newsapi_domains() -> str:
    return "coindesk.com,cointelegraph.com,decrypt.co,theblock.co"


def fetch_from_primary_apis(config: CryptoNewsConfig) -> FetchResult:
    items: list[dict[str, Any]] = []
    health_detail: dict[str, dict[str, str]] = {}

    if not config.cryptopanic_key:
        health_detail["cryptopanic"] = _health("disabled", "NO_KEY", "CRYPTOPANIC_API_KEY is missing")
    else:
        try:
            cp_params = urlencode(
                {
                    "auth_token": config.cryptopanic_key,
                    "kind": "news",
                    "currencies": "BTC",
                    "public": "true",
                    "regions": "en",
                }
            )
            cp_data = _http_fetch(
                f"https://cryptopanic.com/api/v1/posts/?{cp_params}",
                config=config,
                expect_json=True,
            )
            count = 0
            for item in cp_data.get("results", []):
                title = str(item.get("title", "")).strip()
                url = str(item.get("url", "")).strip()
                if not title or not url:
                    continue
                items.append(
                    {
                        "source": "cryptopanic",
                        "title": title,
                        "url": url,
                        "published_at": item.get("published_at"),
                        "heat": float(item.get("votes", {}).get("positive", 0) or 0),
                    }
                )
                count += 1
            if count == 0:
                health_detail["cryptopanic"] = _health("degraded", "EMPTY", "No results returned")
            else:
                health_detail["cryptopanic"] = _health("ok", "OK", f"Fetched {count} items")
        except Exception as exc:
            logger.exception("cryptopanic_fetch_failed")
            health_detail["cryptopanic"] = _health("down", _error_code(exc), str(exc)[:180])

    if not config.newsapi_key:
        health_detail["newsapi"] = _health("disabled", "NO_KEY", "NEWSAPI_KEY is missing")
    else:
        try:
            params = urlencode(
                {
                    "q": "bitcoin OR btc OR crypto",
                    "language": "en",
                    "sortBy": "publishedAt",
                    "domains": _newsapi_domains(),
                    "pageSize": "30",
                    "apiKey": config.newsapi_key,
                }
            )
            data = _http_fetch(
                f"https://newsapi.org/v2/everything?{params}",
                config=config,
                expect_json=True,
            )
            count = 0
            for article in data.get("articles", []):
                title = str(article.get("title", "")).strip()
                url = str(article.get("url", "")).strip()
                if not title or not url:
                    continue
                items.append(
                    {
                        "source": "newsapi",
                        "title": title,
                        "url": url,
                        "published_at": article.get("publishedAt"),
                        "heat": 0.0,
                    }
                )
                count += 1
            if count == 0:
                health_detail["newsapi"] = _health("degraded", "EMPTY", "No articles returned")
            else:
                health_detail["newsapi"] = _health("ok", "OK", f"Fetched {count} items")
        except Exception as exc:
            logger.exception("newsapi_fetch_failed")
            health_detail["newsapi"] = _health("down", _error_code(exc), str(exc)[:180])

    if not config.thenewsapi_key:
        health_detail["thenewsapi"] = _health("disabled", "NO_KEY", "THENEWSAPI_KEY is missing")
    else:
        try:
            params = urlencode(
                {
                    "api_token": config.thenewsapi_key,
                    "language": "en",
                    "search": "bitcoin OR btc OR crypto",
                    "domains": _newsapi_domains(),
                    "limit": "30",
                }
            )
            data = _http_fetch(
                f"https://api.thenewsapi.com/v1/news/all?{params}",
                config=config,
                expect_json=True,
            )
            count = 0
            for article in data.get("data", []):
                title = str(article.get("title", "")).strip()
                url = str(article.get("url", "")).strip()
                if not title or not url:
                    continue
                items.append(
                    {
                        "source": "thenewsapi",
                        "title": title,
                        "url": url,
                        "published_at": article.get("published_at"),
                        "heat": 0.0,
                    }
                )
                count += 1
            if count == 0:
                health_detail["thenewsapi"] = _health("degraded", "EMPTY", "No articles returned")
            else:
                health_detail["thenewsapi"] = _health("ok", "OK", f"Fetched {count} items")
        except Exception as exc:
            logger.exception("thenewsapi_fetch_failed")
            health_detail["thenewsapi"] = _health("down", _error_code(exc), str(exc)[:180])

    return FetchResult(items=items, health_detail=health_detail)


def _extract_atom_link(entry: ElementTree.Element) -> str:
    ns = "{http://www.w3.org/2005/Atom}"
    for link in entry.findall(f"{ns}link"):
        href = link.get("href")
        rel = (link.get("rel") or "alternate").lower()
        if href and rel in {"alternate", ""}:
            return href.strip()
    link = entry.findtext(f"{ns}link")
    return (link or "").strip()


def _parse_feed_items(xml_text: str, source: str) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(xml_text)
    parsed: list[dict[str, Any]] = []

    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        published = item.findtext("pubDate") or item.findtext("published")
        if title and url:
            parsed.append(
                {
                    "source": source,
                    "title": title,
                    "url": url,
                    "published_at": published,
                    "heat": 0.0,
                }
            )

    ns = "{http://www.w3.org/2005/Atom}"
    for entry in root.findall(f".//{ns}entry"):
        title = (entry.findtext(f"{ns}title") or "").strip()
        url = _extract_atom_link(entry)
        published = entry.findtext(f"{ns}updated") or entry.findtext(f"{ns}published")
        if title and url:
            parsed.append(
                {
                    "source": source,
                    "title": title,
                    "url": url,
                    "published_at": published,
                    "heat": 0.0,
                }
            )

    return parsed


def fetch_from_rss_feeds(config: CryptoNewsConfig) -> FetchResult:
    feeds = {
        "coindesk_rss": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "cointelegraph_rss": "https://cointelegraph.com/rss",
        "decrypt_rss": "https://decrypt.co/feed",
        "theblock_rss": "https://www.theblock.co/rss.xml",
    }

    items: list[dict[str, Any]] = []
    health_detail: dict[str, dict[str, str]] = {}

    for source, url in feeds.items():
        try:
            xml = _http_fetch(url, config=config, expect_json=False)
            parsed = _parse_feed_items(xml, source)
            items.extend(parsed)
            if parsed:
                health_detail[source] = _health("ok", "OK", f"Fetched {len(parsed)} items")
            else:
                health_detail[source] = _health("degraded", "EMPTY", "Feed is reachable but empty")
        except Exception as exc:
            logger.exception("rss_fetch_failed source=%s", source)
            health_detail[source] = _health("down", _error_code(exc), str(exc)[:180])

    return FetchResult(items=items, health_detail=health_detail)


def fetch_sources(config: CryptoNewsConfig) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, dict[str, str]]]:
    primary = fetch_from_primary_apis(config)
    rss = fetch_from_rss_feeds(config)

    source_health_detail = {**primary.health_detail, **rss.health_detail}
    source_health = {source: detail["status"] for source, detail in source_health_detail.items()}

    all_items = [*primary.items, *rss.items]
    if not all_items:
        raise CryptoNewsUpstreamError("All news sources failed.")

    return all_items, source_health, source_health_detail


def _canonicalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
    filtered_query = [
        (k, v)
        for k, v in query_pairs
        if not k.lower().startswith("utm_")
        and k.lower() not in {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source"}
    ]
    path = parsed.path or "/"
    clean = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path.rstrip("/") or "/",
            urlencode(filtered_query, doseq=True),
            "",
        )
    )
    return clean


def _title_fingerprint(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", title.lower())).strip()
    return cleaned or title.lower().strip()


def normalize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        if not title or not url:
            continue
        published_at = _parse_dt(str(item.get("published_at", "")))
        normalized.append(
            {
                "source": str(item.get("source", "unknown")),
                "title": title,
                "url": url,
                "canonical_url": _canonicalize_url(url),
                "published_at": published_at,
                "heat": float(item.get("heat", 0.0) or 0.0),
                "title_fingerprint": _title_fingerprint(title),
            }
        )
    return normalized


def dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped_by_url: dict[str, dict[str, Any]] = {}
    title_count: dict[str, int] = {}

    for item in items:
        title_key = item["title_fingerprint"]
        title_count[title_key] = title_count.get(title_key, 0) + 1

        url_key = item["canonical_url"]
        existing = deduped_by_url.get(url_key)
        if existing is None or item["published_at"] > existing["published_at"]:
            deduped_by_url[url_key] = item

    result = list(deduped_by_url.values())
    for item in result:
        item["duplicate_count"] = title_count.get(item["title_fingerprint"], 1)
    return result


def _extract_risk_tags(title: str) -> list[str]:
    text = title.lower()
    tags = [tag for tag, words in RISK_KEYWORDS.items() if any(word in text for word in words)]
    return tags[:2]


def score_importance(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = _utcnow()
    scored: list[dict[str, Any]] = []
    for item in items:
        title = item["title"]
        tags = _extract_risk_tags(title)
        keyword_score = min(50, len(tags) * 25)

        age_hours = max(0.0, (now - item["published_at"]).total_seconds() / 3600.0)
        if age_hours <= 6:
            recency_score = 20
        elif age_hours <= 24:
            recency_score = 16
        elif age_hours <= 48:
            recency_score = 12
        elif age_hours <= 72:
            recency_score = 8
        else:
            recency_score = 4

        source_weight = SOURCE_WEIGHTS.get(item["source"], 0.75)
        source_score = round(15 * source_weight)

        heat_raw = item.get("heat", 0.0) + (item.get("duplicate_count", 1) - 1) * 2
        heat_score = max(0, min(15, int(heat_raw)))

        importance = max(0, min(100, keyword_score + recency_score + source_score + heat_score))
        scored.append(
            {
                **item,
                "risk_tags": tags,
                "importance_score": importance,
                "impact_level": "high" if importance >= 75 else "medium" if importance >= 50 else "low",
            }
        )

    scored.sort(key=lambda x: (x["importance_score"], x["published_at"]), reverse=True)
    return scored


def merge_and_rank_items(primary_items: list[dict[str, Any]], rss_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_items([*primary_items, *rss_items])
    deduped = dedupe_items(normalized)

    now = _utcnow()
    in_24h = [item for item in deduped if (now - item["published_at"]) <= timedelta(hours=24)]
    if len(in_24h) >= 4:
        candidate_items = in_24h
    else:
        in_72h = [item for item in deduped if (now - item["published_at"]) <= timedelta(hours=72)]
        candidate_items = in_72h if in_72h else deduped

    return score_importance(candidate_items)


def _extract_json_payload(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise CryptoNewsUpstreamError("LLM response is not JSON")
        return json.loads(stripped[start : end + 1])


def summarize_with_llm(item: dict[str, Any], lang: str, config: CryptoNewsConfig) -> NewsItemSummary:
    if not config.ai_key:
        raise CryptoNewsConfigError("Missing LLM API key for crypto news summarization.")

    prompt = {
        "lang": lang,
        "title": item["title"],
        "source": item["source"],
        "risk_tags": item.get("risk_tags", []),
        "importance_score": item["importance_score"],
        "schema": {
            "summary": "2-3 sentences",
            "risk_tags": ["regulation|security|liquidity|market-structure|macro"],
            "impact_level": "high|medium|low",
            "why_it_matters": "1 sentence",
        },
    }

    if config.ai_provider == "gemini":
        body = {
            "systemInstruction": {"parts": [{"text": "Return JSON only. No markdown."}]},
            "contents": [{"parts": [{"text": json.dumps(prompt, ensure_ascii=False)}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        request = Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{config.ai_model}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": config.ai_key,
            },
        )
        try:
            with urlopen(request, timeout=_timeout_total(config)) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 429:
                raise CryptoNewsRateLimitError("Gemini rate limit reached") from exc
            raise CryptoNewsUpstreamError(f"Gemini HTTP error: {exc.code}") from exc
        except URLError as exc:
            raise CryptoNewsUpstreamError(f"Gemini unreachable: {exc}") from exc

        parts = parsed.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
        payload = _extract_json_payload(text)
        return NewsItemSummary.model_validate(payload)

    body = {
        "model": config.ai_model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a crypto risk analyst. Return strict JSON only with keys: "
                    "summary, risk_tags, impact_level, why_it_matters."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }
    request = Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {config.ai_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=_timeout_total(config)) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 429:
            raise CryptoNewsRateLimitError("OpenAI rate limit reached") from exc
        raise CryptoNewsUpstreamError(f"OpenAI HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise CryptoNewsUpstreamError(f"OpenAI unreachable: {exc}") from exc

    content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
    payload = _extract_json_payload(content)
    return NewsItemSummary.model_validate(payload)


def _fallback_summary(item: dict[str, Any], lang: str) -> NewsItemSummary:
    tags = item.get("risk_tags", [])[:2]
    impact = item["impact_level"]
    if lang == "en":
        return NewsItemSummary(
            summary=f"{item['title']} may influence short-term BTC risk sentiment.",
            risk_tags=tags,
            impact_level=impact,
            why_it_matters="Monitor position sizing and leverage if this theme continues to develop.",
        )
    return NewsItemSummary(
        summary=f"{item['title']} 可能影響短期 BTC 風險情緒。",
        risk_tags=tags,
        impact_level=impact,
        why_it_matters="若此議題持續發酵，建議留意倉位與槓桿管理。",
    )


def _build_overview(items: list[dict[str, Any]], lang: str) -> list[str]:
    lines: list[str] = []
    for item in items[:5]:
        if lang == "en":
            lines.append(f"{item['impact_level'].upper()} | {item['title']}")
        else:
            level_map = {"high": "高", "medium": "中", "low": "低"}
            lines.append(f"{level_map.get(item['impact_level'], '中')}影響 | {item['title']}")
    return lines


def build_digest(lang: str, limit: int, config: CryptoNewsConfig) -> dict[str, Any]:
    primary = fetch_from_primary_apis(config)
    rss = fetch_from_rss_feeds(config)

    all_items = [*primary.items, *rss.items]
    source_health_detail = {**primary.health_detail, **rss.health_detail}
    source_health = {source: detail["status"] for source, detail in source_health_detail.items()}

    if not all_items:
        raise CryptoNewsUpstreamError("All news sources failed.")

    scored = merge_and_rank_items(primary.items, rss.items)
    if not scored:
        raise CryptoNewsUpstreamError("No valid news items after normalization and ranking.")

    selected = scored[: max(3, limit)]
    items: list[dict[str, Any]] = []
    for idx, raw in enumerate(selected):
        summary = _fallback_summary(raw, lang)
        if idx < config.llm_max_items:
            try:
                summary = summarize_with_llm(raw, lang, config)
            except CryptoNewsConfigError:
                summary = _fallback_summary(raw, lang)
            except (CryptoNewsRateLimitError, CryptoNewsUpstreamError):
                summary = _fallback_summary(raw, lang)
            except Exception:  # pragma: no cover - guard provider/schema drift
                logger.exception("crypto_news_summarize_failed source=%s", raw.get("source"))
                summary = _fallback_summary(raw, lang)

        item_id = hashlib.sha256(f"{raw['title']}|{raw['canonical_url']}".encode("utf-8")).hexdigest()[:16]
        items.append(
            {
                "id": item_id,
                "title": raw["title"],
                "source": raw["source"],
                "published_at": raw["published_at"].isoformat().replace("+00:00", "Z"),
                "url": raw["url"],
                "importance_score": raw["importance_score"],
                "impact_level": summary.impact_level,
                "risk_tags": summary.risk_tags[:2],
                "summary": summary.summary,
                "why_it_matters": summary.why_it_matters,
            }
        )

    return {
        "as_of": _utcnow().isoformat().replace("+00:00", "Z"),
        "lang": lang,
        "daily_overview": _build_overview(items, lang),
        "items": items,
        "source_health": source_health,
        "source_health_detail": source_health_detail,
    }


def _cache_age_hours(row: dict[str, Any]) -> Optional[float]:
    updated_at = _parse_optional_utc(str(row.get("updated_at", "")))
    if updated_at is None:
        payload = row.get("payload")
        if isinstance(payload, dict):
            updated_at = _parse_optional_utc(str(payload.get("as_of", "")))
    if updated_at is None:
        return None
    return max(0.0, (_utcnow() - updated_at).total_seconds() / 3600.0)


def _augment_payload(
    payload: dict[str, Any],
    *,
    cache_hit: bool,
    fallback_used: bool,
    stale_age_hours: Optional[float],
) -> dict[str, Any]:
    result = dict(payload)
    result["cache_hit"] = cache_hit
    result["fallback_used"] = fallback_used
    result["stale_age_hours"] = round(stale_age_hours, 2) if stale_age_hours is not None else None
    return result


def get_news_digest(lang: str = "zh", limit: Optional[int] = None, force_refresh: bool = False) -> dict[str, Any]:
    config = _load_config()
    store = CryptoNewsStore.from_env()

    safe_lang = "en" if lang == "en" else "zh"
    safe_limit = max(3, min(limit or config.default_limit, 12))
    today = _utcnow().date()

    if not force_refresh:
        cached = store.get_digest(today, safe_lang)
        if cached and isinstance(cached.get("payload"), dict):
            age = _cache_age_hours(cached)
            if age is not None and age <= config.cache_ttl_hours:
                return _augment_payload(
                    cached["payload"],
                    cache_hit=True,
                    fallback_used=False,
                    stale_age_hours=age,
                )

        latest = store.get_latest_success(safe_lang, max_age_hours=config.cache_ttl_hours)
        if latest and isinstance(latest.get("payload"), dict):
            return _augment_payload(
                latest["payload"],
                cache_hit=True,
                fallback_used=False,
                stale_age_hours=_cache_age_hours(latest),
            )

    try:
        digest = build_digest(safe_lang, safe_limit, config)
    except (CryptoNewsUpstreamError, CryptoNewsRateLimitError, CryptoNewsConfigError):
        stale = store.get_latest_success(safe_lang, max_age_hours=48)
        if stale and isinstance(stale.get("payload"), dict):
            return _augment_payload(
                stale["payload"],
                cache_hit=True,
                fallback_used=True,
                stale_age_hours=_cache_age_hours(stale),
            )
        raise

    store.upsert_digest(today, safe_lang, digest)
    return _augment_payload(digest, cache_hit=False, fallback_used=False, stale_age_hours=0.0)


def refresh_news_digest(lang: str = "both") -> dict[str, Any]:
    targets = ["zh", "en"] if lang == "both" else ["en" if lang == "en" else "zh"]
    results = {}
    for target in targets:
        results[target] = get_news_digest(lang=target, force_refresh=True)
    return {"status": "ok", "results": results}


__all__ = [
    "CryptoNewsConfigError",
    "CryptoNewsRateLimitError",
    "CryptoNewsUpstreamError",
    "build_digest",
    "dedupe_items",
    "fetch_from_primary_apis",
    "fetch_from_rss_feeds",
    "fetch_sources",
    "get_news_digest",
    "merge_and_rank_items",
    "normalize_items",
    "refresh_news_digest",
    "score_importance",
]
