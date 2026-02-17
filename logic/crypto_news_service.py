"""Daily crypto news aggregation, ranking, and summarization."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
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
    "coindesk_rss": 0.92,
    "cointelegraph_rss": 0.88,
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


@dataclass(frozen=True)
class CryptoNewsConfig:
    default_limit: int
    cache_ttl_hours: int
    timeout_seconds: float
    cryptopanic_key: str
    ai_provider: str
    ai_model: str
    ai_key: str


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
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%d %H:%M:%S"):
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


def _load_config() -> CryptoNewsConfig:
    default_limit = int(os.getenv("CRYPTO_NEWS_DEFAULT_LIMIT", "6"))
    cache_ttl_hours = int(os.getenv("CRYPTO_NEWS_CACHE_TTL_HOURS", "24"))
    timeout_seconds = float(os.getenv("CRYPTO_NEWS_TIMEOUT_SEC", "8"))
    cp_key = os.getenv("CRYPTOPANIC_API_KEY", "").strip()

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
        timeout_seconds=max(1.0, timeout_seconds),
        cryptopanic_key=cp_key,
        ai_provider=provider,
        ai_model=ai_model,
        ai_key=ai_key,
    )


def _http_json(url: str, timeout: float, headers: Optional[dict[str, str]] = None) -> Any:
    request = Request(url, method="GET", headers=headers or {})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        if exc.code == 429:
            raise CryptoNewsRateLimitError("News provider rate limited") from exc
        raise CryptoNewsUpstreamError(f"News provider HTTP {exc.code}") from exc
    except URLError as exc:
        raise CryptoNewsUpstreamError(f"News provider unreachable: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CryptoNewsUpstreamError("News provider returned invalid JSON") from exc


def _http_text(url: str, timeout: float) -> str:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        if exc.code == 429:
            raise CryptoNewsRateLimitError("News provider rate limited") from exc
        raise CryptoNewsUpstreamError(f"News provider HTTP {exc.code}") from exc
    except URLError as exc:
        raise CryptoNewsUpstreamError(f"News provider unreachable: {exc}") from exc


def fetch_sources(config: CryptoNewsConfig) -> tuple[list[dict[str, Any]], dict[str, str]]:
    all_items: list[dict[str, Any]] = []
    source_health = {
        "cryptopanic": "disabled" if not config.cryptopanic_key else "ok",
        "coindesk_rss": "ok",
        "cointelegraph_rss": "ok",
    }

    if config.cryptopanic_key:
        try:
            cp_params = urlencode({"auth_token": config.cryptopanic_key, "kind": "news", "currencies": "BTC"})
            cp_data = _http_json(f"https://cryptopanic.com/api/v1/posts/?{cp_params}", config.timeout_seconds)
            for item in cp_data.get("results", []):
                all_items.append(
                    {
                        "source": "cryptopanic",
                        "title": str(item.get("title", "")).strip(),
                        "url": str(item.get("url", "")).strip(),
                        "published_at": item.get("published_at"),
                        "heat": float(item.get("votes", {}).get("positive", 0) or 0),
                    }
                )
        except Exception:
            logger.exception("cryptopanic_fetch_failed")
            source_health["cryptopanic"] = "degraded"

    try:
        xml = _http_text("https://www.coindesk.com/arc/outboundfeeds/rss/", config.timeout_seconds)
        root = ElementTree.fromstring(xml)
        for item in root.findall(".//item"):
            all_items.append(
                {
                    "source": "coindesk_rss",
                    "title": (item.findtext("title") or "").strip(),
                    "url": (item.findtext("link") or "").strip(),
                    "published_at": item.findtext("pubDate"),
                    "heat": 0.0,
                }
            )
    except Exception:
        logger.exception("coindesk_rss_fetch_failed")
        source_health["coindesk_rss"] = "degraded"

    try:
        xml = _http_text("https://cointelegraph.com/rss", config.timeout_seconds)
        root = ElementTree.fromstring(xml)
        for item in root.findall(".//item"):
            all_items.append(
                {
                    "source": "cointelegraph_rss",
                    "title": (item.findtext("title") or "").strip(),
                    "url": (item.findtext("link") or "").strip(),
                    "published_at": item.findtext("pubDate"),
                    "heat": 0.0,
                }
            )
    except Exception:
        logger.exception("cointelegraph_rss_fetch_failed")
        source_health["cointelegraph_rss"] = "degraded"

    if not all_items:
        raise CryptoNewsUpstreamError("All news sources failed.")

    return all_items, source_health


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
                "published_at": published_at,
                "heat": float(item.get("heat", 0.0) or 0.0),
            }
        )
    return normalized


def dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    title_count: dict[str, int] = {}

    for item in items:
        title_key = " ".join(item["title"].lower().split())
        title_count[title_key] = title_count.get(title_key, 0) + 1
        key = item["url"].lower().rstrip("/")
        existing = deduped.get(key)
        if existing is None or item["published_at"] > existing["published_at"]:
            deduped[key] = item

    result = list(deduped.values())
    for item in result:
        title_key = " ".join(item["title"].lower().split())
        item["duplicate_count"] = title_count.get(title_key, 1)
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
            with urlopen(request, timeout=config.timeout_seconds) as response:
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
        with urlopen(request, timeout=config.timeout_seconds) as response:
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
    source_items, source_health = fetch_sources(config)
    normalized = normalize_items(source_items)
    deduped = dedupe_items(normalized)
    scored = score_importance(deduped)
    selected = scored[: max(3, limit)]

    items: list[dict[str, Any]] = []
    for raw in selected:
        try:
            summary = summarize_with_llm(raw, lang, config)
        except CryptoNewsConfigError:
            summary = _fallback_summary(raw, lang)
        except (CryptoNewsRateLimitError, CryptoNewsUpstreamError):
            summary = _fallback_summary(raw, lang)

        item_id = hashlib.sha256(f"{raw['title']}|{raw['url']}".encode("utf-8")).hexdigest()[:16]
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
    }


def get_news_digest(lang: str = "zh", limit: Optional[int] = None, force_refresh: bool = False) -> dict[str, Any]:
    config = _load_config()
    store = CryptoNewsStore.from_env()

    safe_lang = "en" if lang == "en" else "zh"
    safe_limit = max(3, min(limit or config.default_limit, 12))
    today = _utcnow().date()

    if not force_refresh:
        cached = store.get_digest(today, safe_lang)
        if cached and isinstance(cached.get("payload"), dict):
            payload = dict(cached["payload"])
            payload["cache_hit"] = True
            return payload

    try:
        digest = build_digest(safe_lang, safe_limit, config)
    except (CryptoNewsUpstreamError, CryptoNewsRateLimitError, CryptoNewsConfigError):
        stale = store.get_latest_digest(safe_lang)
        if stale and isinstance(stale.get("payload"), dict):
            payload = dict(stale["payload"])
            payload["cache_hit"] = True
            payload["stale"] = True
            return payload
        raise

    store.upsert_digest(today, safe_lang, digest)
    digest["cache_hit"] = False
    return digest


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
    "fetch_sources",
    "get_news_digest",
    "normalize_items",
    "refresh_news_digest",
    "score_importance",
]
