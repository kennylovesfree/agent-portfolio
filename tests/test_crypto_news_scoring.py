from __future__ import annotations

from datetime import timedelta

from logic import crypto_news_service as service


def test_normalize_dedupe_and_score_importance() -> None:
    now = service._utcnow()
    items = [
        {
            "source": "coindesk_rss",
            "title": "SEC opens new ETF review for Bitcoin markets",
            "url": "https://example.com/a",
            "published_at": (now - timedelta(hours=2)).isoformat(),
            "heat": 2,
        },
        {
            "source": "cointelegraph_rss",
            "title": "SEC opens new ETF review for Bitcoin markets",
            "url": "https://example.com/a/",
            "published_at": (now - timedelta(hours=1)).isoformat(),
            "heat": 1,
        },
        {
            "source": "cryptopanic",
            "title": "Protocol hack leads to major exploit on exchange",
            "url": "https://example.com/b",
            "published_at": (now - timedelta(hours=3)).isoformat(),
            "heat": 6,
        },
    ]

    normalized = service.normalize_items(items)
    assert len(normalized) == 3

    deduped = service.dedupe_items(normalized)
    assert len(deduped) == 2

    scored = service.score_importance(deduped)
    assert scored[0]["importance_score"] >= scored[1]["importance_score"]
    assert scored[0]["impact_level"] in {"high", "medium", "low"}
    assert isinstance(scored[0]["risk_tags"], list)
