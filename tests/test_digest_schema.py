from __future__ import annotations

import unittest
from datetime import timedelta

from logic import crypto_news_service as service


class DigestSchemaTests(unittest.TestCase):
    def _config(self) -> service.CryptoNewsConfig:
        return service.CryptoNewsConfig(
            default_limit=6,
            cache_ttl_hours=12,
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
            http_retries=1,
            cryptopanic_key="",
            newsapi_key="",
            thenewsapi_key="",
            ai_provider="gemini",
            ai_model="gemini-2.0-flash",
            ai_key="",
        )

    def test_build_digest_returns_contract_fields(self) -> None:
        now = service._utcnow()
        original_primary = service.fetch_from_primary_apis
        original_rss = service.fetch_from_rss_feeds

        service.fetch_from_primary_apis = lambda _cfg: service.FetchResult(
            items=[
                {
                    "source": "newsapi",
                    "title": "Liquidity conditions tighten after macro shock",
                    "url": "https://example.com/c",
                    "published_at": (now - timedelta(hours=4)).isoformat(),
                    "heat": 1,
                }
            ],
            health_detail={"newsapi": {"status": "ok", "code": "OK", "message": "fetched"}},
        )
        service.fetch_from_rss_feeds = lambda _cfg: service.FetchResult(
            items=[],
            health_detail={"coindesk_rss": {"status": "degraded", "code": "EMPTY", "message": "empty"}},
        )

        try:
            digest = service.build_digest("zh", 6, self._config())
        finally:
            service.fetch_from_primary_apis = original_primary
            service.fetch_from_rss_feeds = original_rss

        self.assertTrue({"as_of", "lang", "daily_overview", "items", "source_health", "source_health_detail"}.issubset(digest.keys()))
        self.assertEqual(digest["lang"], "zh")
        self.assertIsInstance(digest["daily_overview"], list)
        self.assertTrue(digest["items"])

        item = digest["items"][0]
        self.assertTrue(
            {
                "id",
                "title",
                "source",
                "published_at",
                "url",
                "importance_score",
                "impact_level",
                "risk_tags",
                "summary",
                "why_it_matters",
            }.issubset(item.keys())
        )


if __name__ == "__main__":
    unittest.main()
