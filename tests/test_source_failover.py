from __future__ import annotations

import unittest
from datetime import timedelta

from logic import crypto_news_service as service


class SourceFailoverTests(unittest.TestCase):
    def _config(self) -> service.CryptoNewsConfig:
        return service.CryptoNewsConfig(
            default_limit=6,
            cache_ttl_hours=12,
            llm_max_items=3,
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

    def test_build_digest_uses_rss_when_primary_fails(self) -> None:
        now = service._utcnow()
        original_primary = service.fetch_from_primary_apis
        original_rss = service.fetch_from_rss_feeds

        service.fetch_from_primary_apis = lambda _cfg: service.FetchResult(
            items=[],
            health_detail={
                "cryptopanic": {"status": "down", "code": "HTTP_503", "message": "temporary"},
                "newsapi": {"status": "disabled", "code": "NO_KEY", "message": "missing"},
                "thenewsapi": {"status": "disabled", "code": "NO_KEY", "message": "missing"},
            },
        )
        service.fetch_from_rss_feeds = lambda _cfg: service.FetchResult(
            items=[
                {
                    "source": "coindesk_rss",
                    "title": "ETF approval review update",
                    "url": "https://example.com/a",
                    "published_at": (now - timedelta(hours=2)).isoformat(),
                    "heat": 0,
                }
            ],
            health_detail={
                "coindesk_rss": {"status": "ok", "code": "OK", "message": "fetched"},
            },
        )

        try:
            digest = service.build_digest("en", 6, self._config())
        finally:
            service.fetch_from_primary_apis = original_primary
            service.fetch_from_rss_feeds = original_rss

        self.assertTrue(digest["items"])
        self.assertEqual(digest["source_health"]["coindesk_rss"], "ok")
        self.assertEqual(digest["source_health_detail"]["coindesk_rss"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
