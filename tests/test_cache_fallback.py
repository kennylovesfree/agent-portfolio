from __future__ import annotations

import unittest

from logic import crypto_news_service as service


class _FakeStore:
    def __init__(self, row: dict) -> None:
        self._row = row

    def get_digest(self, *_args, **_kwargs):
        return None

    def get_latest_success(self, *_args, **_kwargs):
        return self._row

    def upsert_digest(self, *_args, **_kwargs):
        return None


class CacheFallbackTests(unittest.TestCase):
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

    def test_get_news_digest_falls_back_to_recent_cache(self) -> None:
        now = service._utcnow()
        row = {
            "digest_date": now.date().isoformat(),
            "lang": "zh",
            "updated_at": now.isoformat().replace("+00:00", "Z"),
            "payload": {
                "as_of": now.isoformat().replace("+00:00", "Z"),
                "lang": "zh",
                "daily_overview": ["高影響 | 測試"],
                "items": [],
                "source_health": {"coindesk_rss": "ok"},
                "source_health_detail": {"coindesk_rss": {"status": "ok", "code": "OK", "message": "cached"}},
            },
        }

        original_load = service._load_config
        original_store = service.CryptoNewsStore.from_env
        original_build = service.build_digest

        service._load_config = self._config
        service.CryptoNewsStore.from_env = classmethod(lambda _cls: _FakeStore(row))
        service.build_digest = lambda *_args, **_kwargs: (_ for _ in ()).throw(service.CryptoNewsUpstreamError("all failed"))

        try:
            payload = service.get_news_digest(lang="zh", force_refresh=True)
        finally:
            service._load_config = original_load
            service.CryptoNewsStore.from_env = original_store
            service.build_digest = original_build

        self.assertTrue(payload["cache_hit"])
        self.assertTrue(payload["fallback_used"])
        self.assertEqual(payload["lang"], "zh")
        self.assertIsNotNone(payload["stale_age_hours"])


if __name__ == "__main__":
    unittest.main()
