from __future__ import annotations

import io
import unittest
from urllib.error import HTTPError

from logic import crypto_news_service as service


class _FakeResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args) -> None:
        return None


class HttpFetchRetryHeadersTests(unittest.TestCase):
    def _config(self) -> service.CryptoNewsConfig:
        return service.CryptoNewsConfig(
            default_limit=6,
            cache_ttl_hours=12,
            llm_max_items=3,
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
            http_retries=2,
            cryptopanic_key="",
            newsapi_key="",
            thenewsapi_key="",
            ai_provider="gemini",
            ai_model="gemini-2.0-flash",
            ai_key="",
        )

    def test_http_fetch_retries_and_sets_headers(self) -> None:
        calls = []

        def fake_urlopen(request, timeout=0):
            calls.append(request)
            if len(calls) == 1:
                raise HTTPError(
                    request.full_url,
                    503,
                    "temporary",
                    hdrs=None,
                    fp=io.BytesIO(b"temporary unavailable"),
                )
            return _FakeResponse('{"ok": true}')

        original = service.urlopen
        service.urlopen = fake_urlopen
        try:
            payload = service._http_fetch(
                "https://example.com/news",
                config=self._config(),
                expect_json=True,
            )
        finally:
            service.urlopen = original

        self.assertTrue(payload["ok"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0].headers.get("User-agent"))
        self.assertTrue(calls[0].headers.get("Accept"))
        self.assertTrue(calls[0].headers.get("Accept-language"))


if __name__ == "__main__":
    unittest.main()
