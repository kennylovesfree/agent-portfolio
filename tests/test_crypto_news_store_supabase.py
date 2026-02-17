from __future__ import annotations

import io
import json
from unittest.mock import patch

from logic.crypto_news_store import CryptoNewsStore, SupabaseConfig


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args) -> None:
        return None


def test_get_digest_and_upsert() -> None:
    store = CryptoNewsStore(
        SupabaseConfig(
            url="https://demo.supabase.co",
            service_key="test-key",
            timeout_seconds=3,
        )
    )

    calls = []

    def fake_urlopen(request, timeout=0):
        calls.append((request.method, request.full_url, timeout, request.headers))
        if request.method == "GET":
            return _FakeResponse([
                {
                    "digest_date": "2026-02-17",
                    "lang": "zh",
                    "payload": {"items": []},
                    "updated_at": "2026-02-17T00:10:00Z",
                }
            ])
        return _FakeResponse([
            {
                "digest_date": "2026-02-17",
                "lang": "zh",
                "payload": {"items": [{"id": "x"}]},
            }
        ])

    with patch("logic.crypto_news_store.urlopen", side_effect=fake_urlopen):
        row = store.get_latest_digest("zh")
        assert row is not None
        assert row["lang"] == "zh"

        saved = store.upsert_digest(__import__("datetime").date(2026, 2, 17), "zh", {"items": [{"id": "x"}]})
        assert saved["payload"]["items"][0]["id"] == "x"

    assert calls[0][0] == "GET"
    assert calls[1][0] == "POST"
