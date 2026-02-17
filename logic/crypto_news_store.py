"""Supabase-backed store for crypto news digest cache."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class CryptoNewsStoreConfigError(RuntimeError):
    """Raised when Supabase config is missing."""


class CryptoNewsStoreError(RuntimeError):
    """Raised when Supabase request fails."""


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    service_key: str
    timeout_seconds: float


class CryptoNewsStore:
    def __init__(self, config: SupabaseConfig) -> None:
        self._config = config

    @classmethod
    def from_env(cls) -> "CryptoNewsStore":
        url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        timeout = float(os.getenv("CRYPTO_NEWS_STORE_TIMEOUT_SEC", "8"))
        if not url or not key:
            raise CryptoNewsStoreConfigError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing.")
        return cls(SupabaseConfig(url=url, service_key=key, timeout_seconds=max(1.0, timeout)))

    def get_digest(self, digest_date: date, lang: str) -> Optional[dict[str, Any]]:
        params = urlencode(
            {
                "select": "digest_date,lang,payload,updated_at",
                "digest_date": f"eq.{digest_date.isoformat()}",
                "lang": f"eq.{lang}",
                "limit": "1",
            }
        )
        rows = self._request_json("GET", f"/rest/v1/crypto_news_digest_cache?{params}")
        if not rows:
            return None
        return rows[0]

    def get_latest_digest(self, lang: str) -> Optional[dict[str, Any]]:
        params = urlencode(
            {
                "select": "digest_date,lang,payload,updated_at",
                "lang": f"eq.{lang}",
                "order": "digest_date.desc",
                "limit": "1",
            }
        )
        rows = self._request_json("GET", f"/rest/v1/crypto_news_digest_cache?{params}")
        if not rows:
            return None
        return rows[0]

    def upsert_digest(self, digest_date: date, lang: str, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = "/rest/v1/crypto_news_digest_cache?on_conflict=digest_date,lang"
        body = [{"digest_date": digest_date.isoformat(), "lang": lang, "payload": payload}]
        rows = self._request_json(
            "POST",
            endpoint,
            body=body,
            extra_headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )
        if not rows:
            raise CryptoNewsStoreError("Supabase upsert returned empty response.")
        return rows[0]

    def _request_json(
        self,
        method: str,
        path: str,
        body: Any | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        headers = {
            "apikey": self._config.service_key,
            "Authorization": f"Bearer {self._config.service_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            f"{self._config.url}{path}",
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urlopen(request, timeout=self._config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise CryptoNewsStoreError(f"Supabase HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise CryptoNewsStoreError(f"Supabase unavailable: {exc}") from exc

        if not raw:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CryptoNewsStoreError("Supabase response is not valid JSON.") from exc
