from __future__ import annotations

import unittest
from unittest.mock import patch

from logic.advice_service import (
    AdviceConfigError,
    AdviceRequest,
    AdviceResponse,
    _normalize_advice_payload,
    generate_advice,
)


def _sample_payload() -> AdviceRequest:
    return AdviceRequest.model_validate(
        {
            "profile": {"age": 35, "riskLevel": "balanced", "taxRegion": "TW", "horizonYears": 10},
            "portfolio": {
                "totalUsd": 1000,
                "expectedReturn": 0.08,
                "volatility": 0.16,
                "maxDrawdown": 0.2,
                "positions": [],
                "allocationSuggestion": [],
            },
            "locale": "zh-TW",
        }
    )


def _sample_response(provider: str, model: str) -> AdviceResponse:
    return AdviceResponse.model_validate(
        {
            "summary": "測試",
            "risk_level": "medium",
            "actions": [{"title": "調整", "reason": "測試", "priority": "medium"}],
            "watchouts": ["注意風險"],
            "disclaimer": "僅供參考",
            "model_meta": {"provider": provider, "model": model, "latency_ms": 10},
        }
    )


class AdviceServiceTests(unittest.TestCase):
    def test_generate_advice_uses_gemini_when_provider_set(self) -> None:
        payload = _sample_payload()
        with patch.dict(
            "os.environ",
            {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "x", "GEMINI_MODEL": "gemini-2.0-flash"},
            clear=False,
        ):
            with patch("logic.advice_service._call_gemini", return_value=_sample_response("gemini", "gemini-2.0-flash")):
                result = generate_advice(payload)
        self.assertEqual(result.model_meta.provider, "gemini")

    def test_generate_advice_gemini_missing_key_raises(self) -> None:
        payload = _sample_payload()
        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": ""}, clear=False):
            with self.assertRaises(AdviceConfigError):
                generate_advice(payload)

    def test_normalize_advice_payload_handles_schema_drift(self) -> None:
        raw = {
            "summary": {"text": "市場波動升高"},
            "risk_level": "高風險",
            "actions": ["降低單一資產集中", {"title": "提高現金比重"}],
            "watchouts": "注意流動性風險",
            "disclaimer": None,
        }
        normalized = _normalize_advice_payload(
            raw,
            provider="gemini",
            model="gemini-3-flash-preview",
            latency_ms=25,
            locale="zh-TW",
        )
        self.assertEqual(normalized["risk_level"], "high")
        self.assertTrue(len(normalized["actions"]) >= 1)
        self.assertIsInstance(normalized["watchouts"], list)
        self.assertIn("model_meta", normalized)


if __name__ == "__main__":
    unittest.main()
