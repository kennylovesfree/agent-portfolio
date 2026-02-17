from __future__ import annotations

import unittest
from datetime import timedelta

from logic import crypto_news_service as service


class DedupeRankTests(unittest.TestCase):
    def test_merge_and_rank_dedupes_with_canonical_url(self) -> None:
        now = service._utcnow()
        primary_items = [
            {
                "source": "newsapi",
                "title": "SEC updates ETF review process",
                "url": "https://example.com/a?utm_source=foo",
                "published_at": (now - timedelta(hours=2)).isoformat(),
                "heat": 0,
            }
        ]
        rss_items = [
            {
                "source": "coindesk_rss",
                "title": "SEC updates ETF review process",
                "url": "https://example.com/a",
                "published_at": (now - timedelta(hours=1)).isoformat(),
                "heat": 0,
            },
            {
                "source": "cointelegraph_rss",
                "title": "Major exchange exploit sparks security concerns",
                "url": "https://example.com/b",
                "published_at": (now - timedelta(hours=3)).isoformat(),
                "heat": 0,
            },
        ]

        ranked = service.merge_and_rank_items(primary_items, rss_items)

        urls = [item["canonical_url"] for item in ranked]
        self.assertEqual(len(urls), len(set(urls)))
        self.assertGreaterEqual(ranked[0]["importance_score"], ranked[-1]["importance_score"])
        self.assertIn(ranked[0]["impact_level"], {"high", "medium", "low"})


if __name__ == "__main__":
    unittest.main()
