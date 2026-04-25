"""Candidate universe construction.

Two sources merged:
1. Static sector watchlists (editable).
2. Social trending tickers via Apewisdom (aggregates Reddit / X mentions).

Why Apewisdom? As of 2024+, Reddit's public .json endpoints are blocked (403)
for unauthenticated requests. Apewisdom exposes an open JSON API that already
aggregates mentions and upvotes across WSB / stocks / options / investing /
stocktwits, which is exactly what we need for a trending-ticker feed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import urllib.request
import urllib.error


# -----------------------------------------------------------------------------
# Static sector lists. Edit freely; keys become sector labels in the final report.
# -----------------------------------------------------------------------------
SECTOR_WATCHLIST: Dict[str, List[str]] = {
    "storage_memory": ["MU", "WDC", "STX", "SNDK"],
    "solar": ["FSLR", "ENPH", "NXT", "ARRY", "SEDG"],
    "nuclear": ["CCJ", "SMR", "OKLO", "NNE", "VST", "CEG", "BWXT", "LEU"],
    "ai_semi": ["NVDA", "AMD", "AVGO", "TSM", "MRVL"],
}

# -----------------------------------------------------------------------------
# Apewisdom social trending feed.
# Docs: https://apewisdom.io/api/
# Filters: "all-stocks" | "wallstreetbets" | "stocks" | "options" | ...
# -----------------------------------------------------------------------------
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/{filter}/page/{page}"

# Non-equity symbols returned by apewisdom we want to exclude.
# (BTC / ETH / DOGE etc. – this project is equities-only for now.)
CRYPTO_BLACKLIST: Set[str] = {
    "BTC", "ETH", "DOGE", "SHIB", "XRP", "ADA", "SOL", "MATIC", "DOT",
    "AVAX", "LINK", "LTC", "BCH", "UNI", "ATOM", "FIL", "TRX", "ALGO",
    "PEPE", "BONK", "WIF", "HBAR",
}


@dataclass
class TickerCandidate:
    ticker: str
    sectors: Set[str] = field(default_factory=set)  # e.g. {"nuclear"}
    reddit_mentions: int = 0
    reddit_score: int = 0  # upvotes (apewisdom-aggregated)
    reddit_rank: int | None = None
    reddit_rank_prev_24h: int | None = None

    def merge(self, other: "TickerCandidate") -> None:
        self.sectors |= other.sectors
        self.reddit_mentions += other.reddit_mentions
        self.reddit_score += other.reddit_score


def _http_get_json(url: str, timeout: int = 15) -> dict | None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
            ),
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"[universe] fetch failed: {url} -> {exc}")
        return None
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[universe] parse failed: {url} -> {exc}")
        return None


def scan_social_trending(
    source_filter: str = "all-stocks",
    top_n: int = 15,
    min_mentions: int = 10,
) -> List[dict]:
    """Pull top-N trending tickers from Apewisdom.

    Returns a list of dicts with keys: ticker, name, mentions, upvotes,
    rank, rank_24h_ago.
    """
    url = APEWISDOM_URL.format(filter=source_filter, page=1)
    payload = _http_get_json(url)
    if not payload:
        print("[universe] apewisdom unreachable; skipping social trending")
        return []

    results = payload.get("results", []) or []
    trimmed: List[dict] = []
    for r in results:
        tkr = (r.get("ticker") or "").upper()
        if not tkr or tkr in CRYPTO_BLACKLIST:
            continue
        mentions = int(r.get("mentions") or 0)
        if mentions < min_mentions:
            continue
        trimmed.append({
            "ticker": tkr,
            "name": r.get("name"),
            "mentions": mentions,
            "upvotes": int(r.get("upvotes") or 0),
            "rank": r.get("rank"),
            "rank_24h_ago": r.get("rank_24h_ago"),
        })
        if len(trimmed) >= top_n:
            break
    return trimmed


def build_universe(
    sector_watchlist: Dict[str, List[str]] | None = None,
    reddit_top_n: int = 10,
    reddit_min_mentions: int = 10,
    include_reddit: bool = True,
) -> List[TickerCandidate]:
    """Build the combined candidate universe: sector lists ∪ social trending."""
    sector_watchlist = sector_watchlist or SECTOR_WATCHLIST

    registry: Dict[str, TickerCandidate] = {}

    # Static sector watchlists
    for sector, tickers in sector_watchlist.items():
        for t in tickers:
            t = t.upper()
            cand = registry.setdefault(t, TickerCandidate(ticker=t))
            cand.sectors.add(sector)

    # Social trending (apewisdom)
    if include_reddit:
        print("[universe] scanning social trending (Apewisdom) ...")
        trending = scan_social_trending(
            top_n=reddit_top_n,
            min_mentions=reddit_min_mentions,
        )
        for item in trending:
            tkr = item["ticker"]
            cand = registry.setdefault(tkr, TickerCandidate(ticker=tkr))
            cand.reddit_mentions = item["mentions"]
            cand.reddit_score = item["upvotes"]
            cand.reddit_rank = item.get("rank")
            cand.reddit_rank_prev_24h = item.get("rank_24h_ago")
            if not cand.sectors:
                cand.sectors.add("social_trending")
        print(
            f"[universe] social top {len(trending)}: "
            f"{[i['ticker'] for i in trending]}"
        )

    universe = sorted(registry.values(), key=lambda c: c.ticker)
    print(f"[universe] total candidates: {len(universe)}")
    return universe


def dump_universe(universe: List[TickerCandidate], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "ticker": c.ticker,
            "sectors": sorted(c.sectors),
            "reddit_mentions": c.reddit_mentions,
            "reddit_score": c.reddit_score,
            "reddit_rank": c.reddit_rank,
            "reddit_rank_prev_24h": c.reddit_rank_prev_24h,
        }
        for c in universe
    ]
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
