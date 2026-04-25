"""Stage 2: cross-sectional ranking of full-pipeline decisions.

For each ticker that survived stage-1 screening we run the full
TradingAgentsGraph to obtain a BUY/HOLD/SELL verdict plus the detailed
Portfolio Manager narrative. This module then:

1. Parses each narrative to pull out structured trade parameters
   (entry / stop / targets / horizon / conviction cues) using regex heuristics.
2. Computes a composite score on four axes and sorts the survivors.
3. Emits a compact ranked-leaderboard dict ready to render.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Data class
# -----------------------------------------------------------------------------
@dataclass
class RankedDecision:
    ticker: str
    sectors: List[str]
    direction: str  # BUY / HOLD / SELL / UNKNOWN
    conviction: str  # high / medium / low
    entry: Optional[Tuple[float, float]] = None  # (low, high) or (price, price)
    stop_loss: Optional[float] = None
    take_profits: List[float] = field(default_factory=list)
    time_horizon: str = ""
    risk_reward: Optional[float] = None
    composite_score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    final_decision_text: str = ""
    screener_score: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "sectors": self.sectors,
            "direction": self.direction,
            "conviction": self.conviction,
            "entry": list(self.entry) if self.entry else None,
            "stop_loss": self.stop_loss,
            "take_profits": self.take_profits,
            "time_horizon": self.time_horizon,
            "risk_reward": self.risk_reward,
            "composite_score": round(self.composite_score, 3),
            "score_breakdown": {k: round(v, 3) for k, v in self.score_breakdown.items()},
            "screener_score": self.screener_score,
            "error": self.error,
        }


# -----------------------------------------------------------------------------
# Narrative parsing (best-effort regex on Portfolio Manager output)
# -----------------------------------------------------------------------------

# Direction / rating. Real PM uses many formats:
#   - "FINAL TRANSACTION PROPOSAL: **BUY**"
#   - "1. **Rating**: Buy"   /  "### 1. Rating\n**Buy**"
#   - "Rating: Overweight"
#   - standalone "**Buy**" at the start
_DIRECTION_RE = re.compile(
    r"(?:"
    r"FINAL\s+TRANSACTION\s+PROPOSAL:\s*\**\s*"       # classic format
    r"|\*{0,2}Rating\*{0,2}[\s:]+\*{0,2}\s*"          # **Rating**: Buy
    r"|\*\*"                                            # **Buy** standalone
    r")"
    r"(BUY|Buy|SELL|Sell|HOLD|Hold|OVERWEIGHT|Overweight|UNDERWEIGHT|Underweight)",
)

# entry range: "$199-$200", "~$199-200", "scaling in ... $118-$120", "between $86 and $88"
_ENTRY_RANGE_RE = re.compile(
    r"(?:entry|buy|enter|scale[\s-]*in|add|accumulate|between|range|long|current\s+levels)"
    r"[^\n]{0,60}?"
    r"~?\$\s?(\d{1,6}(?:\.\d+)?)\s*(?:[-–—]|to|and)\s*~?\$?\s?(\d{1,6}(?:\.\d+)?)",
    re.IGNORECASE,
)

# single-price entry: "at current levels (~$481)", "Enter long ~$199", "entry at $123.85"
_ENTRY_SINGLE_RE = re.compile(
    r"(?:entry|enter|long|current\s+levels|buy\s+at|accumulate\s+at|scale[\s-]*in)"
    r"[^\n]{0,30}?"
    r"[~(]?\$\s?(\d{1,6}(?:\.\d+)?)",
    re.IGNORECASE,
)

# stop-loss: "$184", "stop-loss at $108", "invalidation below $192", "Stops: Initial at $184"
_STOP_RE = re.compile(
    r"(?:stop[\s-]*loss|stops?(?:\s*:\s*(?:initial\s+)?)?(?:at|below)?|hard\s+stop|invalidation|"
    r"cut\s+loss|downside\s+stop|protective\s+stop)"
    r"[^\n]{0,40}?"
    r"\$\s?(\d{1,6}(?:\.\d+)?)",
    re.IGNORECASE,
)

# take-profit: capture ALL TP levels
_TP_RE = re.compile(
    r"(?:take[\s-]*profit|(?:price\s+)?target|tp\s*\d*|profit[\s-]*target|"
    r"first\s+target|second\s+target|upside\s+target|exit\s+target|"
    r"initial\s+target|stretch\s+target|near[\s-]*term\s+target)"
    r"[^\n]{0,40}?"
    r"\$\s?(\d{1,6}(?:\.\d+)?)",
    re.IGNORECASE,
)

# time horizon
_HORIZON_RE = re.compile(
    r"(?:time[\s-]*horizon|hold(?:ing)?\s*period|horizon|investment\s+horizon)[:\s]*([^\n.;]{5,80})",
    re.IGNORECASE,
)

# conviction
_CONVICTION_RE = re.compile(
    r"\b(high|strong|medium|moderate|low|weak)\s+conviction\b",
    re.IGNORECASE,
)


def _parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_decision_text(text: str) -> Dict[str, Any]:
    """Regex-extract structured trade parameters from Portfolio Manager output."""
    if not text:
        return {}

    out: Dict[str, Any] = {}

    m = _DIRECTION_RE.search(text)
    raw_dir = m.group(1).upper() if m else "UNKNOWN"
    # normalize Overweight -> BUY, Underweight -> SELL
    out["direction"] = {
        "BUY": "BUY", "SELL": "SELL", "HOLD": "HOLD",
        "OVERWEIGHT": "BUY", "UNDERWEIGHT": "SELL",
    }.get(raw_dir, "UNKNOWN")

    m = _ENTRY_RANGE_RE.search(text)
    if m:
        lo = _parse_float(m.group(1))
        hi = _parse_float(m.group(2))
        if lo and hi:
            out["entry"] = (min(lo, hi), max(lo, hi))
    if "entry" not in out:
        m = _ENTRY_SINGLE_RE.search(text)
        if m:
            v = _parse_float(m.group(1))
            if v:
                out["entry"] = (v, v)

    m = _STOP_RE.search(text)
    if m:
        v = _parse_float(m.group(1))
        if v:
            out["stop_loss"] = v

    tps = []
    for match in _TP_RE.finditer(text):
        v = _parse_float(match.group(1))
        if v is not None:
            tps.append(v)
    # dedupe & sort ascending, keep top 3
    tps = sorted(set(tps))[:3]
    if tps:
        out["take_profits"] = tps

    m = _HORIZON_RE.search(text)
    if m:
        out["time_horizon"] = m.group(1).strip()[:80]

    m = _CONVICTION_RE.search(text)
    if m:
        word = m.group(1).lower()
        out["conviction"] = {
            "high": "high", "strong": "high",
            "medium": "medium", "moderate": "medium",
            "low": "low", "weak": "low",
        }.get(word, "medium")
    else:
        # infer from direction + presence of hedging language
        hedging_hits = len(re.findall(r"\b(however|but|risk|caution|uncertain)\b",
                                      text.lower()))
        out["conviction"] = "medium" if hedging_hits <= 4 else "low"

    return out


def _risk_reward_ratio(
    entry: Optional[Tuple[float, float]],
    stop: Optional[float],
    tps: List[float],
    direction: str,
) -> Optional[float]:
    """Compute R:R = reward / risk, using mid-entry and first take-profit."""
    if not entry or stop is None or not tps:
        return None
    entry_mid = (entry[0] + entry[1]) / 2
    first_tp = tps[0]
    if direction == "BUY":
        risk = entry_mid - stop
        reward = first_tp - entry_mid
    elif direction == "SELL":
        risk = stop - entry_mid
        reward = entry_mid - first_tp
    else:
        return None
    if risk <= 0:
        return None
    return round(reward / risk, 2)


# -----------------------------------------------------------------------------
# Composite scoring
# -----------------------------------------------------------------------------
# Weights (sum to 1.0). Tweak freely.
WEIGHT_DIRECTION = 0.30
WEIGHT_RR = 0.40
WEIGHT_CONVICTION = 0.20
WEIGHT_DIVERSIFICATION = 0.10


def _direction_score(direction: str) -> float:
    return {"BUY": 1.0, "HOLD": 0.4, "SELL": 0.0, "UNKNOWN": 0.2}.get(direction, 0.2)


def _rr_score(rr: Optional[float]) -> float:
    """R:R >= 3 -> 1.0; <= 0.5 -> 0.0; linear between."""
    if rr is None:
        return 0.3  # absence penalty, not a zero
    if rr >= 3:
        return 1.0
    if rr <= 0.5:
        return 0.0
    return (rr - 0.5) / 2.5


def _conviction_score(conv: str) -> float:
    return {"high": 1.0, "medium": 0.6, "low": 0.3}.get(conv, 0.3)


def compute_composite(
    ranked: List[RankedDecision],
) -> None:
    """Fill composite_score on each ranked item, accounting for sector spread."""
    # Sector frequency pass
    sector_counts: Dict[str, int] = {}
    for r in ranked:
        for s in r.sectors:
            sector_counts[s] = sector_counts.get(s, 0) + 1

    for r in ranked:
        d = _direction_score(r.direction)
        rr = _rr_score(r.risk_reward)
        cv = _conviction_score(r.conviction)
        # Diversification bonus: reward tickers from under-represented sectors.
        if r.sectors:
            avg_freq = sum(sector_counts.get(s, 1) for s in r.sectors) / len(r.sectors)
            div = max(0.0, min(1.0, 1.0 / avg_freq))
        else:
            div = 0.5

        composite = (
            WEIGHT_DIRECTION * d
            + WEIGHT_RR * rr
            + WEIGHT_CONVICTION * cv
            + WEIGHT_DIVERSIFICATION * div
        )
        r.composite_score = composite
        r.score_breakdown = {
            "direction": d,
            "risk_reward": rr,
            "conviction": cv,
            "diversification": div,
        }


def rank(ranked: List[RankedDecision]) -> List[RankedDecision]:
    """Compute composite scores and return a new list sorted desc."""
    compute_composite(ranked)
    return sorted(ranked, key=lambda r: r.composite_score, reverse=True)


# -----------------------------------------------------------------------------
# Convenience: build a RankedDecision from a raw pipeline output.
# -----------------------------------------------------------------------------
def build_ranked_decision(
    ticker: str,
    sectors: List[str],
    final_decision_text: str,
    screener_score: float,
    error: str = "",
) -> RankedDecision:
    parsed = parse_decision_text(final_decision_text)
    rd = RankedDecision(
        ticker=ticker,
        sectors=sectors,
        direction=parsed.get("direction", "UNKNOWN"),
        conviction=parsed.get("conviction", "medium"),
        entry=parsed.get("entry"),
        stop_loss=parsed.get("stop_loss"),
        take_profits=parsed.get("take_profits", []),
        time_horizon=parsed.get("time_horizon", ""),
        final_decision_text=final_decision_text,
        screener_score=screener_score,
        error=error,
    )
    rd.risk_reward = _risk_reward_ratio(
        rd.entry, rd.stop_loss, rd.take_profits, rd.direction
    )
    return rd
