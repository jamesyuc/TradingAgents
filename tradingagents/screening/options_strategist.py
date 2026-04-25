"""Stage 3: Options strategy recommendations.

Given Stage 2's structured output (direction / entry / stop / target / horizon),
this module:
1. Pulls the option chain for relevant expiries from yfinance.
2. Formats ATM-ish strikes into a compact summary the LLM can digest.
3. Asks a screener LLM to recommend 2-3 concrete option strategies with
   strike, expiry, expected premium, max loss / max gain / breakeven.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from tradingagents.dataflows.stockstats_utils import yf_ticker


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass
class OptionStrategy:
    name: str  # e.g. "Long Call", "Bull Call Spread"
    legs: List[Dict[str, Any]] = field(default_factory=list)
    # each leg: {"side":"buy"/"sell", "type":"call"/"put",
    #            "strike":float, "expiry":str, "premium":float}
    max_loss: Optional[float] = None
    max_gain: Optional[float] = None  # None means unlimited
    breakeven: Optional[float] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "legs": self.legs,
            "max_loss": self.max_loss,
            "max_gain": self.max_gain,
            "breakeven": self.breakeven,
            "rationale": self.rationale,
        }


@dataclass
class OptionsRecommendation:
    ticker: str
    direction: str
    current_price: Optional[float] = None
    entry: Optional[Tuple[float, float]] = None
    stop_loss: Optional[float] = None
    take_profits: List[float] = field(default_factory=list)
    horizon: str = ""
    chain_summary: str = ""  # compact text sent to LLM
    strategies: List[OptionStrategy] = field(default_factory=list)
    raw_llm_output: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "current_price": self.current_price,
            "entry": list(self.entry) if self.entry else None,
            "stop_loss": self.stop_loss,
            "take_profits": self.take_profits,
            "horizon": self.horizon,
            "strategies": [s.to_dict() for s in self.strategies],
            "error": self.error,
        }


# -----------------------------------------------------------------------------
# Option chain helpers
# -----------------------------------------------------------------------------
def _pick_expiries(
    available: Tuple[str, ...],
    horizon: str,
    trade_date: str,
) -> List[str]:
    """Pick 1-3 expiries aligned with the investment horizon."""
    today = datetime.strptime(trade_date, "%Y-%m-%d")

    # Parse horizon keywords into target DTE ranges.
    horizon_lower = horizon.lower()
    if any(k in horizon_lower for k in ["week", "day", "intraday", "short"]):
        targets_days = [14, 30]
    elif any(k in horizon_lower for k in ["month", "swing", "medium", "quarter"]):
        targets_days = [30, 60, 90]
    else:
        targets_days = [45, 90, 120]

    picked = []
    for tgt in targets_days:
        target_date = today + timedelta(days=tgt)
        # Find closest expiry on or after target_date.
        best = None
        for exp_str in available:
            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
            if exp_dt >= target_date:
                if best is None or exp_dt < datetime.strptime(best, "%Y-%m-%d"):
                    best = exp_str
        if best and best not in picked:
            picked.append(best)
    # fallback: last available
    if not picked and available:
        picked.append(available[-1])
    return picked[:3]


def _format_chain_for_llm(
    ticker: str,
    current_price: float,
    expiries: List[str],
    n_strikes_each_side: int = 6,
) -> str:
    """Pull option chain data and format into a compact text block for LLM."""
    tk = yf_ticker(ticker)
    sections = []
    for exp in expiries:
        try:
            chain = tk.option_chain(exp)
        except Exception as exc:
            sections.append(f"Expiry {exp}: FAILED ({exc})")
            continue

        for kind, df in [("CALLS", chain.calls), ("PUTS", chain.puts)]:
            if df.empty:
                continue
            # Find ATM index.
            atm_idx = (df["strike"] - current_price).abs().idxmin()
            lo = max(df.index.min(), atm_idx - n_strikes_each_side)
            hi = min(df.index.max(), atm_idx + n_strikes_each_side)
            subset = df.loc[lo:hi]
            cols = ["strike", "lastPrice", "bid", "ask",
                    "impliedVolatility", "openInterest", "volume"]
            cols = [c for c in cols if c in subset.columns]
            table = subset[cols].to_string(index=False)
            sections.append(f"### {ticker} {kind} — Expiry {exp}\n{table}")
    return "\n\n".join(sections) if sections else "(no chain data available)"


def get_current_price(ticker: str) -> Optional[float]:
    """Get latest close price."""
    try:
        tk = yf_ticker(ticker)
        hist = tk.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


# -----------------------------------------------------------------------------
# LLM prompt
# -----------------------------------------------------------------------------
_OPTIONS_SYS = """You are a senior options strategist. Given a stock's directional
thesis (from the investment committee) and its option chain snapshot, recommend
2-3 specific, actionable option strategies.

For EACH strategy output a JSON object inside a JSON array:
[
  {
    "name": "Bull Call Spread",
    "legs": [
      {"side": "buy", "type": "call", "strike": 120, "expiry": "2026-06-18", "premium": 5.50},
      {"side": "sell", "type": "call", "strike": 150, "expiry": "2026-06-18", "premium": 1.20}
    ],
    "max_loss": 430,
    "max_gain": 2570,
    "breakeven": 124.30,
    "rationale": "Captures 25% upside to target with capped risk..."
  }
]

Rules:
- Use ONLY strikes and premiums visible in the chain data (lastPrice column).
  If bid/ask are 0 (after-hours), use lastPrice as the premium estimate.
- Pick expiries that MATCH the stated time horizon.
- For BUY thesis: favor long calls, bull call spreads, or cash-secured puts.
- For SELL thesis: favor long puts, bear put spreads, or covered calls.
- For HOLD: favor iron condors, strangles, or no-action.
- Include at least one defined-risk strategy (spread).
- All dollar amounts are per-contract (×100 multiplier already applied).
- Be specific: exact strike, exact expiry, exact premium.
- Output ONLY the JSON array, no prose.
"""

_OPTIONS_USER = """## Stock thesis
Ticker: {ticker}
Current price: ${current_price:.2f}
Direction: {direction}
Entry: {entry}
Stop-loss: {stop_loss}
Take-profit: {take_profits}
Time horizon: {horizon}

## Option chain data
{chain_summary}
"""


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_strategies(text: str) -> List[OptionStrategy]:
    """Best-effort parse of LLM JSON array into OptionStrategy list."""
    if not text:
        return []
    m = _JSON_ARRAY_RE.search(text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    strategies = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        strategies.append(OptionStrategy(
            name=str(item.get("name", "Unknown")),
            legs=item.get("legs", []),
            max_loss=item.get("max_loss"),
            max_gain=item.get("max_gain"),
            breakeven=item.get("breakeven"),
            rationale=str(item.get("rationale", "")),
        ))
    return strategies


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def recommend_options(
    ticker: str,
    direction: str,
    entry: Optional[Tuple[float, float]],
    stop_loss: Optional[float],
    take_profits: List[float],
    horizon: str,
    trade_date: str,
    llm,
) -> OptionsRecommendation:
    """Generate option strategy recommendations for one ticker."""
    rec = OptionsRecommendation(
        ticker=ticker,
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        take_profits=take_profits,
        horizon=horizon,
    )

    # Current price
    price = get_current_price(ticker)
    if price is None:
        rec.error = "Could not fetch current price"
        return rec
    rec.current_price = price

    # Pick expiries
    tk = yf_ticker(ticker)
    try:
        avail = tk.options
    except Exception as exc:
        rec.error = f"No options data: {exc}"
        return rec
    if not avail:
        rec.error = "No expiry dates available"
        return rec

    expiries = _pick_expiries(avail, horizon, trade_date)
    print(f"[options] {ticker}: picked expiries {expiries}")

    # Build chain summary
    chain_summary = _format_chain_for_llm(ticker, price, expiries)
    rec.chain_summary = chain_summary

    # LLM call
    entry_str = (
        f"${entry[0]:.1f}-${entry[1]:.1f}" if entry else "not specified"
    )
    stop_str = f"${stop_loss:.1f}" if stop_loss else "not specified"
    tp_str = ", ".join(f"${tp:.1f}" for tp in take_profits) if take_profits else "not specified"

    user_msg = _OPTIONS_USER.format(
        ticker=ticker,
        current_price=price,
        direction=direction,
        entry=entry_str,
        stop_loss=stop_str,
        take_profits=tp_str,
        horizon=horizon or "not specified",
        chain_summary=chain_summary,
    )

    try:
        resp = llm.invoke([
            ("system", _OPTIONS_SYS),
            ("human", user_msg),
        ])
        raw = getattr(resp, "content", "") or ""
        rec.raw_llm_output = raw
        rec.strategies = _parse_strategies(raw)
    except Exception as exc:
        rec.error = f"LLM failed: {type(exc).__name__}: {exc}"

    return rec


def recommend_options_batch(
    ranked_decisions: List[Dict[str, Any]],
    trade_date: str,
    llm,
) -> List[OptionsRecommendation]:
    """Run options recommendations for a list of ranked decisions."""
    results = []
    for i, rd in enumerate(ranked_decisions, start=1):
        ticker = rd["ticker"]
        direction = rd.get("direction", "UNKNOWN")
        entry = tuple(rd["entry"]) if rd.get("entry") else None
        stop_loss = rd.get("stop_loss")
        take_profits = rd.get("take_profits", [])
        horizon = rd.get("time_horizon", "")

        print(f"\n[options] ({i}/{len(ranked_decisions)}) {ticker} ...")
        rec = recommend_options(
            ticker=ticker,
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profits=take_profits,
            horizon=horizon,
            trade_date=trade_date,
            llm=llm,
        )
        if rec.error:
            print(f"[options] {ticker} error: {rec.error}")
        else:
            print(
                f"[options] {ticker}: {len(rec.strategies)} strategies recommended"
            )
            for s in rec.strategies:
                legs_str = " / ".join(
                    f"{l['side']} {l['type']} ${l['strike']} @${l.get('premium','?')}"
                    for l in s.legs
                )
                print(
                    f"  - {s.name}: {legs_str} | "
                    f"maxloss=${s.max_loss} maxgain=${s.max_gain} "
                    f"BE=${s.breakeven}"
                )
        results.append(rec)
    return results


def render_options_report(recommendations: List[OptionsRecommendation]) -> str:
    """Render a Markdown report of options recommendations."""
    lines = ["## Stage 3: Options Strategy Recommendations\n"]

    for rec in recommendations:
        lines.append(f"### {rec.ticker} — {rec.direction}")
        if rec.current_price:
            lines.append(f"Current price: **${rec.current_price:.2f}**")
        entry_str = (
            f"${rec.entry[0]:.1f}-${rec.entry[1]:.1f}" if rec.entry else "n/a"
        )
        stop_str = f"${rec.stop_loss:.1f}" if rec.stop_loss else "n/a"
        tp_str = ", ".join(f"${tp:.1f}" for tp in rec.take_profits) if rec.take_profits else "n/a"
        lines.append(
            f"Stock plan: entry {entry_str} / stop {stop_str} / target {tp_str} / "
            f"horizon: {rec.horizon or 'n/a'}\n"
        )

        if rec.error:
            lines.append(f"**Error**: {rec.error}\n")
            continue

        if not rec.strategies:
            lines.append("*No strategies generated.*\n")
            continue

        for j, s in enumerate(rec.strategies, start=1):
            # Extract expiry from first leg for the title
            expiry_str = ""
            if s.legs:
                expiry_str = s.legs[0].get("expiry", "")
            title = f"**Strategy {j}: {s.name}**"
            if expiry_str:
                title += f"  (expires **{expiry_str}**)"
            lines.append(title + "\n")
            lines.append("| Side | Type | Strike | Expiry | Premium |")
            lines.append("|------|------|--------|--------|---------|")
            for leg in s.legs:
                lines.append(
                    f"| {leg.get('side','')} | "
                    f"{leg.get('type','')} | ${leg.get('strike','')} | "
                    f"{leg.get('expiry','')} | ${leg.get('premium','')} |"
                )
            lines.append("")
            if s.max_loss is not None:
                try:
                    lines.append(f"- **Max Loss**: ${float(s.max_loss):,.0f}")
                except (TypeError, ValueError):
                    lines.append(f"- **Max Loss**: {s.max_loss}")
            if s.max_gain is not None:
                try:
                    lines.append(f"- **Max Gain**: ${float(s.max_gain):,.0f}")
                except (TypeError, ValueError):
                    lines.append(f"- **Max Gain**: {s.max_gain}")
            elif s.max_gain is None and s.name and "long" in s.name.lower():
                lines.append("- **Max Gain**: Unlimited")
            if s.breakeven is not None:
                try:
                    lines.append(f"- **Breakeven**: ${float(s.breakeven):,.2f}")
                except (TypeError, ValueError):
                    lines.append(f"- **Breakeven**: {s.breakeven}")
            if s.rationale:
                lines.append(f"- **Rationale**: {s.rationale}")
            lines.append("")

        lines.append("---\n")

    return "\n".join(lines)
