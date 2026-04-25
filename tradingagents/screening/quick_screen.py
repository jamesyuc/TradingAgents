"""Stage 1: quick screening.

For each candidate we run only two analysts (Market + Fundamentals) plus a
lightweight screener LLM call that emits a structured JSON score. This is
intentionally cheaper than the full TradingAgents pipeline so we can filter
~30 tickers down to ~10 before paying for the expensive bull/bear/risk debates.

We avoid the main graph entirely here. Instead we drive each analyst node
directly through its ReAct-style tool loop, which gives us:
    - full control over recursion/tool budget per ticker
    - no bull/bear/research-manager/trader/risk path (saves ~70% of tokens)
    - easy failure isolation (one ticker throwing doesn't poison the batch)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.fundamentals_analyst import (
    create_fundamentals_analyst,
)
from tradingagents.agents.utils.agent_utils import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_indicators,
    get_stock_data,
)

from .universe import TickerCandidate


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass
class QuickScreenResult:
    ticker: str
    sectors: List[str]
    reddit_mentions: int
    reddit_score: int
    # outputs
    market_report: str = ""
    fundamentals_report: str = ""
    screener_score: float = 0.0  # 0..10
    screener_direction: str = "hold"  # bullish / neutral / bearish / hold
    screener_rationale: str = ""
    error: str = ""
    elapsed_sec: float = 0.0
    tool_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "sectors": self.sectors,
            "reddit_mentions": self.reddit_mentions,
            "reddit_score": self.reddit_score,
            "screener_score": self.screener_score,
            "screener_direction": self.screener_direction,
            "screener_rationale": self.screener_rationale,
            "market_report": self.market_report,
            "fundamentals_report": self.fundamentals_report,
            "error": self.error,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "tool_calls": self.tool_calls,
        }


# -----------------------------------------------------------------------------
# Helpers: drive a single analyst node end-to-end (ReAct tool loop) by hand.
# -----------------------------------------------------------------------------
def _run_analyst_until_done(
    analyst_fn,
    tool_node: ToolNode,
    state: Dict[str, Any],
    report_key: str,
    max_tool_iters: int = 8,
) -> tuple[str, int]:
    """Drive a TradingAgents analyst node to completion outside the main graph.

    TradingAgents analysts are ReAct-style: they alternate analyst -> tools
    until the LLM returns a message with no tool calls (the final report).
    """
    call_count = 0
    for _ in range(max_tool_iters):
        out = analyst_fn(state)
        # Analyst node returns { "messages": [AIMessage], report_key: "..." }
        state["messages"] = state["messages"] + out["messages"]
        report = out.get(report_key, "") or ""

        last = out["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            return report, call_count

        # Let the tool node execute the requested tools, append ToolMessages.
        call_count += len(tool_calls)
        tool_out = tool_node.invoke({"messages": state["messages"]})
        # tool_node returns {"messages": [ToolMessage, ...]}
        state["messages"] = state["messages"] + tool_out["messages"]

    # exhausted loop; return whatever we have
    return state.get(report_key, "") or "", call_count


# -----------------------------------------------------------------------------
# Screener LLM prompt: turn the two reports into a structured JSON score.
# -----------------------------------------------------------------------------
_SCREENER_SYS = """You are a buy-side analyst doing a *fast* pre-screen.

You will receive a market/technical report and a fundamentals report for ONE
ticker. Your job is to emit a compact JSON verdict that the next stage will
use to decide whether to run a full investment committee on this ticker.

Be decisive. It is fine to say "skip".

Output ONLY a single JSON object, no prose before or after:
{
  "score": <float 0-10, higher means more interesting right now>,
  "direction": "bullish" | "neutral" | "bearish",
  "conviction": "low" | "medium" | "high",
  "key_catalysts": [<short strings>],
  "key_risks": [<short strings>],
  "one_line_thesis": "<<=25 words>",
  "skip_reason": "<empty string, unless score<3 then explain why to skip>"
}

Scoring rubric:
- 8-10: clear multi-factor setup, both technicals and fundamentals align
- 5-7:  mixed but tradable, one side supports the other
- 3-4:  weak, would need special edge to act
- 0-2:  skip, no actionable edge
"""

_SCREENER_USER = """Ticker: {ticker}
Sectors / tags: {sectors}
Social buzz: {mentions} mentions, {upvotes} upvotes (Apewisdom)
Trade date: {trade_date}

=== MARKET / TECHNICAL REPORT ===
{market_report}

=== FUNDAMENTALS REPORT ===
{fundamentals_report}
"""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_screener_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from LLM output."""
    if not text:
        return {}
    # First try direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to first {...} block.
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def quick_screen(
    candidates: List[TickerCandidate],
    trade_date: str,
    quick_llm,
    verbose: bool = True,
) -> List[QuickScreenResult]:
    """Run stage-1 screening for every candidate. Returns scored list."""
    market_analyst = create_market_analyst(quick_llm)
    fundamentals_analyst = create_fundamentals_analyst(quick_llm)
    market_tools = ToolNode([get_stock_data, get_indicators])
    fundamentals_tools = ToolNode(
        [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]
    )

    results: List[QuickScreenResult] = []

    for i, cand in enumerate(candidates, start=1):
        print(
            f"\n[screen] ({i}/{len(candidates)}) {cand.ticker} "
            f"sectors={sorted(cand.sectors)} ..."
        )
        start = time.time()
        res = QuickScreenResult(
            ticker=cand.ticker,
            sectors=sorted(cand.sectors),
            reddit_mentions=cand.reddit_mentions,
            reddit_score=cand.reddit_score,
        )

        try:
            # --- Market analyst ---
            mkt_state = {
                "messages": [HumanMessage(content=cand.ticker)],
                "company_of_interest": cand.ticker,
                "trade_date": trade_date,
                "market_report": "",
            }
            mkt_report, mkt_calls = _run_analyst_until_done(
                market_analyst, market_tools, mkt_state, "market_report"
            )
            res.market_report = mkt_report
            res.tool_calls += mkt_calls

            # --- Fundamentals analyst ---
            fnd_state = {
                "messages": [HumanMessage(content=cand.ticker)],
                "company_of_interest": cand.ticker,
                "trade_date": trade_date,
                "fundamentals_report": "",
            }
            fnd_report, fnd_calls = _run_analyst_until_done(
                fundamentals_analyst,
                fundamentals_tools,
                fnd_state,
                "fundamentals_report",
            )
            res.fundamentals_report = fnd_report
            res.tool_calls += fnd_calls

            # --- Screener LLM verdict ---
            user_msg = _SCREENER_USER.format(
                ticker=cand.ticker,
                sectors=", ".join(sorted(cand.sectors)),
                mentions=cand.reddit_mentions,
                upvotes=cand.reddit_score,
                trade_date=trade_date,
                market_report=mkt_report or "(no market report)",
                fundamentals_report=fnd_report or "(no fundamentals report)",
            )
            verdict_msg = quick_llm.invoke([
                ("system", _SCREENER_SYS),
                ("human", user_msg),
            ])
            verdict_text = getattr(verdict_msg, "content", "") or ""
            verdict = _parse_screener_json(verdict_text)

            res.screener_score = float(verdict.get("score", 0) or 0)
            res.screener_direction = str(
                verdict.get("direction", "neutral") or "neutral"
            )
            res.screener_rationale = str(
                verdict.get("one_line_thesis", "") or verdict_text[:300]
            )
            # Store the full verdict in the report for later auditing.
            res.market_report = mkt_report
            res.fundamentals_report = fnd_report

        except Exception as exc:  # pylint: disable=broad-except
            res.error = f"{type(exc).__name__}: {exc}"
            print(f"[screen] {cand.ticker} failed: {res.error}")

        res.elapsed_sec = time.time() - start
        if verbose:
            print(
                f"[screen] {cand.ticker} -> score={res.screener_score:.1f} "
                f"dir={res.screener_direction} "
                f"({res.elapsed_sec:.1f}s, {res.tool_calls} tool calls) "
                f"| {res.screener_rationale[:120]}"
            )
        results.append(res)

    return results


def select_top_n(
    results: List[QuickScreenResult],
    top_n: int = 10,
    min_score: float = 5.0,
) -> List[QuickScreenResult]:
    """Pick the top-N by screener_score, with a floor."""
    scored = [r for r in results if not r.error and r.screener_score >= min_score]
    scored.sort(key=lambda r: r.screener_score, reverse=True)
    return scored[:top_n]
