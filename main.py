"""Multi-ticker screening + ranking entrypoint.

Pipeline (two-stage, see tradingagents/screening/):
    1. Universe  = static sector watchlists  ∪  Apewisdom social trending
    2. Stage 1   = Market + Fundamentals analysts only, plus a screener LLM
                   that emits a JSON verdict & 0-10 score  (fast, cheap)
    3. Stage 2   = Top-N survivors run the full TradingAgentsGraph
                   (Bull/Bear/Trader/Risk debate + Portfolio Manager)
    4. Ranker    = parses each PM narrative for entry / stop / target,
                   computes risk-reward, and sorts by composite score

Artifacts per run land in ./screening_runs/<trade_date>/.
"""

from datetime import datetime

from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.screening.pipeline import run_pipeline


load_dotenv()

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
config = DEFAULT_CONFIG.copy()
# Only grok-4-fast (via AI Builders Space) is verified to transmit OpenAI
# function-tool calls correctly for TradingAgents' LangGraph tool loop.
config["llm_provider"] = "aibuilders"
config["backend_url"] = "https://space.ai-builders.com/backend/v1"
config["deep_think_llm"] = "grok-4-fast"
config["quick_think_llm"] = "grok-4-fast"
config["max_debate_rounds"] = 1

config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}


if __name__ == "__main__":
    trade_date = datetime.now().strftime("%Y-%m-%d")
    print(f"[main] Running two-stage screener as of {trade_date}")

    result = run_pipeline(
        trade_date=trade_date,
        config=config,
        # Universe construction
        reddit_top_n=10,
        reddit_min_mentions=20,
        include_reddit=True,
        # Stage 1 -> Stage 2 handoff
        stage1_min_score=5.5,
        top_n_for_deep=6,
        out_dir="./screening_runs",
    )
    print(
        f"\n[main] Done. Universe={result['universe_size']}, "
        f"stage2={result['stage2_count']}, "
        f"elapsed={result['overall_elapsed_sec'] / 60:.1f} min"
    )
