"""Multi-ticker screening & ranking pipeline.

Modules:
- universe: static sector watchlists + Reddit trending tickers.
- quick_screen: stage-1 lightweight scoring (price + fundamentals only).
- ranker: stage-2 full-pipeline results aggregation and cross-sectional ranking.
- pipeline: orchestrates the full two-stage flow.
"""
