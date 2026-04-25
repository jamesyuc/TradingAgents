"""End-to-end three-stage screening + ranking + options pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from .quick_screen import QuickScreenResult, quick_screen, select_top_n
from .ranker import RankedDecision, build_ranked_decision, rank
from .options_strategist import recommend_options_batch, render_options_report
from .universe import TickerCandidate, build_universe, dump_universe


def run_pipeline(
    trade_date: str,
    config: Dict[str, Any] | None = None,
    reddit_top_n: int = 10,
    reddit_min_mentions: int = 10,
    include_reddit: bool = True,
    stage1_min_score: float = 5.0,
    top_n_for_deep: int = 8,
    out_dir: str | Path = "./screening_runs",
    custom_universe: List[TickerCandidate] | None = None,
) -> Dict[str, Any]:
    """Run the full two-stage pipeline and write reports.

    Returns a dict with the final leaderboard + per-stage artifacts.
    """
    cfg = dict(config or DEFAULT_CONFIG)
    out_dir = Path(out_dir) / trade_date
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()

    # --------------------------------------------------------------
    # 1. Build universe
    # --------------------------------------------------------------
    if custom_universe is not None:
        universe = custom_universe
        print(f"[pipeline] custom universe provided: {len(universe)} tickers")
    else:
        universe = build_universe(
            reddit_top_n=reddit_top_n,
            reddit_min_mentions=reddit_min_mentions,
            include_reddit=include_reddit,
        )
    dump_universe(universe, out_dir / "universe.json")

    # --------------------------------------------------------------
    # 2. Stage 1: quick screen
    # --------------------------------------------------------------
    # We construct one TradingAgentsGraph just to borrow its LLM clients.
    # (Creating the graph is cheap; it's the propagate() that costs.)
    tg = TradingAgentsGraph(debug=False, config=cfg)
    quick_llm = tg.quick_thinking_llm

    print("\n" + "=" * 72)
    print(f"STAGE 1: quick screen on {len(universe)} candidates")
    print("=" * 72)
    stage1_start = time.time()
    stage1: List[QuickScreenResult] = quick_screen(universe, trade_date, quick_llm)
    stage1_elapsed = time.time() - stage1_start
    (out_dir / "stage1_results.json").write_text(
        json.dumps([r.to_dict() for r in stage1], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[pipeline] stage 1 done in {stage1_elapsed / 60:.1f} min")

    # --------------------------------------------------------------
    # 3. Pick top-N, run full pipeline on each
    # --------------------------------------------------------------
    survivors = select_top_n(stage1, top_n=top_n_for_deep, min_score=stage1_min_score)
    print(
        f"\n[pipeline] stage 1 picked top {len(survivors)} "
        f"(min_score={stage1_min_score}): "
        f"{[s.ticker for s in survivors]}"
    )
    if not survivors:
        print("[pipeline] no survivors; aborting.")
        return {
            "trade_date": trade_date,
            "universe_size": len(universe),
            "stage1": [r.to_dict() for r in stage1],
            "stage2": [],
            "leaderboard": [],
        }

    print("\n" + "=" * 72)
    print(f"STAGE 2: full pipeline on {len(survivors)} survivors")
    print("=" * 72)

    ranked_items: List[RankedDecision] = []
    for i, s in enumerate(survivors, start=1):
        print(
            f"\n[pipeline] ({i}/{len(survivors)}) deep dive on {s.ticker} "
            f"(stage1 score={s.screener_score:.1f})"
        )
        t0 = time.time()
        try:
            final_state, _short_label = tg.propagate(s.ticker, trade_date)
            # The full Portfolio Manager narrative lives in
            # final_state["final_trade_decision"]; _short_label is just
            # "BUY"/"SELL"/"HOLD" (compressed by process_signal).
            full_narrative = final_state.get("final_trade_decision", "") or ""
            rd = build_ranked_decision(
                ticker=s.ticker,
                sectors=s.sectors,
                final_decision_text=full_narrative,
                screener_score=s.screener_score,
            )
        except Exception as exc:  # pylint: disable=broad-except
            rd = build_ranked_decision(
                ticker=s.ticker,
                sectors=s.sectors,
                final_decision_text="",
                screener_score=s.screener_score,
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"[pipeline] {s.ticker} failed: {rd.error}")
        ranked_items.append(rd)
        print(
            f"[pipeline] {s.ticker} -> {rd.direction} "
            f"entry={rd.entry} stop={rd.stop_loss} tp={rd.take_profits} "
            f"RR={rd.risk_reward} ({time.time() - t0:.1f}s)"
        )

    # --------------------------------------------------------------
    # 4. Rank
    # --------------------------------------------------------------
    leaderboard = rank(ranked_items)

    # Write full pipeline narratives separately so the JSON stays readable.
    narratives_dir = out_dir / "narratives"
    narratives_dir.mkdir(exist_ok=True)
    for r in leaderboard:
        (narratives_dir / f"{r.ticker}.md").write_text(
            f"# {r.ticker} — {r.direction} (composite={r.composite_score:.2f})\n\n"
            f"Sectors: {', '.join(r.sectors)}\n\n"
            f"R:R={r.risk_reward}  Entry={r.entry}  Stop={r.stop_loss}  "
            f"TP={r.take_profits}  Horizon={r.time_horizon}\n\n"
            f"## Portfolio Manager narrative\n\n"
            f"{r.final_decision_text}\n",
            encoding="utf-8",
        )

    print("\n" + "=" * 72)
    print("LEADERBOARD")
    print("=" * 72)
    print(render_leaderboard(leaderboard))

    # --------------------------------------------------------------
    # 5. Stage 3: Options strategy recommendations
    # --------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"STAGE 3: Options strategies for {len(leaderboard)} ranked tickers")
    print("=" * 72)

    options_input = [r.to_dict() for r in leaderboard]
    options_recs = recommend_options_batch(options_input, trade_date, quick_llm)
    options_report_md = render_options_report(options_recs)

    (out_dir / "options_report.md").write_text(options_report_md, encoding="utf-8")
    (out_dir / "options_report.json").write_text(
        json.dumps([r.to_dict() for r in options_recs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(options_report_md)

    overall_elapsed = time.time() - overall_start

    # Generate unified report
    full_report_md = render_full_report(
        trade_date=trade_date,
        stage1=stage1,
        leaderboard=leaderboard,
        options_recs=options_recs,
        elapsed_sec=overall_elapsed,
    )
    (out_dir / "REPORT.md").write_text(full_report_md, encoding="utf-8")
    print("\n" + full_report_md)

    result = {
        "trade_date": trade_date,
        "universe_size": len(universe),
        "stage1_count": len(stage1),
        "stage2_count": len(ranked_items),
        "overall_elapsed_sec": round(overall_elapsed, 1),
        "stage1": [r.to_dict() for r in stage1],
        "stage2_leaderboard": [r.to_dict() for r in leaderboard],
        "stage3_options": [r.to_dict() for r in options_recs],
    }
    (out_dir / "final_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[pipeline] total elapsed: {overall_elapsed / 60:.1f} min")
    print(f"[pipeline] artifacts -> {out_dir}")

    # Generate HTML report
    try:
        html = render_html_report(
            trade_date=trade_date,
            stage1=stage1,
            leaderboard=leaderboard,
            options_recs=options_recs,
            elapsed_sec=overall_elapsed,
        )
        (out_dir / "report.html").write_text(html, encoding="utf-8")
        print(f"[pipeline] HTML report -> {out_dir / 'report.html'}")
    except Exception as exc:
        print(f"[pipeline] HTML generation failed: {exc}")

    return result


def render_leaderboard(leaderboard: List[RankedDecision]) -> str:
    """Pretty-print the top decisions as a monospace table."""
    if not leaderboard:
        return "(empty)"
    header = (
        f"{'Rank':<5}{'Ticker':<8}{'Dir':<6}{'Conv':<8}"
        f"{'Entry':<18}{'Stop':<8}{'TP':<22}{'RR':<6}{'Score':<7}"
    )
    lines = [header, "-" * len(header)]
    for i, r in enumerate(leaderboard, start=1):
        entry_str = (
            f"${r.entry[0]:.1f}-{r.entry[1]:.1f}"
            if r.entry and r.entry[0] != r.entry[1]
            else (f"${r.entry[0]:.1f}" if r.entry else "-")
        )
        tp_str = ",".join(f"${tp:.1f}" for tp in r.take_profits[:3]) or "-"
        lines.append(
            f"{i:<5}{r.ticker:<8}{r.direction:<6}{r.conviction:<8}"
            f"{entry_str:<18}"
            f"{('$' + format(r.stop_loss, '.1f')) if r.stop_loss else '-':<8}"
            f"{tp_str:<22}"
            f"{(str(r.risk_reward) if r.risk_reward else '-'):<6}"
            f"{r.composite_score:.2f}"
        )
    return "\n".join(lines)


def render_full_report(
    trade_date: str,
    stage1: List[QuickScreenResult],
    leaderboard: List[RankedDecision],
    options_recs: list,
    elapsed_sec: float,
) -> str:
    """Generate a single unified Markdown report across all stages."""
    from .options_strategist import OptionsRecommendation

    # Build lookup: ticker -> options rec
    opts_map = {}
    for rec in options_recs:
        if isinstance(rec, OptionsRecommendation):
            opts_map[rec.ticker] = rec
        elif isinstance(rec, dict):
            opts_map[rec.get("ticker", "")] = rec

    # Build lookup: ticker -> stage1 result
    s1_map = {}
    for s in stage1:
        t = s.ticker if isinstance(s, QuickScreenResult) else s.get("ticker", "")
        s1_map[t] = s

    lines = []
    lines.append(f"# Trading Agents Screening Report\n")
    lines.append(f"**Date**: {trade_date}  ")
    lines.append(f"**Pipeline**: 3-stage (quick screen → investment committee → options strategist)  ")
    lines.append(f"**Universe**: {len(stage1)} candidates screened\n")
    lines.append("---\n")

    # ── Stage 1 summary ──
    lines.append("## Stage 1: Quick Screen\n")
    lines.append("| Rank | Ticker | Score | Direction | One-line Thesis |")
    lines.append("|------|--------|-------|-----------|-----------------|")
    s1_sorted = sorted(
        stage1,
        key=lambda x: (x.screener_score if isinstance(x, QuickScreenResult) else x.get("screener_score", 0)),
        reverse=True,
    )
    for i, s in enumerate(s1_sorted, start=1):
        if isinstance(s, QuickScreenResult):
            t, sc, d, r = s.ticker, s.screener_score, s.screener_direction, s.screener_rationale
        else:
            t = s.get("ticker", "")
            sc = s.get("screener_score", 0)
            d = s.get("screener_direction", "")
            r = s.get("screener_rationale", "")
        bold = "**" if t in [rd.ticker for rd in leaderboard] else ""
        lines.append(f"| {i} | {bold}{t}{bold} | {sc:.1f} | {d} | {r[:80]} |")
    top_tickers = [rd.ticker for rd in leaderboard]
    lines.append(f"\n**Top {len(leaderboard)} passed to Stage 2**: {', '.join(top_tickers)}\n")
    lines.append("---\n")

    # ── Stage 2 leaderboard ──
    lines.append("## Stage 2: Investment Committee Leaderboard\n")
    lines.append("| Rank | Ticker | Direction | Conviction | Entry | Stop | Target | R:R | Score |")
    lines.append("|------|--------|-----------|------------|-------|------|--------|-----|-------|")
    for i, r in enumerate(leaderboard, start=1):
        entry_str = f"${r.entry[0]:.0f}-${r.entry[1]:.0f}" if r.entry and r.entry[0] != r.entry[1] else (f"${r.entry[0]:.0f}" if r.entry else "-")
        stop_str = f"${r.stop_loss:.0f}" if r.stop_loss else "-"
        tp_str = ", ".join(f"${tp:.0f}" for tp in r.take_profits[:2]) or "-"
        rr_str = f"{r.risk_reward:.1f}" if r.risk_reward else "-"
        lines.append(f"| **{i}** | **{r.ticker}** | {r.direction} | {r.conviction} | {entry_str} | {stop_str} | {tp_str} | {rr_str} | {r.composite_score:.2f} |")
    lines.append("")

    # ── Per-ticker detail: basis + stock plan + options ──
    lines.append("---\n")
    for i, r in enumerate(leaderboard, start=1):
        lines.append(f"## #{i} {r.ticker} — {r.direction}\n")

        # Decision basis from stage 1
        s1 = s1_map.get(r.ticker)
        if s1:
            thesis = s1.screener_rationale if isinstance(s1, QuickScreenResult) else s1.get("screener_rationale", "")
            lines.append("### 决策依据\n")
            lines.append(f"**一句话**: {thesis}\n")
            # Key catalysts/risks from the PM narrative (extract bullet points)
            narrative = r.final_decision_text or ""
            # Extract a concise summary: first 2 paragraphs or executive summary
            paras = [p.strip() for p in narrative.split("\n\n") if p.strip()]
            # Find executive summary paragraph
            exec_para = ""
            thesis_para = ""
            for p in paras:
                p_lower = p.lower()
                if "executive summary" in p_lower or "entry" in p_lower[:50]:
                    exec_para = p
                elif "thesis" in p_lower[:30] or "investment thesis" in p_lower[:30]:
                    thesis_para = p
            if exec_para:
                # Clean up markdown formatting
                clean = exec_para
                for prefix in ["2. **Executive Summary**:", "2. **Executive Summary**: ", "### 2. Executive Summary\n"]:
                    clean = clean.replace(prefix, "")
                lines.append(f"**执行摘要**: {clean.strip()[:500]}\n")
            if thesis_para:
                clean = thesis_para
                for prefix in ["3. **Investment Thesis**:", "3. **Investment Thesis**: "]:
                    clean = clean.replace(prefix, "")
                lines.append(f"**核心逻辑**: {clean.strip()[:400]}\n")

        # Stock trade plan
        lines.append("### 股票交易计划\n")
        entry_str = f"${r.entry[0]:.1f} - ${r.entry[1]:.1f}" if r.entry and r.entry[0] != r.entry[1] else (f"${r.entry[0]:.1f}" if r.entry else "未指定")
        stop_str = f"${r.stop_loss:.1f}" if r.stop_loss else "未指定"
        tp_str = " / ".join(f"${tp:.1f}" for tp in r.take_profits) if r.take_profits else "未指定"
        rr_str = f"{r.risk_reward:.1f}:1" if r.risk_reward else "未计算"
        lines.append(f"| 字段 | 值 |")
        lines.append(f"|------|----|")
        lines.append(f"| 建仓区间 | {entry_str} |")
        lines.append(f"| 止损 | {stop_str} |")
        lines.append(f"| 止盈 | {tp_str} |")
        lines.append(f"| 风险收益比 | {rr_str} |")
        lines.append(f"| 确信度 | {r.conviction} |")
        lines.append(f"| 时间周期 | {r.time_horizon or '未指定'} |")
        lines.append("")

        # Options strategies
        opts = opts_map.get(r.ticker)
        if opts:
            strategies = opts.strategies if isinstance(opts, OptionsRecommendation) else []
            current_price = opts.current_price if isinstance(opts, OptionsRecommendation) else None
            error = opts.error if isinstance(opts, OptionsRecommendation) else ""

            lines.append("### 期权策略建议\n")
            if current_price:
                lines.append(f"当前价格: **${current_price:.2f}**\n")
            if error:
                lines.append(f"⚠️ {error}\n")
            elif not strategies:
                lines.append("*未生成期权策略*\n")
            else:
                for j, s in enumerate(strategies, start=1):
                    expiry = s.legs[0].get("expiry", "") if s.legs else ""
                    lines.append(f"**策略 {j}: {s.name}** (到期 {expiry})\n")
                    lines.append("| 操作 | 类型 | 行权价 | 到期日 | 权利金 |")
                    lines.append("|------|------|--------|--------|--------|")
                    for leg in s.legs:
                        side_zh = "买入" if leg.get("side") == "buy" else "卖出"
                        type_zh = "看涨" if leg.get("type") == "call" else "看跌"
                        lines.append(
                            f"| {side_zh} | {type_zh} | ${leg.get('strike','')} | "
                            f"{leg.get('expiry','')} | ${leg.get('premium','')} |"
                        )
                    lines.append("")
                    parts = []
                    if s.max_loss is not None:
                        try:
                            parts.append(f"最大亏损 **${float(s.max_loss):,.0f}**")
                        except (TypeError, ValueError):
                            parts.append(f"最大亏损 {s.max_loss}")
                    if s.max_gain is not None:
                        try:
                            parts.append(f"最大收益 **${float(s.max_gain):,.0f}**")
                        except (TypeError, ValueError):
                            parts.append(f"最大收益 **{s.max_gain}**")
                    elif "long" in s.name.lower():
                        parts.append("最大收益 **无限**")
                    if s.breakeven is not None:
                        try:
                            parts.append(f"盈亏平衡 **${float(s.breakeven):,.2f}**")
                        except (TypeError, ValueError):
                            parts.append(f"盈亏平衡 {s.breakeven}")
                    lines.append(" | ".join(parts) + "\n")
                    if s.rationale:
                        lines.append(f"> {s.rationale}\n")

        lines.append("---\n")

    # Footer
    lines.append(f"\n*Generated by TradingAgents 3-stage pipeline | {trade_date} | elapsed {elapsed_sec/60:.1f} min*")
    return "\n".join(lines)


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_html_report(
    trade_date: str,
    stage1,
    leaderboard: List[RankedDecision],
    options_recs: list,
    elapsed_sec: float,
) -> str:
    """Generate a self-contained dark-theme HTML report."""
    from .options_strategist import OptionsRecommendation

    opts_map = {}
    for rec in options_recs:
        if isinstance(rec, OptionsRecommendation):
            opts_map[rec.ticker] = rec

    s1_map = {}
    for s in stage1:
        t = s.ticker if hasattr(s, "ticker") else s.get("ticker", "")
        s1_map[t] = s

    def _s1_val(s, key):
        return getattr(s, key, None) if hasattr(s, key) else s.get(key, "")

    # Stage 1 table rows
    s1_sorted = sorted(stage1, key=lambda x: _s1_val(x, "screener_score") or 0, reverse=True)
    top_tickers = {r.ticker for r in leaderboard}
    s1_rows = ""
    for i, s in enumerate(s1_sorted, 1):
        t = _s1_val(s, "ticker")
        sc = _s1_val(s, "screener_score") or 0
        d = _s1_val(s, "screener_direction") or ""
        r = _html_escape(str(_s1_val(s, "screener_rationale") or "")[:90])
        cls = ' class="highlight-row"' if t in top_tickers else ""
        s1_rows += f'<tr{cls}><td>{i}</td><td>{t}</td><td class="num">{sc:.1f}</td><td><span class="badge badge-buy">{d}</span></td><td>{r}</td></tr>\n'

    # Stage 2 table rows
    s2_rows = ""
    for i, r in enumerate(leaderboard, 1):
        e = f"${r.entry[0]:.0f}-${r.entry[1]:.0f}" if r.entry and r.entry[0] != r.entry[1] else (f"${r.entry[0]:.0f}" if r.entry else "-")
        st = f"${r.stop_loss:.0f}" if r.stop_loss else "-"
        tp = ", ".join(f"${t:.0f}" for t in r.take_profits[:2]) or "-"
        rr = f"{r.risk_reward:.1f}:1" if r.risk_reward else "-"
        cls = ' class="highlight-row"' if i == 1 else ""
        s2_rows += f'<tr{cls}><td>{i}</td><td>{r.ticker}</td><td><span class="badge badge-buy">{r.direction}</span></td><td>{r.conviction}</td><td class="num">{e}</td><td class="num" style="color:var(--red)">{st}</td><td class="num" style="color:var(--green)">{tp}</td><td class="num">{rr}</td><td class="num">{r.composite_score:.2f}</td></tr>\n'

    # Ticker detail cards
    cards_html = ""
    for i, r in enumerate(leaderboard, 1):
        s1 = s1_map.get(r.ticker)
        thesis = _html_escape(str(_s1_val(s1, "screener_rationale") or "")) if s1 else ""

        # Extract executive summary from narrative
        narrative = r.final_decision_text or ""
        exec_summary = ""
        for p in narrative.split("\n\n"):
            if "executive summary" in p.lower()[:50] or ("entry" in p.lower()[:80] and "$" in p[:200]):
                exec_summary = p
                for prefix in ["2. **Executive Summary**:", "2. **Executive Summary**: ", "### 2. Executive Summary\n"]:
                    exec_summary = exec_summary.replace(prefix, "")
                break
        exec_summary = _html_escape(exec_summary.strip()[:500])

        # Core thesis paragraph
        core_logic = ""
        for p in narrative.split("\n\n"):
            if "thesis" in p.lower()[:40] or "investment thesis" in p.lower()[:40]:
                core_logic = p
                for prefix in ["3. **Investment Thesis**:", "3. **Investment Thesis**: "]:
                    core_logic = core_logic.replace(prefix, "")
                break
        core_logic = _html_escape(core_logic.strip()[:400])

        e_str = f"${r.entry[0]:.1f} - ${r.entry[1]:.1f}" if r.entry and r.entry[0] != r.entry[1] else (f"${r.entry[0]:.1f}" if r.entry else "-")
        st_str = f"${r.stop_loss:.1f}" if r.stop_loss else "-"
        tp_str = " / ".join(f"${t:.1f}" for t in r.take_profits) if r.take_profits else "-"
        rr_str = f"{r.risk_reward:.1f} : 1" if r.risk_reward else "-"
        price_str = ""
        opts = opts_map.get(r.ticker)
        if opts and opts.current_price:
            price_str = f"${opts.current_price:.2f}"

        # Options HTML
        opts_html = ""
        if opts and opts.strategies:
            for j, s in enumerate(opts.strategies, 1):
                expiry = s.legs[0].get("expiry", "") if s.legs else ""
                legs_rows = ""
                for leg in s.legs:
                    side_zh = "买入" if leg.get("side") == "buy" else "卖出"
                    type_zh = "看涨 Call" if leg.get("type") == "call" else "看跌 Put"
                    legs_rows += f'<tr><td>{side_zh}</td><td>{type_zh}</td><td class="num">${leg.get("strike","")}</td><td>{leg.get("expiry","")}</td><td class="num">${leg.get("premium","")}</td></tr>\n'

                metrics = []
                if s.max_loss is not None:
                    try:
                        metrics.append(f'<span class="opt-metric"><span class="label">最大亏损</span> <span class="val" style="color:var(--red)">${float(s.max_loss):,.0f}</span></span>')
                    except (TypeError, ValueError):
                        metrics.append(f'<span class="opt-metric"><span class="label">最大亏损</span> <span class="val">{s.max_loss}</span></span>')
                if s.max_gain is not None:
                    try:
                        metrics.append(f'<span class="opt-metric"><span class="label">最大收益</span> <span class="val" style="color:var(--green)">${float(s.max_gain):,.0f}</span></span>')
                    except (TypeError, ValueError):
                        metrics.append(f'<span class="opt-metric"><span class="label">最大收益</span> <span class="val" style="color:var(--green)">{s.max_gain}</span></span>')
                elif "long" in s.name.lower():
                    metrics.append('<span class="opt-metric"><span class="label">最大收益</span> <span class="val" style="color:var(--green)">无限</span></span>')
                if s.breakeven is not None:
                    try:
                        metrics.append(f'<span class="opt-metric"><span class="label">盈亏平衡</span> <span class="val">${float(s.breakeven):,.2f}</span></span>')
                    except (TypeError, ValueError):
                        metrics.append(f'<span class="opt-metric"><span class="label">盈亏平衡</span> <span class="val">{s.breakeven}</span></span>')

                rationale_html = f'<div class="opt-rationale">{_html_escape(s.rationale)}</div>' if s.rationale else ""
                opts_html += f'''<div class="opt-strat">
<div class="opt-strat-name">策略 {j}: {_html_escape(s.name)} <span class="expiry">到期 {expiry}</span></div>
<div class="opt-legs"><table><thead><tr><th>操作</th><th>类型</th><th>行权价</th><th>到期日</th><th>权利金</th></tr></thead><tbody>{legs_rows}</tbody></table></div>
<div class="opt-metrics">{"".join(metrics)}</div>
{rationale_html}
</div>\n'''

        cards_html += f'''<div class="ticker-card">
<div class="ticker-header">
<h2><span class="rank-badge">{i}</span>{r.ticker} — {r.direction}</h2>
<span class="price-tag">当前 {price_str} &nbsp; R:R {rr_str}</span>
</div>
<div class="sub-title">决策依据</div>
<div class="basis">
<strong>一句话:</strong> {thesis}<br><br>
{"<strong>执行摘要:</strong> " + exec_summary + "<br><br>" if exec_summary else ""}
{"<strong>核心逻辑:</strong> " + core_logic if core_logic else ""}
</div>
<div class="sub-title">股票交易计划</div>
<div class="plan-grid">
<div class="plan-item"><div class="label">建仓区间</div><div class="value accent">{e_str}</div></div>
<div class="plan-item"><div class="label">止损</div><div class="value red">{st_str}</div></div>
<div class="plan-item"><div class="label">止盈</div><div class="value green">{tp_str}</div></div>
<div class="plan-item"><div class="label">风险收益比</div><div class="value accent">{rr_str}</div></div>
<div class="plan-item"><div class="label">确信度</div><div class="value">{r.conviction}</div></div>
<div class="plan-item"><div class="label">时间周期</div><div class="value">{_html_escape(r.time_horizon or "-")[:40]}</div></div>
</div>
<div class="sub-title">期权策略建议</div>
{opts_html if opts_html else "<p style='color:var(--muted)'>未生成期权策略</p>"}
</div>\n'''

    # Summary comparison grid
    summary_cols = ""
    for r in leaderboard[:6]:
        opts = opts_map.get(r.ticker)
        e = f"${r.entry[0]:.0f}-${r.entry[1]:.0f}" if r.entry and r.entry[0] != r.entry[1] else (f"${r.entry[0]:.0f}" if r.entry else "-")
        st = f"${r.stop_loss:.0f}" if r.stop_loss else "-"
        tp = ", ".join(f"${t:.0f}" for t in r.take_profits[:1]) or "-"
        rr = f"{r.risk_reward:.1f}:1" if r.risk_reward else "-"
        best_opt = "-"
        opt_exp = "-"
        opt_ml = "-"
        opt_mg = "-"
        if opts and opts.strategies:
            # pick the defined-risk spread
            for s in opts.strategies:
                if "spread" in s.name.lower():
                    legs_desc = "/".join(f"${l.get('strike','')}" for l in s.legs)
                    best_opt = f"{s.name} {legs_desc}"
                    opt_exp = s.legs[0].get("expiry", "-") if s.legs else "-"
                    try:
                        opt_ml = f"${float(s.max_loss):,.0f}" if s.max_loss else "-"
                    except (TypeError, ValueError):
                        opt_ml = str(s.max_loss)
                    try:
                        opt_mg = f"${float(s.max_gain):,.0f}" if s.max_gain else "-"
                    except (TypeError, ValueError):
                        opt_mg = str(s.max_gain)
                    break

        summary_cols += f'''<div class="summary-col">
<h3 style="color:var(--accent)">{r.ticker}</h3>
<div class="dir"><span class="badge badge-buy">{r.direction}</span> Score {r.composite_score:.2f}</div>
<table>
<tr><td>Entry</td><td class="num">{e}</td></tr>
<tr><td>Stop</td><td class="num" style="color:var(--red)">{st}</td></tr>
<tr><td>Target</td><td class="num" style="color:var(--green)">{tp}</td></tr>
<tr><td>R:R</td><td class="num">{rr}</td></tr>
<tr><td>Best Option</td><td>{_html_escape(best_opt)[:30]}</td></tr>
<tr><td>Expiry</td><td>{opt_exp}</td></tr>
<tr><td>Max Loss</td><td class="num" style="color:var(--red)">{opt_ml}</td></tr>
<tr><td>Max Gain</td><td class="num" style="color:var(--green)">{opt_mg}</td></tr>
</table>
</div>\n'''

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Trading Agents Report — {trade_date}</title>
<style>
:root{{--bg:#0f1629;--bg2:#1a2340;--card:#1e2a4a;--accent:#00c9a7;--accent2:#00a3ff;--red:#ff4757;--green:#2ed573;--gold:#ffa502;--text:#e8eaf6;--muted:#7b8ca8;--border:#2a3a5c}}
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.6}}.container{{max-width:1100px;margin:0 auto;padding:24px 20px}}
.header{{text-align:center;padding:48px 0 32px;border-bottom:1px solid var(--border);margin-bottom:32px}}.header h1{{font-size:2.2rem;font-weight:700;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}.header .meta{{color:var(--muted);font-size:.9rem;margin-top:8px}}
.badge{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:.75rem;font-weight:600;margin:0 4px}}.badge-buy{{background:rgba(46,213,115,.15);color:var(--green)}}.badge-stage{{background:rgba(0,201,167,.12);color:var(--accent)}}
.section{{margin-bottom:40px}}.section-title{{font-size:1.3rem;font-weight:700;color:var(--accent);margin-bottom:16px;display:flex;align-items:center;gap:8px}}.section-title::before{{content:'';width:4px;height:20px;background:var(--accent);border-radius:2px}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}th{{background:var(--bg2);color:var(--accent);font-weight:600;text-align:left;padding:10px 12px;border-bottom:2px solid var(--border)}}td{{padding:10px 12px;border-bottom:1px solid var(--border)}}tr:hover td{{background:rgba(0,201,167,.04)}}.num{{font-family:'SF Mono','Fira Code',monospace}}.highlight-row td{{background:rgba(0,201,167,.08);font-weight:600}}
.ticker-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:28px;margin-bottom:28px}}.ticker-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:12px}}.ticker-header h2{{font-size:1.5rem}}.ticker-header .price-tag{{font-family:monospace;font-size:1.1rem;color:var(--accent);background:rgba(0,201,167,.1);padding:4px 14px;border-radius:8px}}.rank-badge{{display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:50%;background:var(--accent);color:var(--bg);font-weight:800;font-size:.95rem;margin-right:8px}}
.sub-title{{font-size:1rem;font-weight:700;color:var(--gold);margin:20px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--border)}}
.plan-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}}.plan-item{{background:var(--bg2);padding:12px;border-radius:8px;text-align:center}}.plan-item .label{{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}}.plan-item .value{{font-size:1.05rem;font-weight:700;margin-top:4px;font-family:monospace}}.plan-item .value.green{{color:var(--green)}}.plan-item .value.red{{color:var(--red)}}.plan-item .value.accent{{color:var(--accent)}}
.basis{{background:var(--bg2);border-radius:8px;padding:16px;margin:10px 0;font-size:.88rem;color:var(--text);line-height:1.7}}.basis strong{{color:var(--accent)}}
.opt-strat{{background:var(--bg2);border-radius:8px;padding:16px;margin:10px 0;border-left:3px solid var(--accent2)}}.opt-strat-name{{font-weight:700;font-size:.95rem;color:var(--accent2);margin-bottom:8px}}.opt-strat-name .expiry{{font-weight:400;color:var(--muted);font-size:.85rem;margin-left:6px}}
.opt-legs table{{font-size:.82rem}}.opt-legs th{{background:transparent;color:var(--muted);font-size:.75rem;text-transform:uppercase;border-bottom:1px solid var(--border);padding:6px 8px}}.opt-legs td{{padding:6px 8px;border-bottom:1px solid rgba(42,58,92,.5)}}
.opt-metrics{{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-size:.82rem}}.opt-metric{{background:rgba(0,163,255,.08);padding:4px 10px;border-radius:6px}}.opt-metric .label{{color:var(--muted)}}.opt-metric .val{{font-weight:700;font-family:monospace}}.opt-rationale{{font-size:.82rem;color:var(--muted);margin-top:8px;font-style:italic}}
.summary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:16px 0}}.summary-col{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px;text-align:center}}.summary-col h3{{font-size:1.3rem;margin-bottom:4px}}.summary-col .dir{{font-size:.85rem;margin-bottom:12px}}.summary-col table{{font-size:.82rem;text-align:left}}.summary-col td{{padding:5px 8px;border:none}}.summary-col td:first-child{{color:var(--muted)}}
.footer{{text-align:center;padding:32px 0;color:var(--muted);font-size:.8rem;border-top:1px solid var(--border);margin-top:40px}}
@media(max-width:700px){{.summary-grid{{grid-template-columns:1fr}}.plan-grid{{grid-template-columns:repeat(3,1fr)}}}}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>Trading Agents Screening Report</h1>
<div class="meta">{trade_date} &nbsp;|&nbsp; <span class="badge badge-stage">Stage 1</span> Quick Screen → <span class="badge badge-stage">Stage 2</span> Investment Committee → <span class="badge badge-stage">Stage 3</span> Options Strategist &nbsp;|&nbsp; {len(stage1)} candidates → {len(leaderboard)} deep dives</div>
</div>
<div class="section">
<div class="section-title">Stage 1: Quick Screen</div>
<p style="color:var(--muted);font-size:.85rem;margin-bottom:12px">Market + Fundamentals analysts — {len(stage1)} candidates</p>
<table><thead><tr><th>#</th><th>Ticker</th><th>Score</th><th>Direction</th><th>One-line Thesis</th></tr></thead><tbody>{s1_rows}</tbody></table>
</div>
<div class="section">
<div class="section-title">Stage 2: Investment Committee Leaderboard</div>
<p style="color:var(--muted);font-size:.85rem;margin-bottom:12px">4 analysts → Bull/Bear debate → Trader → Risk debate → Portfolio Manager</p>
<table><thead><tr><th>#</th><th>Ticker</th><th>Dir</th><th>Conv</th><th>Entry</th><th>Stop</th><th>Target</th><th>R:R</th><th>Score</th></tr></thead><tbody>{s2_rows}</tbody></table>
</div>
{cards_html}
<div class="section">
<div class="section-title">总览对比</div>
<div class="summary-grid">{summary_cols}</div>
</div>
<div class="footer">
Generated by TradingAgents 3-Stage Pipeline &nbsp;|&nbsp; {trade_date} &nbsp;|&nbsp; elapsed {elapsed_sec/60:.1f} min<br>
Data: yfinance (OHLCV + fundamentals) + Apewisdom (social trending) + option chains<br><br>
⚠️ 本报告由 AI agents 生成，仅供参考，不构成投资建议。请自行做好尽职调查。
</div>
</div>
</body>
</html>'''
