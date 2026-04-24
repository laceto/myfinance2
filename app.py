"""
app.py — Streamlit UI for the Italian Equities Trader assistants.

Wraps both ask_bo_trader (range breakout) and ask_ma_trader (MA crossover)
so users can run either or both analyses from a single web interface.

Usage:
    streamlit run app.py

Environment:
    OPENAI_API_KEY must be set (loaded from .env via python-dotenv).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Italian Equities Trader",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = "data/results/it/analysis_results.parquet"

# ---------------------------------------------------------------------------
# Data loading — cached so the parquet is read once per session
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading parquet data…")
def _load_parquet(path: str) -> pd.DataFrame:
    """Load analysis_results.parquet and return the full DataFrame."""
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _signal_label(sig: int) -> str:
    """Human-readable label for a +1 / 0 / -1 signal integer."""
    if sig == 1:
        return "LONG (+1)"
    if sig == -1:
        return "SHORT (-1)"
    return "FLAT (0)"


def _bool_label(value: bool | None, true_str: str = "Yes", false_str: str = "No") -> str:
    if value is None:
        return "n/a"
    return true_str if value else false_str


def _confirmed_label(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "CONFIRMED" if value else "WEAK"


# ---------------------------------------------------------------------------
# BO Breakout render
# ---------------------------------------------------------------------------


def _render_bo(analysis, ticker: str) -> None:
    """Render a TraderAnalysis (BO breakout) result as Streamlit widgets."""
    from ask_bo_trader import TraderAnalysis  # noqa: F401 — import for type checking

    a = analysis

    # --- Overview ---
    st.subheader(f"BO Breakout — {ticker}")
    st.markdown(f"*{a.description}*")

    regime_label = {1: "Bullish", 0: "Sideways", -1: "Bearish"}.get(a.regime, str(a.regime))
    c1, c2, c3 = st.columns(3)
    c1.metric("Regime", regime_label)
    c2.metric("Confluence", a.confluence.replace("_", " ").title())
    c3.metric("Vol Trend", f"{a.vol_trend:.2f}x")

    st.divider()

    # --- Timeframes ---
    st.markdown("**Signal Timeframes**")
    tf_items = [
        ("20d", "rhi_20 / rlo_20", a.short_term),
        ("50d", "rhi_50 / rlo_50", a.medium_term),
        ("150d", "rhi_150 / rlo_150", a.long_term),
    ]
    cols = st.columns(3)
    for (window, level_label, tf), col in zip(tf_items, cols):
        with col:
            flip_tag = "  [FLIP]" if tf.fresh_flip else ""
            sig_info = f"{_signal_label(tf.signal)}{flip_tag}  age={tf.signal_age}d"
            st.markdown(f"**{window}** — {sig_info}")
            c_res, c_sup = col.columns(2)
            c_res.metric(
                f"Resistance ({window.replace('d', '')})",
                f"{tf.resistance:.4f}",
                delta=f"{tf.dist_to_resistance_pct:+.2f}% from close",
                delta_color="inverse",
            )
            c_sup.metric(
                f"Support ({window.replace('d', '')})",
                f"{tf.support:.4f}",
                delta=f"{tf.dist_to_support_pct:+.2f}% from close",
                delta_color="inverse",
            )
            if tf.momentum_pct is not None:
                col.caption(f"Momentum: {tf.momentum_pct:+.2f}%")
            col.caption(tf.commentary)

    # rh4 / rl4 — structural swing anchors relevant to all timeframes
    r = a.risk
    st.markdown("Structural levels:")
    c1, c2 = st.columns(2)
    c1.metric("rh4 — peak resistance", f"{r.peak_resistance:.4f}")
    c2.metric("rl4 — major floor", f"{r.major_floor:.4f}")

    st.divider()

    # --- Turtle ---
    st.markdown("**Turtle Signal (rtt_5020)**")
    tt = a.turtle
    c1, c2 = st.columns(2)
    c1.metric("Signal", _signal_label(tt.signal))
    c2.metric("Aligns with rbo_20", _bool_label(tt.aligned_with_rbo_20))
    st.caption(tt.commentary)

    st.divider()

    # --- Range quality ---
    st.markdown("**Range Quality**")
    if a.range_quality is not None:
        rq = a.range_quality
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Res touches", rq.n_resistance_touches)
        c2.metric("Sup touches", rq.n_support_touches)
        c3.metric("Consol bars", rq.consolidation_bars)
        c4.metric("Band width", f"{rq.band_width_pct:.2f}%")
        st.caption(
            f"Sideways: {rq.is_sideways}  |  "
            f"Slope: {rq.slope_pct_per_day:+.4f}%/day"
        )
        st.caption(rq.commentary)
    else:
        st.info("No consolidation window — ticker is currently in an active trend.")

    st.divider()

    # --- Volatility compression ---
    st.markdown("**Volatility Compression**")
    if a.volatility_compression is not None:
        vc = a.volatility_compression
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Band width", f"{vc.band_width_pct:.2f}%")
        c2.metric("BW slope", f"{vc.band_width_slope:+.6f}/bar")
        c3.metric("BW rank", f"{vc.band_width_pct_rank:.1f}%ile")
        c4.metric("Compressed", "YES" if vc.is_compressed else "No")
        st.caption(vc.commentary)
    else:
        st.info("Insufficient history for compression estimate.")

    st.divider()

    # --- Volume quality ---
    st.markdown("**Volume Quality**")
    if a.volume_quality is not None:
        vq = a.volume_quality
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quiet", _bool_label(vq.is_quiet))
        c2.metric("Declining", _bool_label(vq.is_declining))
        c3.metric("Mean vol", f"{vq.vol_trend_mean:.3f}")
        c4.metric("Breakout vol", _confirmed_label(vq.breakout_confirmed))
        st.caption(vq.commentary)
    else:
        st.info("No volume history available.")

    st.divider()

    # --- Risk ---
    # rh4 / rl4 already shown in the Signal Timeframes structural levels row above.
    st.markdown("**Risk Levels**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Long**")
        st.metric("Stop (rlo_20)", r.long_stop)
        st.metric("Structural stop (rlo_150)", r.long_structural_stop)
    with c2:
        st.markdown("**Short**")
        st.metric("Stop (rhi_20)", r.short_stop)
        st.metric("Structural stop (rhi_150)", r.short_structural_stop)

    st.divider()

    # --- Verdict ---
    st.markdown("**Verdict**")
    st.success(a.verdict)


# ---------------------------------------------------------------------------
# MA Crossover render
# ---------------------------------------------------------------------------


def _render_ma(analysis, ticker: str) -> None:
    """Render a MATraderAnalysis (MA crossover) result as Streamlit widgets."""
    from ask_ma_trader import MATraderAnalysis  # noqa: F401 — import for type checking

    a = analysis

    # --- Overview ---
    st.subheader(f"MA Crossover — {ticker}")
    st.markdown(f"*{a.description}*")

    regime_label = {1: "Bullish", 0: "Sideways", -1: "Bearish"}.get(a.regime, str(a.regime))
    c1, c2 = st.columns(2)
    c1.metric("Regime", regime_label)
    c2.metric("Confluence", a.confluence.replace("_", " ").title())

    st.divider()

    # --- EMA/SMA timeframes ---
    st.markdown("**Signal Timeframes**")
    tf_items = [
        ("Short (50/100)", a.short_term),
        ("Medium (100/150)", a.medium_term),
    ]
    cols = st.columns(2)
    for (label, tf), col in zip(tf_items, cols):
        with col:
            flip_tag = "  [FLIP]" if tf.fresh_flip else ""
            agree_tag = "EMA/SMA agree" if tf.ema_sma_agree else "EMA/SMA DISAGREE"
            st.metric(label, f"{_signal_label(tf.ema_signal)}{flip_tag}", delta=f"Age: {tf.signal_age}d")
            st.caption(f"SMA signal: {_signal_label(tf.sma_signal)}  |  {agree_tag}")
            st.caption(
                f"Fast dist: {tf.dist_fast_ma_pct:+.2f}%  "
                f"Slow dist: {tf.dist_slow_ma_pct:+.2f}%"
            )
            st.caption(tf.commentary)

    st.divider()

    # --- Triple confluence ---
    st.markdown("**Triple Confluence (50/100/150)**")
    tc = a.triple_confluence
    flip_tag = "  [FLIP]" if tc.fresh_flip else ""
    agree_tag = "EMA+SMA agree" if tc.agree else "EMA/SMA DISAGREE"
    c1, c2, c3 = st.columns(3)
    c1.metric("EMA signal", f"{_signal_label(tc.ema_signal)}{flip_tag}", delta=f"Age: {tc.signal_age}d")
    c2.metric("SMA signal", _signal_label(tc.sma_signal))
    c3.metric("Agreement", agree_tag)
    st.caption(tc.commentary)

    st.divider()

    # --- Trend strength ---
    st.markdown("**Trend Strength**")
    if a.trend_strength is not None:
        ts = a.trend_strength
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RSI (14)", f"{ts.rsi:.1f}")
        c2.metric("ADX", f"{ts.adx:.1f}")
        c3.metric("ADX slope", f"{ts.adx_slope:+.4f}/bar", delta=f"r²={ts.adx_slope_r2:.2f}")
        c4.metric("Trending", "YES" if ts.is_trending else "No")
        c1, c2 = st.columns(2)
        c1.metric("MA gap", f"{ts.ma_gap_pct:+.2f}%")
        c2.metric("Gap slope", f"{ts.ma_gap_slope:+.4f}/bar", delta=f"r²={ts.ma_gap_slope_r2:.2f}")
        st.caption(ts.commentary)
    else:
        st.info("Insufficient history for ADX/RSI computation.")

    st.divider()

    # --- Volume quality ---
    st.markdown("**Volume Quality**")
    vq = a.volume_quality
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vol trend", f"{vq.vol_trend:.2f}x")
    c2.metric("Vol on crossover", str(vq.vol_on_crossover) if vq.vol_on_crossover is not None else "n/a")
    c3.metric("Confirmed", _confirmed_label(vq.is_confirmed))
    c4.metric("Sustained", _bool_label(vq.is_sustained, "YES", "No"))
    st.caption(vq.commentary)

    st.divider()

    # --- Risk ---
    st.markdown("**Risk Levels**")
    r = a.risk
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Long**")
        st.metric("Stop (ATR)", r.long_stop)
        st.metric("Structural stop (rl4)", r.long_structural_stop)
    with c2:
        st.markdown("**Short**")
        st.metric("Stop (ATR)", r.short_stop)
        st.metric("Structural stop (rh4)", r.short_structural_stop)
    c1, c2 = st.columns(2)
    c1.metric("Peak resistance (rh4)", r.peak_resistance)
    c2.metric("Major floor (rl4)", r.major_floor)

    st.divider()

    # --- Verdict ---
    st.markdown("**Verdict**")
    st.success(a.verdict)


# ---------------------------------------------------------------------------
# Analysis runners
# ---------------------------------------------------------------------------


def _run_bo(df_ticker: pd.DataFrame, ticker: str, question: str | None) -> tuple[dict, object]:
    """Build the BO snapshot and call the AI. Returns (snapshot, TraderAnalysis)."""
    from ta.breakout.bo_snapshot import build_snapshot
    from ask_bo_trader import ask_bo_trader

    snapshot = {"ticker": ticker, **build_snapshot(df_ticker)}
    analysis = ask_bo_trader(snapshot, ticker=ticker, question=question)
    return snapshot, analysis


def _run_ma(df_ticker: pd.DataFrame, ticker: str, question: str | None) -> tuple[dict, object]:
    """Build the MA snapshot and call the AI. Returns (snapshot, MATraderAnalysis)."""
    from ta.ma.ma_snapshot import build_snapshot
    from ask_ma_trader import ask_ma_trader

    snapshot = {"ticker": ticker, **build_snapshot(df_ticker)}
    analysis = ask_ma_trader(snapshot, ticker=ticker, question=question)
    return snapshot, analysis


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar() -> dict:
    """Render the sidebar and return a dict of user inputs."""
    with st.sidebar:
        st.title("Italian Equities Trader")
        st.caption("Borsa Italiana — AI-powered technical analysis")
        st.divider()

        ticker = st.text_input(
            "Ticker symbol",
            value="A2A.MI",
            help="Yahoo Finance ticker, e.g. A2A.MI, ENI.MI, ENEL.MI",
        ).strip().upper()

        mode = st.selectbox(
            "Analysis mode",
            options=["BO Breakout", "MA Crossover", "Both"],
            index=0,
            help=(
                "BO Breakout: range breakout signals (rbo_*, rhi_*, rlo_*).\n\n"
                "MA Crossover: moving average crossover signals (rema_*, rsma_*).\n\n"
                "Both: run both analyses in sequence."
            ),
        )

        question = st.text_input(
            "Follow-up question (optional)",
            value="",
            placeholder="e.g. Should I add to this position?",
            help="Appended to the snapshot before sending to the model.",
        ).strip() or None

        st.divider()

        with st.expander("Advanced"):
            data_path = st.text_input(
                "Parquet path",
                value=DEFAULT_DATA_PATH,
                help="Path to analysis_results.parquet",
            )

        run = st.button("Run Analysis", type="primary", use_container_width=True)

    return {
        "ticker": ticker,
        "mode": mode,
        "question": question,
        "data_path": data_path,
        "run": run,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    inputs = _sidebar()

    st.title("Italian Equities Trader")
    st.caption(
        "Powered by OpenAI structured output. "
        "Select a ticker and analysis mode in the sidebar, then click **Run Analysis**."
    )

    # --- Pre-flight checks ---
    data_path = Path(inputs["data_path"])
    if not data_path.exists():
        st.error(f"Parquet file not found: `{data_path}`. Check the path in Advanced settings.")
        return

    import os

    if not os.environ.get("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY is not set. Add it to your `.env` file or set it as an "
            "environment variable before launching the app."
        )
        return

    # --- Trigger on button click ---
    if inputs["run"]:
        ticker = inputs["ticker"]
        mode = inputs["mode"]
        question = inputs["question"]

        if not ticker:
            st.warning("Enter a ticker symbol in the sidebar.")
            return

        # Load data (cached after first load)
        df = _load_parquet(str(data_path))
        df_ticker = df[df["symbol"] == ticker].copy()

        if df_ticker.empty:
            available = df["symbol"].unique()[:20].tolist()
            st.error(
                f"Ticker **{ticker}** not found in the parquet.\n\n"
                f"Sample of available tickers: `{available}`"
            )
            return

        st.session_state["last_ticker"] = ticker
        st.session_state["last_mode"] = mode

        # --- Run selected analysis ---
        if mode in ("BO Breakout", "Both"):
            with st.spinner(f"Running BO breakout analysis for {ticker}…"):
                try:
                    bo_snapshot, bo_analysis = _run_bo(df_ticker, ticker, question)
                    st.session_state["bo_snapshot"] = bo_snapshot
                    st.session_state["bo_analysis"] = bo_analysis
                    st.session_state["bo_error"] = None
                except Exception as exc:
                    st.session_state["bo_error"] = str(exc)
                    st.session_state["bo_analysis"] = None

        if mode in ("MA Crossover", "Both"):
            with st.spinner(f"Running MA crossover analysis for {ticker}…"):
                try:
                    ma_snapshot, ma_analysis = _run_ma(df_ticker, ticker, question)
                    st.session_state["ma_snapshot"] = ma_snapshot
                    st.session_state["ma_analysis"] = ma_analysis
                    st.session_state["ma_error"] = None
                except Exception as exc:
                    st.session_state["ma_error"] = str(exc)
                    st.session_state["ma_analysis"] = None

    # --- Render cached results (persist across reruns) ---
    mode = st.session_state.get("last_mode", inputs["mode"])
    ticker = st.session_state.get("last_ticker", inputs["ticker"])

    show_bo = mode in ("BO Breakout", "Both") and "bo_analysis" in st.session_state
    show_ma = mode in ("MA Crossover", "Both") and "ma_analysis" in st.session_state

    if not show_bo and not show_ma:
        st.info("Configure the sidebar and click **Run Analysis** to start.")
        return

    # Use tabs when both analyses are present
    if show_bo and show_ma:
        tab_bo, tab_ma = st.tabs(["BO Breakout", "MA Crossover"])
    elif show_bo:
        tab_bo = st.container()
        tab_ma = None
    else:
        tab_bo = None
        tab_ma = st.container()

    if show_bo:
        with tab_bo:
            err = st.session_state.get("bo_error")
            if err:
                st.error(f"BO analysis failed: {err}")
            else:
                bo_snap = st.session_state["bo_snapshot"]
                with st.expander("Raw snapshot (JSON)", expanded=False):
                    st.json(bo_snap)
                _render_bo(st.session_state["bo_analysis"], ticker)

    if show_ma:
        with tab_ma:
            err = st.session_state.get("ma_error")
            if err:
                st.error(f"MA analysis failed: {err}")
            else:
                ma_snap = st.session_state["ma_snapshot"]
                with st.expander("Raw snapshot (JSON)", expanded=False):
                    st.json(ma_snap)
                _render_ma(st.session_state["ma_analysis"], ticker)


if __name__ == "__main__":
    main()
