"""
Trading summary report for Italian equity signals.

Modified Version:
- Actionable Summary: Shows Enter/Exit actions from the last 4 trading days.
- Per-Signal Tables: Shows latest status for active signals.
- Silences internal HOLD logs.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from algoshort.trading_summary import (
    get_multi_symbol_summary,
    print_multi_symbol_summary,
)

# UTF-8 Configuration
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

RESULTS_PATH = Path("./data/results/it/analysis_results.parquet")
DASHBOARD_PATH = Path("./data/results/it/trading_dashboard.xlsx")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Silence internal algoshort logs (Mutes the HOLD LONG/SHORT noise)
logging.getLogger("algoshort").setLevel(logging.WARNING)
logging.getLogger("algoshort.trading_summary").setLevel(logging.WARNING)

_PS_SUFFIXES: dict[str, str] = {
    "equal":    "_shares_equal",
    "constant": "_shares_constant",
    "concave":  "_shares_concave",
    "convex":   "_shares_convex",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(path: Path) -> dict[str, pd.DataFrame]:
    log.info("Loading results from %s", path)
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    data_dict = {symbol: grp.copy() for symbol, grp in df.groupby("symbol")}
    log.info("Loaded %d symbols.", len(data_dict))
    return data_dict

def detect_signal_columns(df: pd.DataFrame) -> list[str]:
    cols = set(df.columns)
    return sorted(c for c in cols if f"{c}_stop_loss" in cols)

def build_position_cols(signal: str, available: set[str]) -> dict[str, str] | None:
    mapping = {
        label: f"{signal}{suffix}"
        for label, suffix in _PS_SUFFIXES.items()
        if f"{signal}{suffix}" in available
    }
    return mapping or None

def summaries_to_dataframe(summaries: list[dict], signal: str) -> pd.DataFrame:
    rows = []
    for s in summaries:
        if "error" in s:
            rows.append({"signal": signal, "ticker": s["ticker"], "error": s["error"]})
            continue
        row = {
            "signal":           signal,
            "ticker":           s["ticker"],
            "last_date":        pd.to_datetime(s["last_date"]),
            "price":            s["current_price"],
            "position":         s["position_direction"],
            "action":           s["trade_action"],
            "signal_changed":   s["signal_changed"],
            "stop_loss":        s["stop_loss"],
            "risk_pct":         s["risk_pct"],
        }
        for label, shares in s.get("position_sizes", {}).items():
            row[f"shares_{label}"] = shares
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_dict = load_results(RESULTS_PATH)
    if not data_dict:
        sys.exit("Error: No data found.")

    first_df = next(iter(data_dict.values()))
    signal_columns = detect_signal_columns(first_df)
    available_cols = set(first_df.columns)

    if not signal_columns:
        stop_loss_cols = sorted(c for c in available_cols if c.endswith("_stop_loss"))
        all_cols = sorted(available_cols)
        log.error(
            "No signal columns detected. "
            "detect_signal_columns requires each signal column to have a matching "
            "'<signal>_stop_loss' column, but none were found.\n"
            "  stop_loss columns found: %s\n"
            "  all columns: %s",
            stop_loss_cols,
            all_cols,
        )
        sys.exit(
            "Error: No signal columns found in the results parquet. "
            "Re-run analyze_stock.py to regenerate results with stop-loss columns."
        )

    log.info("Detected %d signal columns: %s", len(signal_columns), signal_columns)

    all_summaries: dict[str, list[dict]] = {}
    all_dashboards: list[pd.DataFrame] = []

    for signal in signal_columns:
        position_cols = build_position_cols(signal, available_cols)
        summaries = get_multi_symbol_summary(
            data_dict=data_dict,
            signal_col=signal,
            stop_loss_col=f"{signal}_stop_loss",
            close_col="close",
            position_cols=position_cols,
            lookback=5, 
        )
        all_summaries[signal] = summaries
        all_dashboards.append(summaries_to_dataframe(summaries, signal))

    dashboard = pd.concat(all_dashboards, ignore_index=True)
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    dashboard.to_excel(DASHBOARD_PATH, index=False)

    # --- DATE LOGIC ---
    # Get all unique dates and identify the cutoff for the last 4 trading days
    unique_dates = sorted(dashboard["last_date"].unique(), reverse=True)
    latest_date = unique_dates[0]
    # Handle cases where there might be fewer than 4 days of data
    cutoff_index = min(3, len(unique_dates) - 1)
    four_days_ago = unique_dates[cutoff_index]

    # -----------------------------------------------------------------------
    # 1. ACTIONABLE SUMMARY (Last 4 Trading Days - Enter/Exit ONLY)
    # -----------------------------------------------------------------------
    is_recent = dashboard["last_date"] >= four_days_ago
    is_action = dashboard["action"].str.contains("Enter|Exit", case=False, na=False)
    
    # Sort by date descending so newest actions appear at the top
    actionable = dashboard[is_recent & is_action].sort_values(by="last_date", ascending=False).copy()

    print(f"\n{'=' * 95}")
    print(f"  ACTIONABLE TRADES: {four_days_ago.date()} to {latest_date.date()} (Last 4 Days)")
    print(f"{'=' * 95}")
    
    if actionable.empty:
        print(f"  No Enter/Exit signals detected in the last 4 trading days.")
    else:
        # We include 'last_date' here because the table now covers multiple days
        display_cols = ["last_date", "signal", "ticker", "price", "action", "stop_loss", "risk_pct"]
        print(actionable[display_cols].to_string(index=False))

    # -----------------------------------------------------------------------
    # 2. PER-SIGNAL TABLES (Latest Day ONLY for active signals)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 95}")
    print(f"  LATEST STATUS FOR ACTIVE SIGNALS ({latest_date.date()})")
    print(f"{'=' * 95}")

    for signal in signal_columns:
        # Filter for tickers that have an action specifically on the latest day
        filtered_summaries = [
            s for s in all_summaries[signal] 
            if pd.to_datetime(s.get("last_date")) == latest_date and 
            any(act in str(s.get("trade_action", "")) for act in ["Enter", "Exit"])
        ]

        if filtered_summaries:
            print(f"\n>>> [ {signal} ]")
            print("-" * 35)
            # detailed=False keeps the final output concise
            print_multi_symbol_summary(filtered_summaries, detailed=False)

    log.info("Process finished. Check %s for the full 5-day history and position sizes.", DASHBOARD_PATH)
    
    # """
# Trading summary report for Italian equity signals.

# Reads analysis_results.parquet (written by analyze_stock.py) and produces,
# for every signal column, a formatted console summary and a flat CSV dashboard
# via algoshort.trading_summary.
# """

# import logging
# import sys
# from pathlib import Path

# import pandas as pd
# from algoshort.trading_summary import (
#     get_multi_symbol_summary,
#     print_multi_symbol_summary,
# )

# # algoshort prints unicode characters that cp1252 (Windows default) cannot
# # encode. Reconfigure stdout/stderr to UTF-8 before any algoshort call.
# sys.stdout.reconfigure(encoding="utf-8")
# sys.stderr.reconfigure(encoding="utf-8")

# RESULTS_PATH = Path("./data/results/it/analysis_results.parquet")
# DASHBOARD_PATH = Path("./data/results/it/trading_dashboard.xlsx")

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
# )
# log = logging.getLogger(__name__)

# # Position-sizing column suffixes written by algoshort.position_sizing.
# _PS_SUFFIXES: dict[str, str] = {
#     "equal":    "_shares_equal",
#     "constant": "_shares_constant",
#     "concave":  "_shares_concave",
#     "convex":   "_shares_convex",
# }


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def load_results(path: Path) -> dict[str, pd.DataFrame]:
#     """Load the combined parquet and split it into per-symbol DataFrames."""
#     log.info("Loading results from %s", path)
#     df = pd.read_parquet(path)
#     data_dict = {symbol: grp.copy() for symbol, grp in df.groupby("symbol")}
#     log.info("Loaded %d symbols: %s", len(data_dict), sorted(data_dict))
#     return data_dict


# def detect_signal_columns(df: pd.DataFrame) -> list[str]:
#     """
#     Return signal column names that have a matching <signal>_stop_loss column.
#     This is the reliable marker that a column is a primary trading signal (as
#     opposed to a combined/grid-search signal or a derived metric).
#     """
#     cols = set(df.columns)
#     return sorted(c for c in cols if f"{c}_stop_loss" in cols)


# def build_position_cols(signal: str, available: set[str]) -> dict[str, str] | None:
#     """
#     Build the position_cols mapping for a given signal, filtering to only
#     columns that actually exist in the DataFrame.  Returns None when none
#     of the sizing columns are present (so the summary module skips them).
#     """
#     mapping = {
#         label: f"{signal}{suffix}"
#         for label, suffix in _PS_SUFFIXES.items()
#         if f"{signal}{suffix}" in available
#     }
#     return mapping or None


# def summaries_to_dataframe(
#     summaries: list[dict], signal: str
# ) -> pd.DataFrame:
#     """Flatten get_multi_symbol_summary output into a single DataFrame row per symbol."""
#     rows = []
#     for s in summaries:
#         if "error" in s:
#             rows.append({"signal": signal, "ticker": s["ticker"], "error": s["error"]})
#             continue
#         row = {
#             "signal":           signal,
#             "ticker":           s["ticker"],
#             "last_date":        s["last_date"],
#             "price":            s["current_price"],
#             "position":         s["position_direction"],
#             "action":           s["trade_action"],
#             "signal_changed":   s["signal_changed"],
#             "stop_loss":        s["stop_loss"],
#             "risk_pct":         s["risk_pct"],
#         }
#         for label, shares in s.get("position_sizes", {}).items():
#             row[f"shares_{label}"] = shares
#         rows.append(row)
#     return pd.DataFrame(rows)


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     data_dict = load_results(RESULTS_PATH)

#     first_df = next(iter(data_dict.values()))
#     signal_columns = detect_signal_columns(first_df)
#     available_cols = set(first_df.columns)
#     log.info("Detected %d signal columns: %s", len(signal_columns), signal_columns)

#     # --- Pass 1: collect all summaries without printing ---
#     all_summaries: dict[str, list[dict]] = {}
#     all_dashboards: list[pd.DataFrame] = []

#     for signal in signal_columns:
#         position_cols = build_position_cols(signal, available_cols)
#         summaries = get_multi_symbol_summary(
#             data_dict=data_dict,
#             signal_col=signal,
#             stop_loss_col=f"{signal}_stop_loss",
#             close_col="close",
#             position_cols=position_cols,
#             lookback=5,
#         )
#         all_summaries[signal] = summaries
#         all_dashboards.append(summaries_to_dataframe(summaries, signal))

#     # -----------------------------------------------------------------------
#     # Consolidated CSV dashboard
#     # -----------------------------------------------------------------------
#     dashboard = pd.concat(all_dashboards, ignore_index=True)
#     DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
#     dashboard.to_excel(DASHBOARD_PATH, index=False)
#     log.info(
#         "Dashboard saved to %s (%d rows, %d signals × %d symbols)",
#         DASHBOARD_PATH,
#         len(dashboard),
#         len(signal_columns),
#         len(data_dict),
#     )

#     # -----------------------------------------------------------------------
#     # 1. Actionable summary: stocks with signal changes on the last bar
#     # -----------------------------------------------------------------------
#     actionable = dashboard[dashboard["signal_changed"] == True].copy()  # noqa: E712
#     print(f"\n{'=' * 70}")
#     print("  ACTIONABLE SIGNALS (signal changed on last bar)")
#     print(f"{'=' * 70}")
#     if actionable.empty:
#         print("  No signal changes on the latest bar.")
#     else:
#         share_cols = [c for c in actionable.columns if c.startswith("shares_")]
#         display_cols = (
#             ["signal", "ticker", "last_date", "price", "action", "stop_loss", "risk_pct"]
#             + share_cols
#         )
#         display_cols = [c for c in display_cols if c in actionable.columns]
#         print(actionable[display_cols].to_string(index=False))

#     # -----------------------------------------------------------------------
#     # 2. Per-signal summary tables with position sizes
#     # -----------------------------------------------------------------------
#     for signal in signal_columns:
#         print(f"\n{'=' * 70}")
#         print(f"  SIGNAL: {signal}")
#         print(f"{'=' * 70}")
#         print_multi_symbol_summary(all_summaries[signal], detailed=False)

#         sig_df = dashboard[dashboard["signal"] == signal]
#         share_cols = [c for c in sig_df.columns if c.startswith("shares_")]
#         if share_cols:
#             print(f"\n  Position Sizes:")
#             size_view = sig_df[["ticker", "position", "action"] + share_cols].copy()
#             print(size_view.to_string(index=False))
