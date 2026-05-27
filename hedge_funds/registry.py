"""
hedge_funds/registry.py — Canonical definition of all 100 hedge funds.

Each entry maps a fund_id (F001–F100) to a FundConfig that fully describes
the fund: name, strategy class, universe, parameters, and trading mode.

All funds default to PAPER mode. Promote to LIVE by updating mode=TradingMode.LIVE
after validating ≥30 days of paper performance.

Fund catalogue:
    F001–F020  Trend Following      (MACrossoverFund variants)
    F021–F035  Breakout             (RangeBreakoutFund variants)
    F036–F050  Mean Reversion       (BollingerMRFund variants)
    F051–F060  Cross-Sect. Momentum (CrossSectionalMomentumFund)
    F061–F070  Volatility           (VolTargetingFund / VIXRegimeFund  [stubs])
    F071–F080  Factor               (FactorFund                        [stubs])
    F081–F085  Crypto               (CryptoMomentumFund)
    F086–F095  AI / LangGraph       (LangGraphFund)
    F096–F100  Options              (OptionsFund                       [stubs])
"""

from __future__ import annotations

from .config import AssetClass, FundConfig, StrategyCategory, TradingMode

# ── Pre-defined universes ──────────────────────────────────────────────────────

_US_LARGE_CAP = ["SPY", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK.B", "JPM"]
_US_TECH = ["QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMD", "AVGO", "ORCL"]
_US_MID_CAP = ["IJH", "MDY", "VXF", "IWR", "VO"]
_SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLP", "XLY", "XLRE", "XLU", "XLB"]
_COMMODITY_ETFS = ["GLD", "SLV", "USO", "UNG", "DBA", "PDBC"]
_DIVERSIFIED = ["SPY", "QQQ", "IWM", "GLD", "TLT", "EEM", "VNQ", "HYG", "LQD", "USO"]
_CRYPTO = ["BTC/USD", "ETH/USD"]
_CRYPTO_FULL = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD"]
_OPTIONS_UNIVERSE = ["SPY", "AAPL", "MSFT", "QQQ"]

# ── Factory helpers ────────────────────────────────────────────────────────────

def _trend(fid, name, desc, universe, fast, slow, ma_type="sma") -> FundConfig:
    return FundConfig(
        fund_id=fid, name=name, description=desc,
        category=StrategyCategory.TREND_FOLLOWING, asset_class=AssetClass.EQUITY,
        strategy_class="hedge_funds.strategies.trend.ma_crossover.MACrossoverFund",
        universe=universe, params={"fast_window": fast, "slow_window": slow, "ma_type": ma_type},
    )


def _breakout(fid, name, desc, universe, window, long_only=False) -> FundConfig:
    return FundConfig(
        fund_id=fid, name=name, description=desc,
        category=StrategyCategory.BREAKOUT, asset_class=AssetClass.EQUITY,
        strategy_class="hedge_funds.strategies.breakout.range_breakout.RangeBreakoutFund",
        universe=universe, params={"bo_window": window, "long_only": long_only},
    )


def _mr(fid, name, desc, universe, window=20, std=2.0, long_only=True) -> FundConfig:
    return FundConfig(
        fund_id=fid, name=name, description=desc,
        category=StrategyCategory.MEAN_REVERSION, asset_class=AssetClass.EQUITY,
        strategy_class="hedge_funds.strategies.mean_reversion.bollinger_mr.BollingerMRFund",
        universe=universe, params={"bb_window": window, "bb_std": std, "long_only": long_only},
    )


def _momentum(fid, name, desc, universe, formation=252, skip=21, top_n=3, bottom_n=0, long_only=True) -> FundConfig:
    return FundConfig(
        fund_id=fid, name=name, description=desc,
        category=StrategyCategory.MOMENTUM, asset_class=AssetClass.EQUITY,
        strategy_class="hedge_funds.strategies.momentum.cross_sectional.CrossSectionalMomentumFund",
        universe=universe,
        params={
            "formation_months": formation, "skip_months": skip,
            "top_n": top_n, "bottom_n": bottom_n, "long_only": long_only,
        },
    )


def _vol(fid, name, desc, universe, cls="VolTargetingFund", **params) -> FundConfig:
    return FundConfig(
        fund_id=fid, name=name, description=desc,
        category=StrategyCategory.VOLATILITY, asset_class=AssetClass.EQUITY,
        strategy_class=f"hedge_funds.strategies.volatility.vol_targeting.{cls}",
        universe=universe, params=dict(params),
    )


def _factor(fid, name, desc, universe, **params) -> FundConfig:
    return FundConfig(
        fund_id=fid, name=name, description=desc,
        category=StrategyCategory.FACTOR, asset_class=AssetClass.EQUITY,
        strategy_class="hedge_funds.strategies.factor.multi_factor.FactorFund",
        universe=universe, params=dict(params),
    )


def _crypto(fid, name, desc, universe, fast, slow, ma_type="ema") -> FundConfig:
    return FundConfig(
        fund_id=fid, name=name, description=desc,
        category=StrategyCategory.CRYPTO, asset_class=AssetClass.CRYPTO,
        strategy_class="hedge_funds.strategies.crypto.btc_momentum.CryptoMomentumFund",
        universe=universe, params={"fast_window": fast, "slow_window": slow, "ma_type": ma_type},
    )


def _ai(fid, name, desc, universe, benchmark="SPY") -> FundConfig:
    return FundConfig(
        fund_id=fid, name=name, description=desc,
        category=StrategyCategory.AI, asset_class=AssetClass.EQUITY,
        strategy_class="hedge_funds.strategies.ai.langgraph_fund.LangGraphFund",
        universe=universe, params={"benchmark": benchmark},
    )


def _options(fid, name, desc, universe) -> FundConfig:
    return FundConfig(
        fund_id=fid, name=name, description=desc,
        category=StrategyCategory.OPTIONS, asset_class=AssetClass.OPTIONS,
        strategy_class="hedge_funds.strategies.options.covered_call.OptionsFund",
        universe=universe, params={},
    )


# ── Registry ───────────────────────────────────────────────────────────────────

FUND_REGISTRY: dict[str, FundConfig] = {

    # ── F001–F020: Trend Following ─────────────────────────────────────────────
    "F001": _trend("F001", "Golden Cross Large Cap", "SMA 50/200 golden cross on US large-cap equities.",
                   _US_LARGE_CAP, 50, 200, "sma"),
    "F002": _trend("F002", "EMA Momentum Mid Cap", "EMA 10/50 crossover on US mid-cap basket.",
                   _US_MID_CAP, 10, 50, "ema"),
    "F003": _trend("F003", "Triple MA Tech", "SMA 20/50/200 alignment on US tech leaders (uses fast/slow pair).",
                   _US_TECH, 20, 100, "sma"),
    "F004": _trend("F004", "MACD Proxy Diversified", "EMA 12/26 crossover (MACD signal proxy) on diversified ETFs.",
                   _DIVERSIFIED, 12, 26, "ema"),
    "F005": _trend("F005", "Donchian Commodity Trend", "SMA 20/60 trend on commodity ETFs.",
                   _COMMODITY_ETFS, 20, 60, "sma"),
    "F006": _trend("F006", "Turtle Classic 20/55", "Turtle system short/long windows on sector ETFs.",
                   _SECTOR_ETFS, 20, 55, "sma"),
    "F007": _trend("F007", "Supertrend Proxy Tech", "EMA 7/14 fast momentum on US tech (supertrend proxy).",
                   _US_TECH, 7, 14, "ema"),
    "F008": _trend("F008", "Ichimoku Proxy SPY", "SMA 9/26 (Ichimoku conversion/base proxy) on large caps.",
                   _US_LARGE_CAP, 9, 26, "sma"),
    "F009": _trend("F009", "Linear Reg Slope Growth", "SMA 5/20 fast momentum as slope proxy on tech.",
                   _US_TECH, 5, 20, "sma"),
    "F010": _trend("F010", "ADX Trend Strength", "EMA 20/50 on diversified basket with trend filter.",
                   _DIVERSIFIED, 20, 50, "ema"),
    "F011": _trend("F011", "EMA Ribbon NDX", "EMA 10/50 ribbon on QQQ and NASDAQ components.",
                   ["QQQ", "AAPL", "MSFT", "NVDA", "META"], 10, 50, "ema"),
    "F012": _trend("F012", "MA Velocity FAANG", "EMA 5/21 velocity on FAANG+ cluster.",
                   ["AAPL", "AMZN", "GOOGL", "META", "NVDA", "TSLA"], 5, 21, "ema"),
    "F013": _trend("F013", "Parabolic SAR Proxy", "SMA 3/10 ultra-short proxy for volatile growth.",
                   ["NVDA", "TSLA", "AMD", "MSTR", "COIN"], 3, 10, "ema"),
    "F014": _trend("F014", "Keltner Trend Sectors", "SMA 20/50 on sector ETFs.",
                   _SECTOR_ETFS, 20, 50, "sma"),
    "F015": _trend("F015", "Price Channel 4-Week", "SMA 20/80 (4-week / 16-week) on commodity ETFs.",
                   _COMMODITY_ETFS, 20, 80, "sma"),
    "F016": _trend("F016", "Momentum MA SPX", "EMA 50/150 with momentum filter on SPX proxies.",
                   ["SPY", "IVV", "VOO", "SPLG"], 50, 150, "ema"),
    "F017": _trend("F017", "Adaptive MA Mid Cap", "EMA 30/90 adaptive proxy on mid caps.",
                   _US_MID_CAP, 30, 90, "ema"),
    "F018": _trend("F018", "Vol-Adj Trend All Cap", "SMA 50/200 vol-adjusted proxy across all caps.",
                   ["SPY", "IWM", "MDY", "QQQ", "VTI"], 50, 200, "sma"),
    "F019": _trend("F019", "Multi-TF MA Alignment", "SMA 100/200 long-term alignment on large caps.",
                   _US_LARGE_CAP, 100, 200, "sma"),
    "F020": _trend("F020", "Regime Adaptive Trend", "EMA 20/100 regime-adaptive on SPY.",
                   ["SPY", "QQQ", "TLT", "GLD"], 20, 100, "ema"),

    # ── F021–F035: Breakout ────────────────────────────────────────────────────
    "F021": _breakout("F021", "52-Week High Breakout", "20-day range breakout on US large caps.",
                      _US_LARGE_CAP, 252, long_only=True),
    "F022": _breakout("F022", "Bollinger Squeeze BO", "20-day breakout on all caps after vol compression.",
                      _DIVERSIFIED, 20),
    "F023": _breakout("F023", "NR7 Narrow Range", "7-day narrow range breakout on liquid equities.",
                      _US_LARGE_CAP, 7),
    "F024": _breakout("F024", "Weekly High Breakout", "5-day (weekly) high breakout on large caps.",
                      _US_LARGE_CAP, 5, long_only=True),
    "F025": _breakout("F025", "Pivot Point Breakout", "10-day range breakout (pivot proxy) on large caps.",
                      _US_LARGE_CAP, 10),
    "F026": _breakout("F026", "High-Volume Breakout", "20-day high breakout on sector ETFs.",
                      _SECTOR_ETFS, 20, long_only=True),
    "F027": _breakout("F027", "Consolidation Breakout", "30-day flat-base breakout on growth stocks.",
                      ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL"], 30, long_only=True),
    "F028": _breakout("F028", "ATR Channel Breakout", "14-day breakout on sector ETFs.",
                      _SECTOR_ETFS, 14),
    "F029": _breakout("F029", "Monthly High Breakout", "21-day (monthly) high breakout on large caps.",
                      _US_LARGE_CAP, 21, long_only=True),
    "F030": _breakout("F030", "Turtle 20-Day Entry", "Classic Turtle 20-day entry on commodity ETFs.",
                      _COMMODITY_ETFS, 20),
    "F031": _breakout("F031", "Turtle 55-Day Entry", "Classic Turtle 55-day entry on commodity ETFs.",
                      _COMMODITY_ETFS, 55),
    "F032": _breakout("F032", "Momentum Breakout", "50-day breakout on tech and growth.",
                      _US_TECH, 50, long_only=True),
    "F033": _breakout("F033", "Wedge Breakout", "40-day breakout (wedge proxy) on diversified.",
                      _DIVERSIFIED, 40),
    "F034": _breakout("F034", "Cup & Handle Proxy", "63-day (quarterly) range breakout on growth.",
                      ["NVDA", "AAPL", "MSFT", "AMZN", "QQQ"], 63, long_only=True),
    "F035": _breakout("F035", "Flag Pennant Breakout", "10-day narrow range breakout on tech.",
                      _US_TECH, 10),

    # ── F036–F050: Mean Reversion ──────────────────────────────────────────────
    "F036": _mr("F036", "Bollinger MR Core", "Classic BB 20-day 2σ long+short MR.",
                _US_LARGE_CAP, 20, 2.0, long_only=False),
    "F037": _mr("F037", "RSI-2 Mean Rev SPY", "BB 2-day 2σ (RSI-2 proxy MR) on SPY.",
                ["SPY"], 2, 2.0, long_only=True),
    "F038": _mr("F038", "Z-Score MR Large Cap", "BB 20-day 2σ long-only MR on large caps.",
                _US_LARGE_CAP, 20, 2.0, long_only=True),
    "F039": _mr("F039", "VWAP Reversion ETFs", "BB 5-day 1.5σ (VWAP proxy) on liquid ETFs.",
                _DIVERSIFIED, 5, 1.5, long_only=True),
    "F040": _mr("F040", "Oversold Bounce All", "BB 20-day 2.5σ deep oversold bounce.",
                _DIVERSIFIED, 20, 2.5, long_only=True),
    "F041": _mr("F041", "Long-Only BB Defensive", "BB 30-day 2σ on defensive sectors.",
                ["XLP", "XLU", "XLV", "XLF", "TLT"], 30, 2.0, long_only=True),
    "F042": _mr("F042", "StatArb Proxy Pairs", "BB 10-day 1σ on correlated ETF pairs.",
                ["XLK", "QQQ", "XLF", "KRE", "GLD", "SLV"], 10, 1.0, long_only=False),
    "F043": _mr("F043", "Band Reversal High Vol", "BB 20-day 2σ on volatile growth stocks.",
                ["NVDA", "TSLA", "AMD", "MSTR", "COIN"], 20, 2.0, long_only=False),
    "F044": _mr("F044", "Stochastic Proxy MR", "BB 14-day 1.5σ (Stoch proxy) on all caps.",
                _US_LARGE_CAP, 14, 1.5, long_only=True),
    "F045": _mr("F045", "CCI Mean Reversion ETFs", "BB 20-day 2σ on sector ETFs.",
                _SECTOR_ETFS, 20, 2.0, long_only=True),
    "F046": _mr("F046", "Donchian Midpoint Rev", "BB 20-day 1σ tight reversion to commodity midpoints.",
                _COMMODITY_ETFS, 20, 1.0, long_only=True),
    "F047": _mr("F047", "RSI-5 Extreme Reversal", "BB 5-day 3σ extreme reversal on SPY.",
                ["SPY", "QQQ", "IWM"], 5, 3.0, long_only=True),
    "F048": _mr("F048", "Gap Fill Strategy", "BB 3-day 2σ gap-fill proxy on liquid large caps.",
                ["AAPL", "MSFT", "AMZN", "GOOGL", "SPY"], 3, 2.0, long_only=True),
    "F049": _mr("F049", "Short-Term Reversal", "BB 5-day 2σ 5-day reversal on large caps.",
                _US_LARGE_CAP, 5, 2.0, long_only=True),
    "F050": _mr("F050", "MACD Divergence MR", "BB 26-day 2σ MACD-window reversion on tech.",
                _US_TECH, 26, 2.0, long_only=False),

    # ── F051–F060: Cross-Sectional Momentum ───────────────────────────────────
    "F051": _momentum("F051", "12-1 Momentum S&P 500", "Classic 12-1 month momentum on S&P 500 proxies.",
                      _US_LARGE_CAP, 252, 21, top_n=3),
    "F052": _momentum("F052", "Dual Momentum Antonacci", "Gary Antonacci dual momentum: SPY vs TLT vs GLD.",
                      ["SPY", "TLT", "GLD", "IEF"], 252, 21, top_n=1),
    "F053": _momentum("F053", "Sector Rotation ETFs", "Monthly sector rotation on 11 SPDR ETFs.",
                      _SECTOR_ETFS, 126, 21, top_n=3),
    "F054": _momentum("F054", "52-Week Proximity Mom", "6-month momentum on large caps.",
                      _US_LARGE_CAP, 126, 21, top_n=3),
    "F055": _momentum("F055", "Post-Earnings Momentum", "3-month momentum on tech (PEAD proxy).",
                      _US_TECH, 63, 5, top_n=3),
    "F056": _momentum("F056", "Small Cap Momentum IWM", "12-month momentum on small cap ETFs.",
                      ["IWM", "VXF", "IJR", "VTWO", "SCHA"], 252, 21, top_n=2),
    "F057": _momentum("F057", "Large Cap Momentum SPY", "12-month momentum on large cap ETFs.",
                      ["SPY", "IVV", "VOO", "SPLG", "QQQ"], 252, 21, top_n=2),
    "F058": _momentum("F058", "Crypto Cross-Section Mom", "3-month cross-sectional on crypto (via equity proxies).",
                      ["COIN", "MSTR", "BITO", "IBIT"], 63, 5, top_n=2),
    "F059": _momentum("F059", "ETF Rotation 20", "Monthly rotation across 20 diverse ETFs.",
                      _DIVERSIFIED + _SECTOR_ETFS[:10], 126, 21, top_n=4),
    "F060": _momentum("F060", "International ETF Mom", "12-month momentum on intl ETFs.",
                      ["VEA", "VWO", "EEM", "EWJ", "EWG", "EWY", "FXI"], 252, 21, top_n=2),

    # ── F061–F070: Volatility ──────────────────────────────────────────────────
    "F061": _vol("F061", "Vol Targeting SPY", "Scale position by inverse realised vol (SPY).",
                 ["SPY"], "VolTargetingFund", target_vol=0.15, lookback=21),
    "F062": _vol("F062", "VIX Regime Positioning", "VIX-regime based risk-on/risk-off (SPY + TLT).",
                 ["SPY", "TLT"], "VIXRegimeFund"),
    "F063": _vol("F063", "GARCH Vol Forecast", "GARCH(1,1) vol forecast sizing on SPY.",
                 ["SPY"], "VolTargetingFund", model="garch"),
    "F064": _vol("F064", "Vol of Vol Signal", "VIX of VIX proxy on SPY.",
                 ["SPY", "QQQ"], "VolTargetingFund"),
    "F065": _vol("F065", "Historical Vol Regime", "Switch equities/bonds on realised vol regime.",
                 ["SPY", "TLT", "GLD"], "VolTargetingFund"),
    "F066": _vol("F066", "ATR Trailing Stop", "ATR-based position exit on momentum portfolio.",
                 _US_LARGE_CAP, "VolTargetingFund", atr_multiplier=2.0),
    "F067": _vol("F067", "Vol Breakout Sectors", "Trade volatility breakout on sector ETFs.",
                 _SECTOR_ETFS, "VolTargetingFund"),
    "F068": _vol("F068", "Correlation Breakdown", "Signal on correlation breakdown in pairs.",
                 ["XLK", "QQQ", "XLF", "KRE"], "VolTargetingFund"),
    "F069": _vol("F069", "Vol Momentum Defensive", "Go defensive when realised vol is rising.",
                 ["SPY", "TLT", "GLD", "IEF"], "VolTargetingFund"),
    "F070": _vol("F070", "Implied vs Realised Vol", "Trade VIX premium: implied > realised → sell.",
                 ["SPY", "VIX"], "VolTargetingFund"),

    # ── F071–F080: Factor ──────────────────────────────────────────────────────
    "F071": _factor("F071", "Quality Factor Screen", "ROE + low debt quality screen on S&P 500.",
                    _US_LARGE_CAP, factor="quality"),
    "F072": _factor("F072", "Value Factor ETFs", "Value factor via VTV, IVE, VLUE.",
                    ["VTV", "IVE", "VLUE", "RPV", "SPVU"], factor="value"),
    "F073": _factor("F073", "Momentum Factor SPX", "Momentum factor via MTUM, QMOM.",
                    ["MTUM", "QMOM", "PDP", "FDMO"], factor="momentum"),
    "F074": _factor("F074", "Low Vol Anomaly", "Low volatility anomaly via SPLV, USMV.",
                    ["SPLV", "USMV", "LVHD", "FDLO"], factor="low_vol"),
    "F075": _factor("F075", "Size Factor Small Cap", "Small cap premium via IWM, VB.",
                    ["IWM", "VB", "IJR", "SCHA", "VTWO"], factor="size"),
    "F076": _factor("F076", "Profitability Factor", "Gross profitability factor via QUAL.",
                    ["QUAL", "DGRW", "DGRO"], factor="profitability"),
    "F077": _factor("F077", "Investment Factor", "Conservative investment factor via VTV.",
                    ["VTV", "IVE", "SPYD"], factor="investment"),
    "F078": _factor("F078", "Multi-Factor Combined", "Combined QUAL + MOM + VAL via LRGF.",
                    ["LRGF", "QMOM", "VLUE", "QUAL"], factor="multi"),
    "F079": _factor("F079", "Jensen Alpha Sectors", "Jensen alpha factor on sector ETFs.",
                    _SECTOR_ETFS, factor="alpha"),
    "F080": _factor("F080", "Sector Rotation Factor", "Factor-driven sector rotation on SPDR ETFs.",
                    _SECTOR_ETFS, factor="sector_rotation"),

    # ── F081–F085: Crypto ──────────────────────────────────────────────────────
    "F081": _crypto("F081", "BTC Momentum 20/50", "EMA 20/50 golden cross on BTC/USD.",
                    ["BTC/USD"], 20, 50, "ema"),
    "F082": _crypto("F082", "ETH Momentum 20/50", "EMA 20/50 golden cross on ETH/USD.",
                    ["ETH/USD"], 20, 50, "ema"),
    "F083": _crypto("F083", "Crypto Donchian BO", "SMA 10/20 breakout on BTC + ETH.",
                    _CRYPTO, 10, 20, "sma"),
    "F084": _crypto("F084", "BTC vs ETH Dual Mom", "Dual momentum: EMA 50/100 on BTC and ETH.",
                    _CRYPTO, 50, 100, "ema"),
    "F085": _crypto("F085", "Crypto Mean Rev", "EMA 5/20 mean-reversion entry on BTC.",
                    ["BTC/USD"], 5, 20, "ema"),

    # ── F086–F095: AI / LangGraph ──────────────────────────────────────────────
    "F086": _ai("F086", "AI Breakout Analyst", "LangGraph breakout AI analysis on US large caps.",
                ["AAPL", "MSFT", "NVDA"]),
    "F087": _ai("F087", "AI MA Crossover Analyst", "LangGraph MA AI analysis on US tech leaders.",
                ["GOOGL", "META", "AMZN"]),
    "F088": _ai("F088", "AI Multi-Strategy Synthesis", "LangGraph multi-strategy synthesis on SPY.",
                ["SPY", "QQQ"]),
    "F089": _ai("F089", "AI News Sentiment", "LangGraph news + TA synthesis on liquid large caps.",
                ["AAPL", "TSLA", "NVDA"]),
    "F090": _ai("F090", "AI Earnings Analysis", "LangGraph pre/post-earnings AI on tech.",
                ["AAPL", "MSFT", "GOOGL"]),
    "F091": _ai("F091", "AI Market Regime", "LangGraph regime classification on SPY + TLT.",
                ["SPY", "TLT"]),
    "F092": _ai("F092", "AI Risk Parity", "LangGraph-guided risk parity allocation.",
                ["SPY", "TLT", "GLD"]),
    "F093": _ai("F093", "AI Options Flow", "LangGraph options flow + price action on SPY.",
                ["SPY", "QQQ"]),
    "F094": _ai("F094", "AI Crypto Multi-Signal", "LangGraph crypto AI (via COIN/MSTR proxies).",
                ["COIN", "MSTR", "BITO"]),
    "F095": _ai("F095", "AI Portfolio Optimizer", "LangGraph full portfolio construction signal.",
                ["SPY", "QQQ", "GLD", "TLT", "IWM"]),

    # ── F096–F100: Options ─────────────────────────────────────────────────────
    "F096": _options("F096", "Covered Call Writing", "Write OTM covered calls on SPY, AAPL, MSFT.", _OPTIONS_UNIVERSE),
    "F097": _options("F097", "Cash-Secured Put", "Sell OTM cash-secured puts on quality stocks.", ["AAPL", "MSFT", "JPM", "JNJ"]),
    "F098": _options("F098", "Iron Condor SPY", "Iron condor on SPY (range-bound).", ["SPY"]),
    "F099": _options("F099", "Wheel Strategy", "CSP → covered call wheel on select stocks.", ["AAPL", "MSFT", "SPY"]),
    "F100": _options("F100", "Calendar Spread SPY", "Buy far-month, sell near-month on SPY.", ["SPY", "QQQ"]),
}


def get_fund(fund_id: str) -> FundConfig:
    """Retrieve a FundConfig by ID. Raises KeyError with a helpful message."""
    if fund_id not in FUND_REGISTRY:
        raise KeyError(
            f"Fund {fund_id!r} not found. Valid IDs: F001–F100. "
            f"Total registered: {len(FUND_REGISTRY)}."
        )
    return FUND_REGISTRY[fund_id]


def funds_by_category(category: str) -> list[FundConfig]:
    """Return all funds matching a StrategyCategory value string."""
    return [f for f in FUND_REGISTRY.values() if f.category.value == category]


def funds_by_asset_class(asset_class: str) -> list[FundConfig]:
    """Return all funds matching an AssetClass value string."""
    return [f for f in FUND_REGISTRY.values() if f.asset_class.value == asset_class]
