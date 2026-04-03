"""
ta — Technical analysis primitives for the myfinance2 trader assistants.

Subpackages:
    breakout  — Range breakout primitives: range quality, volatility compression,
                volume behaviour during consolidation.
    ma        — Moving average crossover primitives: trend strength (RSI, ADX,
                MA gap/slope), volume expansion at crossover bars.

Shared utilities:
    ta.utils  — ols_slope: the only shared mathematical primitive.
                All modules in ta.breakout and ta.ma import from here.
"""
