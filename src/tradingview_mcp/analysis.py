"""Higher-level analysis helpers that combine indicators into a signal summary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def _last(series: pd.Series) -> float | None:
    val = series.dropna()
    return round(float(val.iloc[-1]), 4) if len(val) else None


def technical_summary(df: pd.DataFrame) -> dict:
    """Compute a spread of indicators plus an aggregated BUY/SELL/NEUTRAL signal."""
    close = df["Close"]
    high, low = df["High"], df["Low"]

    rsi_val = ind.rsi(close)
    macd_df = ind.macd(close)
    bb = ind.bollinger_bands(close)
    sma20 = ind.sma(close, 20)
    sma50 = ind.sma(close, 50)
    ema20 = ind.ema(close, 20)
    atr_val = ind.atr(high, low, close)
    stoch = ind.stochastic(high, low, close)

    votes = []  # +1 bullish, -1 bearish
    last_close = float(close.iloc[-1])

    r = _last(rsi_val)
    if r is not None:
        votes.append(1 if r < 30 else -1 if r > 70 else 0)

    if _last(macd_df["hist"]) is not None:
        votes.append(1 if float(macd_df["hist"].dropna().iloc[-1]) > 0 else -1)

    s20, s50 = _last(sma20), _last(sma50)
    if s20 is not None and s50 is not None:
        votes.append(1 if s20 > s50 else -1)

    e20 = _last(ema20)
    if e20 is not None:
        votes.append(1 if last_close > e20 else -1)

    pb = _last(bb["percent_b"])
    if pb is not None:
        votes.append(1 if pb < 0.0 else -1 if pb > 1.0 else 0)

    k = _last(stoch["k"])
    if k is not None:
        votes.append(1 if k < 20 else -1 if k > 80 else 0)

    score = sum(votes)
    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    return {
        "signal": signal,
        "score": score,
        "last_close": round(last_close, 4),
        "indicators": {
            "rsi_14": r,
            "macd": _last(macd_df["macd"]),
            "macd_signal": _last(macd_df["signal"]),
            "macd_hist": _last(macd_df["hist"]),
            "sma_20": s20,
            "sma_50": s50,
            "ema_20": e20,
            "bb_upper": _last(bb["upper"]),
            "bb_lower": _last(bb["lower"]),
            "bb_percent_b": pb,
            "atr_14": _last(atr_val),
            "stoch_k": k,
            "stoch_d": _last(stoch["d"]),
        },
    }
