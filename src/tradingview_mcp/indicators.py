"""Pure pandas/numpy technical indicator implementations.

Kept dependency-free (no TA-Lib) so the server installs cleanly anywhere.
Every function takes a pandas Series/DataFrame of prices and returns a Series
or DataFrame aligned to the input index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing is an EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # When avg_loss is 0 the market only went up -> RSI 100.
    out = out.where(avg_loss != 0, 100.0)
    return out


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands (middle SMA, upper, lower) and %B."""
    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower)
    percent_b = (series - lower) / width.replace(0.0, np.nan)
    return pd.DataFrame(
        {"middle": middle, "upper": upper, "lower": lower, "percent_b": percent_b}
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Stochastic oscillator %K and %D."""
    lowest = low.rolling(window=k_period, min_periods=k_period).min()
    highest = high.rolling(window=k_period, min_periods=k_period).max()
    rng = (highest - lowest).replace(0.0, np.nan)
    k = 100 * (close - lowest) / rng
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Supertrend indicator. Returns the trend line and direction (1 up, -1 down)."""
    atr_series = atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_basic = hl2 + multiplier * atr_series
    lower_basic = hl2 - multiplier * atr_series

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    trend = pd.Series(index=close.index, dtype="float64")
    direction = pd.Series(index=close.index, dtype="float64")

    prev_dir = 1.0
    for i in range(len(close)):
        if i == 0 or np.isnan(atr_series.iloc[i]):
            trend.iloc[i] = np.nan
            direction.iloc[i] = prev_dir
            continue

        if close.iloc[i - 1] > upper.iloc[i - 1]:
            upper.iloc[i] = upper_basic.iloc[i]
        else:
            upper.iloc[i] = min(upper_basic.iloc[i], upper.iloc[i - 1])

        if close.iloc[i - 1] < lower.iloc[i - 1]:
            lower.iloc[i] = lower_basic.iloc[i]
        else:
            lower.iloc[i] = max(lower_basic.iloc[i], lower.iloc[i - 1])

        if close.iloc[i] > upper.iloc[i - 1]:
            prev_dir = 1.0
        elif close.iloc[i] < lower.iloc[i - 1]:
            prev_dir = -1.0
        direction.iloc[i] = prev_dir
        trend.iloc[i] = lower.iloc[i] if prev_dir == 1.0 else upper.iloc[i]

    return pd.DataFrame({"supertrend": trend, "direction": direction})
