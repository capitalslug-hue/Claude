"""Market data access layer backed by Yahoo Finance (via yfinance).

No API key or account required. All functions raise ``DataError`` with a
human-readable message when a symbol is unknown or no data is returned, so the
MCP tools can surface a clean error to the model.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise ImportError(
        "yfinance is required. Install with `pip install tradingview-mcp` "
        "or `pip install yfinance`."
    ) from exc


class DataError(RuntimeError):
    """Raised when market data cannot be retrieved for a request."""


# Valid yfinance periods/intervals, exposed for tool validation and docs.
VALID_PERIODS = (
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
)
VALID_INTERVALS = (
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo",
)


def _normalize(symbol: str) -> str:
    return symbol.strip().upper()


def get_history(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Return an OHLCV DataFrame for ``symbol``.

    Columns: Open, High, Low, Close, Volume. Index is a DatetimeIndex.
    """
    symbol = _normalize(symbol)
    if period not in VALID_PERIODS:
        raise DataError(f"Invalid period '{period}'. Valid: {', '.join(VALID_PERIODS)}")
    if interval not in VALID_INTERVALS:
        raise DataError(
            f"Invalid interval '{interval}'. Valid: {', '.join(VALID_INTERVALS)}"
        )

    ticker = yf.Ticker(symbol)
    try:
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
    except Exception as exc:  # network/SSL/parse errors from the data provider
        raise DataError(
            f"Failed to fetch data for '{symbol}': {exc}. "
            "The market data provider may be unreachable."
        ) from exc
    if df is None or df.empty:
        raise DataError(
            f"No data returned for '{symbol}' (period={period}, interval={interval}). "
            "Check the symbol and that the period/interval combination is supported."
        )
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    if df.empty:
        raise DataError(f"No usable OHLCV rows for '{symbol}'.")
    return df


@lru_cache(maxsize=256)
def _cached_info(symbol: str) -> dict:
    return yf.Ticker(symbol).fast_info  # type: ignore[return-value]


def get_quote(symbol: str) -> dict:
    """Return a lightweight latest-price snapshot for ``symbol``."""
    symbol = _normalize(symbol)
    df = get_history(symbol, period="5d", interval="1d")
    last = df.iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) >= 2 else last["Close"]
    change = float(last["Close"] - prev_close)
    change_pct = float(change / prev_close * 100) if prev_close else 0.0

    quote = {
        "symbol": symbol,
        "price": round(float(last["Close"]), 4),
        "open": round(float(last["Open"]), 4),
        "high": round(float(last["High"]), 4),
        "low": round(float(last["Low"]), 4),
        "volume": int(last["Volume"]),
        "previous_close": round(float(prev_close), 4),
        "change": round(change, 4),
        "change_percent": round(change_pct, 2),
        "as_of": df.index[-1].isoformat(),
    }
    try:
        fast = yf.Ticker(symbol).fast_info
        quote["currency"] = getattr(fast, "currency", None)
        quote["market_cap"] = getattr(fast, "market_cap", None)
    except Exception:
        pass
    return quote
