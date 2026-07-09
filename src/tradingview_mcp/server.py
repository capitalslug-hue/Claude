"""FastMCP server exposing market data, technical analysis, and backtesting tools.

Run directly with ``tradingview-mcp`` (stdio transport) after installing, or
``python -m tradingview_mcp.server``. All data comes from free Yahoo Finance
endpoints via yfinance — no API key or account required.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from . import analysis, backtest, data

mcp = FastMCP("tradingview-mcp")


# --------------------------------------------------------------------------- #
# Price & market data
# --------------------------------------------------------------------------- #
@mcp.tool()
def get_quote(symbol: str) -> dict[str, Any]:
    """Get the latest price snapshot for a ticker (stock, crypto, forex, index).

    Args:
        symbol: Ticker symbol, e.g. "AAPL", "BTC-USD", "EURUSD=X", "^GSPC".
    """
    try:
        return data.get_quote(symbol)
    except data.DataError as exc:
        return {"error": str(exc), "symbol": symbol.upper()}


@mcp.tool()
def market_snapshot(symbols: list[str]) -> dict[str, Any]:
    """Get latest price snapshots for several tickers at once.

    Args:
        symbols: List of ticker symbols, e.g. ["AAPL", "MSFT", "BTC-USD"].
    """
    quotes = []
    for sym in symbols:
        try:
            quotes.append(data.get_quote(sym))
        except data.DataError as exc:
            quotes.append({"symbol": sym.upper(), "error": str(exc)})
    return {"count": len(quotes), "quotes": quotes}


@mcp.tool()
def get_price_history(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    max_rows: int = 250,
) -> dict[str, Any]:
    """Get OHLCV price history for a ticker.

    Args:
        symbol: Ticker symbol.
        period: One of 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max.
        interval: One of 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo.
        max_rows: Cap on returned rows (most recent kept) to limit payload size.
    """
    try:
        df = data.get_history(symbol, period, interval)
    except data.DataError as exc:
        return {"error": str(exc), "symbol": symbol.upper()}

    if len(df) > max_rows:
        df = df.tail(max_rows)
    rows = [
        {
            "date": ts.isoformat(),
            "open": round(float(r.Open), 4),
            "high": round(float(r.High), 4),
            "low": round(float(r.Low), 4),
            "close": round(float(r.Close), 4),
            "volume": int(r.Volume),
        }
        for ts, r in zip(df.index, df.itertuples())
    ]
    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "rows": len(rows),
        "candles": rows,
    }


# --------------------------------------------------------------------------- #
# Technical analysis
# --------------------------------------------------------------------------- #
@mcp.tool()
def get_technical_analysis(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> dict[str, Any]:
    """Compute RSI, MACD, moving averages, Bollinger Bands, ATR, Stochastic and
    an aggregated BUY / SELL / NEUTRAL signal for a ticker.

    Args:
        symbol: Ticker symbol.
        period: History window used for the computation.
        interval: Bar interval.
    """
    try:
        df = data.get_history(symbol, period, interval)
    except data.DataError as exc:
        return {"error": str(exc), "symbol": symbol.upper()}

    summary = analysis.technical_summary(df)
    summary.update({"symbol": symbol.upper(), "period": period, "interval": interval})
    return summary


@mcp.tool()
def get_multiple_analysis(
    symbols: list[str],
    period: str = "6mo",
    interval: str = "1d",
) -> dict[str, Any]:
    """Run technical analysis across several tickers and return their signals.

    Args:
        symbols: List of ticker symbols.
        period: History window.
        interval: Bar interval.
    """
    results = []
    for sym in symbols:
        try:
            df = data.get_history(sym, period, interval)
            summary = analysis.technical_summary(df)
            results.append(
                {
                    "symbol": sym.upper(),
                    "signal": summary["signal"],
                    "score": summary["score"],
                    "last_close": summary["last_close"],
                    "rsi_14": summary["indicators"]["rsi_14"],
                }
            )
        except data.DataError as exc:
            results.append({"symbol": sym.upper(), "error": str(exc)})
    return {"count": len(results), "results": results}


@mcp.tool()
def screen_stocks(
    symbols: list[str],
    signal: str = "BUY",
    period: str = "6mo",
    interval: str = "1d",
) -> dict[str, Any]:
    """Screen a list of tickers and return only those matching a target signal.

    Args:
        symbols: Universe of tickers to screen.
        signal: Target signal to keep: BUY, SELL, or NEUTRAL.
        period: History window.
        interval: Bar interval.
    """
    target = signal.strip().upper()
    matches = []
    for sym in symbols:
        try:
            df = data.get_history(sym, period, interval)
            summary = analysis.technical_summary(df)
            if summary["signal"] == target:
                matches.append(
                    {
                        "symbol": sym.upper(),
                        "score": summary["score"],
                        "last_close": summary["last_close"],
                        "rsi_14": summary["indicators"]["rsi_14"],
                    }
                )
        except data.DataError:
            continue
    matches.sort(key=lambda m: m["score"], reverse=(target == "BUY"))
    return {"signal": target, "matches": matches, "count": len(matches)}


# --------------------------------------------------------------------------- #
# Backtesting
# --------------------------------------------------------------------------- #
@mcp.tool()
def backtest_strategy(
    symbol: str,
    strategy: str = "ema_cross",
    period: str = "2y",
    interval: str = "1d",
    params: dict[str, Any] | None = None,
    fee_bps: float = 5.0,
    include_curve: bool = False,
) -> dict[str, Any]:
    """Backtest a long-only trading strategy on historical data.

    Available strategies: rsi, macd, ema_cross, sma_cross, bollinger, supertrend.

    Args:
        symbol: Ticker symbol.
        strategy: Strategy name (see list above).
        period: History window for the backtest.
        interval: Bar interval.
        params: Optional strategy parameters, e.g. {"fast": 10, "slow": 30}.
        fee_bps: Transaction cost in basis points charged on each position change.
        include_curve: If true, include the full equity curve in the response.
    """
    if strategy.lower() not in backtest.STRATEGIES:
        return {
            "error": f"Unknown strategy '{strategy}'.",
            "valid_strategies": list(backtest.STRATEGIES),
        }
    try:
        df = data.get_history(symbol, period, interval)
    except data.DataError as exc:
        return {"error": str(exc), "symbol": symbol.upper()}

    result = backtest.run_backtest(df, strategy, params, fee_bps, include_curve)
    payload = {
        "symbol": symbol.upper(),
        "strategy": result.strategy,
        "params": result.params,
        "period": period,
        "interval": interval,
        "num_trades": result.num_trades,
        "metrics": result.metrics,
    }
    if include_curve:
        payload["equity_curve"] = result.equity_curve
    return payload


@mcp.tool()
def compare_strategies(
    symbol: str,
    strategies: list[str] | None = None,
    period: str = "2y",
    interval: str = "1d",
    fee_bps: float = 5.0,
) -> dict[str, Any]:
    """Backtest and rank several strategies on one ticker by Sharpe ratio.

    Args:
        symbol: Ticker symbol.
        strategies: Strategies to compare (defaults to all available).
        period: History window.
        interval: Bar interval.
        fee_bps: Transaction cost in basis points per position change.
    """
    strategies = strategies or list(backtest.STRATEGIES)
    try:
        df = data.get_history(symbol, period, interval)
    except data.DataError as exc:
        return {"error": str(exc), "symbol": symbol.upper()}

    rows = []
    for strat in strategies:
        if strat.lower() not in backtest.STRATEGIES:
            rows.append({"strategy": strat, "error": "unknown strategy"})
            continue
        result = backtest.run_backtest(df, strat, None, fee_bps, include_curve=False)
        rows.append({"strategy": result.strategy, **result.metrics})

    ranked = sorted(
        (r for r in rows if "error" not in r),
        key=lambda r: r["sharpe_ratio"],
        reverse=True,
    )
    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "ranked_by": "sharpe_ratio",
        "results": ranked + [r for r in rows if "error" in r],
    }


def main() -> None:
    """Console-script entry point: run the server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
