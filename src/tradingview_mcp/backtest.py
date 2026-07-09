"""A compact vectorized backtesting engine.

Strategies produce a *position* series (1 = long, 0 = flat) evaluated on the
next bar's return to avoid look-ahead bias. Metrics include total/annualized
return, Sharpe, Calmar, max drawdown, and win rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind

STRATEGIES = (
    "rsi",
    "macd",
    "ema_cross",
    "sma_cross",
    "bollinger",
    "supertrend",
)


@dataclass
class BacktestResult:
    strategy: str
    params: dict
    metrics: dict
    num_trades: int
    equity_curve: list = field(default_factory=list)


def _positions(df: pd.DataFrame, strategy: str, params: dict) -> pd.Series:
    """Return a 0/1 long-only position series for the given strategy."""
    close = df["Close"]
    strategy = strategy.lower()

    if strategy == "rsi":
        period = int(params.get("period", 14))
        lower = float(params.get("lower", 30))
        upper = float(params.get("upper", 70))
        r = ind.rsi(close, period)
        # Enter when RSI crosses back above the oversold line, exit above upper.
        raw = pd.Series(np.nan, index=close.index)
        raw[r < lower] = 1.0
        raw[r > upper] = 0.0
        pos = raw.ffill().fillna(0.0)

    elif strategy == "macd":
        m = ind.macd(close)
        pos = (m["macd"] > m["signal"]).astype(float)

    elif strategy == "ema_cross":
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        pos = (ind.ema(close, fast) > ind.ema(close, slow)).astype(float)

    elif strategy == "sma_cross":
        fast = int(params.get("fast", 50))
        slow = int(params.get("slow", 200))
        pos = (ind.sma(close, fast) > ind.sma(close, slow)).astype(float)

    elif strategy == "bollinger":
        period = int(params.get("period", 20))
        num_std = float(params.get("num_std", 2.0))
        bb = ind.bollinger_bands(close, period, num_std)
        raw = pd.Series(np.nan, index=close.index)
        raw[close < bb["lower"]] = 1.0
        raw[close > bb["middle"]] = 0.0
        pos = raw.ffill().fillna(0.0)

    elif strategy == "supertrend":
        period = int(params.get("period", 10))
        multiplier = float(params.get("multiplier", 3.0))
        st = ind.supertrend(df["High"], df["Low"], close, period, multiplier)
        pos = (st["direction"] > 0).astype(float)

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Valid: {', '.join(STRATEGIES)}"
        )

    return pos.fillna(0.0)


def _annualization_factor(index: pd.DatetimeIndex) -> float:
    """Estimate periods-per-year from the median spacing of the index."""
    if len(index) < 3:
        return 252.0
    deltas = np.diff(index.view("int64")) / 1e9  # seconds
    median_sec = float(np.median(deltas))
    if median_sec <= 0:
        return 252.0
    periods_per_day = 86400.0 / median_sec
    # Daily-or-slower bars -> trading days; intraday -> scale by ~6.5h sessions.
    if median_sec >= 82800:  # >= ~23h spacing => daily+
        return 252.0
    return 252.0 * min(periods_per_day, 6.5 * 60)  # cap intraday scaling


def _metrics(returns: pd.Series, equity: pd.Series, trades: int) -> dict:
    ann = _annualization_factor(equity.index)
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    n = len(returns)
    ann_return = float((equity.iloc[-1]) ** (ann / n) - 1.0) if n > 0 and equity.iloc[-1] > 0 else 0.0

    std = float(returns.std())
    sharpe = float(returns.mean() / std * np.sqrt(ann)) if std > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    wins = int((returns[returns != 0] > 0).sum())
    active = int((returns != 0).sum())
    win_rate = float(wins / active) if active else 0.0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate * 100, 2),
        "num_trades": trades,
        "exposure_pct": round(float((returns != 0).mean()) * 100, 2) if n else 0.0,
    }


def run_backtest(
    df: pd.DataFrame,
    strategy: str,
    params: dict | None = None,
    fee_bps: float = 5.0,
    include_curve: bool = False,
) -> BacktestResult:
    """Run a long-only backtest and return metrics.

    ``fee_bps`` is charged (round-trip friendly) on each position change.
    """
    params = dict(params or {})
    pos = _positions(df, strategy, params)
    # Trade on the next bar to avoid look-ahead.
    exec_pos = pos.shift(1).fillna(0.0)
    bar_return = df["Close"].pct_change().fillna(0.0)

    trade_changes = exec_pos.diff().abs().fillna(exec_pos.abs())
    fee = trade_changes * (fee_bps / 10000.0)
    strat_return = exec_pos * bar_return - fee

    equity = (1.0 + strat_return).cumprod()
    num_trades = int((exec_pos.diff().fillna(0) > 0).sum())
    metrics = _metrics(strat_return, equity, num_trades)

    curve: list = []
    if include_curve:
        curve = [
            {"date": ts.isoformat(), "equity": round(float(v), 6)}
            for ts, v in equity.items()
        ]

    return BacktestResult(
        strategy=strategy.lower(),
        params=params,
        metrics=metrics,
        num_trades=num_trades,
        equity_curve=curve,
    )
