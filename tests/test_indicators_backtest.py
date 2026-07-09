"""Offline tests for indicators, analysis, and the backtest engine.

These use a synthetic price series so they run without any network access.
"""

import numpy as np
import pandas as pd
import pytest

from tradingview_mcp import analysis, backtest
from tradingview_mcp import indicators as ind


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    np.random.seed(7)
    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    price = 100 + np.cumsum(np.random.randn(n)) + np.linspace(0, 15, n)
    return pd.DataFrame(
        {
            "Open": price + np.random.randn(n) * 0.2,
            "High": price + np.abs(np.random.randn(n)),
            "Low": price - np.abs(np.random.randn(n)),
            "Close": price,
            "Volume": np.random.randint(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


def test_rsi_bounds(ohlcv):
    r = ind.rsi(ohlcv["Close"]).dropna()
    assert not r.empty
    assert (r >= 0).all() and (r <= 100).all()


def test_macd_hist_is_macd_minus_signal(ohlcv):
    m = ind.macd(ohlcv["Close"]).dropna()
    assert np.allclose(m["hist"], m["macd"] - m["signal"])


def test_bollinger_ordering(ohlcv):
    bb = ind.bollinger_bands(ohlcv["Close"]).dropna()
    assert (bb["upper"] >= bb["middle"]).all()
    assert (bb["middle"] >= bb["lower"]).all()


def test_supertrend_direction_values(ohlcv):
    st = ind.supertrend(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
    assert set(st["direction"].dropna().unique()).issubset({1.0, -1.0})


def test_technical_summary_signal(ohlcv):
    summary = analysis.technical_summary(ohlcv)
    assert summary["signal"] in {"BUY", "SELL", "NEUTRAL"}
    assert "rsi_14" in summary["indicators"]


@pytest.mark.parametrize("strategy", backtest.STRATEGIES)
def test_backtest_runs_for_all_strategies(ohlcv, strategy):
    result = backtest.run_backtest(ohlcv, strategy)
    m = result.metrics
    assert set(m) >= {
        "total_return_pct",
        "sharpe_ratio",
        "calmar_ratio",
        "max_drawdown_pct",
        "win_rate_pct",
        "num_trades",
    }
    assert m["max_drawdown_pct"] <= 0
    assert result.num_trades >= 0


def test_backtest_unknown_strategy_raises(ohlcv):
    with pytest.raises(ValueError):
        backtest.run_backtest(ohlcv, "does_not_exist")


def test_equity_curve_included_when_requested(ohlcv):
    result = backtest.run_backtest(ohlcv, "ema_cross", include_curve=True)
    assert len(result.equity_curve) == len(ohlcv)
    assert {"date", "equity"} <= set(result.equity_curve[0])
