# TradingView MCP

A [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that
gives AI assistants (Claude Desktop, Cursor, etc.) live market data, technical
analysis, and a backtesting engine — powered entirely by **free Yahoo Finance
data**. No API key or account required.

## Features

- **Market data** — latest quotes and OHLCV history for stocks, crypto, forex,
  and indices (any Yahoo Finance symbol: `AAPL`, `BTC-USD`, `EURUSD=X`, `^GSPC`).
- **Technical analysis** — RSI, MACD, SMA/EMA, Bollinger Bands, ATR, Stochastic,
  Supertrend, plus an aggregated **BUY / SELL / NEUTRAL** signal.
- **Screening** — run analysis across a universe of tickers and filter by signal.
- **Backtesting** — six long-only strategies with institutional metrics
  (total & annualized return, Sharpe, Calmar, max drawdown, win rate) and a
  strategy comparison/ranking tool.

Indicators and the backtest engine are implemented in pure `pandas`/`numpy`
(no TA-Lib), so it installs cleanly anywhere.

## Installation

```bash
pip install -e .
# or, for an isolated run:
pip install .
```

Requires Python 3.10+.

## Running

The server speaks MCP over stdio:

```bash
tradingview-mcp
# or
python -m tradingview_mcp.server
```

### Claude Desktop config

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tradingview": {
      "command": "tradingview-mcp"
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `get_quote` | Latest price snapshot for one symbol. |
| `market_snapshot` | Latest snapshots for several symbols at once. |
| `get_price_history` | OHLCV candles for a symbol/period/interval. |
| `get_technical_analysis` | Full indicator suite + aggregated signal. |
| `get_multiple_analysis` | Signals across several symbols. |
| `screen_stocks` | Keep only symbols matching a target signal. |
| `backtest_strategy` | Backtest one strategy with performance metrics. |
| `compare_strategies` | Backtest and rank strategies by Sharpe ratio. |

### Backtest strategies

`rsi`, `macd`, `ema_cross`, `sma_cross`, `bollinger`, `supertrend` — each accepts
optional `params` (e.g. `{"fast": 10, "slow": 30}`).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Notes & disclaimer

Data is sourced from Yahoo Finance and is provided as-is. Backtests are
long-only, use next-bar execution to avoid look-ahead bias, and charge a
configurable per-trade fee. **This project is for research and educational
purposes only and is not financial advice.**

## License

MIT
