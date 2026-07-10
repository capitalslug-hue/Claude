---
name: markov-2-hedge-fund-method
description: Markov 2.0 — Hedge Fund Method (corrected). Regime detection via Markov transition matrices with three v1 flaws fixed (stride sampling, label verification, explicit FILTER/STANDALONE modes). Use when the user asks to run a Markov regime analysis on a ticker, gate a strategy by market regime, build a transition matrix, compute a regime signal, or backtest the Markov hedge fund method. Arguments: ticker(s) and optional flags.
---

# Markov 2.0 — Hedge Fund Method (corrected)

Regime detection: label each day BULL / SIDEWAYS / BEAR from the 20-day
cumulative return (≥ +5% / ≤ −5% / else), build a Markov transition matrix,
and trade the signal **P(bull next) − P(bear next)** — sign is direction,
magnitude is conviction.

## How to run

The engine is `scripts/markov2.py` (relative to this skill's directory).
Dependencies: `pandas numpy matplotlib` (`pip install` if missing). Data comes
from Yahoo's free chart API — no keys, no accounts.

```bash
python3 scripts/markov2.py SPY --years 10 --mode filter --outdir <scratch>
```

Key flags: `--window 20 --bull 0.05 --bear -0.05` (state definition),
`--mode filter|standalone`, `--threshold` (filter gate), `--cap` (standalone
max position), `--enhanced` (return + ATR + relative-volume cluster states),
`--hmm` (Gaussian HMM cross-check; needs `pip install hmmlearn`).

Run it, then relay the output to the user: both matrices, verification report,
current signal, matrix-power forecasts, walk-forward metrics, and send the
equity-curve PNG. Offer to re-run on any ticker the user names.

## The three fixes — never bypass these

1. **FIX 1 — Stride sampling.** Never build the matrix from overlapping rolling
   windows: consecutive 20-day windows share 19 days, which fakes persistence
   on the diagonal. Count transitions between NON-overlapping windows
   (stride = window length). Always show BOTH matrices — overlapping (legacy)
   and stride-sampled (true) — side by side, with the one-line warning that
   only the stride-sampled one is statistically honest. The script does this;
   never present the legacy matrix alone.
2. **FIX 2 — Label verification.** Before showing any table, chart, or matrix,
   the state labels are programmatically self-checked against known periods
   (COVID crash = BEAR, post-COVID rebound = BULL, 2017 grind = SIDEWAYS) plus
   data-derived extremes for any ticker. The script aborts on mismatch. If you
   build any additional display by hand, re-verify the label mapping against
   the data before showing it — v1 shipped with bull/bear swapped in a display.
3. **FIX 3 — Two explicit modes.** Always state which mode is active; never
   leave it ambiguous:
   - **FILTER (default):** the regime gates the user's existing strategy —
     longs only when signal > threshold, shorts only when below, flat in chop.
     The strategy stays theirs; Markov 2.0 decides WHEN it may act.
   - **STANDALONE:** trade the differential directly; position = signal × cap.

## Method notes

- **Multi-day forecasts** come from matrix powers; they converge to the
  stationary distribution, so long-horizon forecasts carry no signal — say so.
- **Backtests are walk-forward only**: expanding window, matrix refit every 20
  bars, positions applied to the NEXT bar. Never test on data the matrix has
  learned from.
- **Enhanced states** (offer, don't force): `--enhanced` clusters on 20-day
  return + ATR + relative volume so "bear and violent" ≠ "bear and asleep";
  report how the matrix and signal change vs price-only.
- **HMM mode** (optional): `--hmm` fits a 3-state Gaussian HMM with no
  hand-made labels and reports agreement with the threshold labels —
  agreement is the green light.

## Reporting

Report honestly: win rate, profit factor, max drawdown, equity curve image,
before-fix vs after-fix comparison. Always end with exactly this caveat:

> "Backtests flatter. The fixed matrix shows uglier, truer numbers — those are
> the only ones worth trading."

This is analysis tooling, not financial advice — keep that framing.
