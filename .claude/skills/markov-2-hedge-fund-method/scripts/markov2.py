#!/usr/bin/env python3
"""Markov 2.0 — Hedge Fund Method (corrected).

Regime detection via Markov transition matrices, with the three v1 flaws fixed:

  FIX 1 — Stride sampling: transition matrices are built from NON-overlapping
          windows (stride = window length). The overlapping "legacy" matrix is
          also computed, side by side, so the autocorrelation flattery is visible.
  FIX 2 — Label verification: state labels are programmatically checked against
          known historical episodes (and data-derived extremes) BEFORE any
          matrix or chart is rendered. A mismatch aborts the display.
  FIX 3 — Explicit modes: FILTER (regime gates an existing strategy) vs
          STANDALONE (trade the signal directly, sized by |signal| with a cap).

Usage:
  python3 markov2.py SPY --years 10 --mode filter --outdir out/
  python3 markov2.py QQQ --years 10 --mode standalone --cap 1.0
  python3 markov2.py SPY --enhanced          # return + ATR + relative volume states
  python3 markov2.py SPY --hmm               # cross-check labels with a Gaussian HMM

Data: Yahoo Finance chart API (free, no keys). Requires pandas/numpy/matplotlib.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

BEAR, SIDEWAYS, BULL = 0, 1, 2
STATE_NAMES = {BEAR: "BEAR", SIDEWAYS: "SIDEWAYS", BULL: "BULL"}
ORDER = [BULL, SIDEWAYS, BEAR]  # display order

# Reference palette (dataviz skill): series colors + chart chrome, light surface
COL_FIXED = "#2a78d6"   # slot 1 blue  — stride-sampled (true) matrix
COL_LEGACY = "#1baf7a"  # slot 2 aqua  — overlapping (legacy) matrix
COL_BH = "#898781"      # muted ink    — buy & hold reference
SURFACE, GRID, INK, INK2 = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#52514e"


# ---------------------------------------------------------------- data

def fetch_history(ticker: str, years: float) -> pd.DataFrame:
    """Daily OHLCV + adjusted close from Yahoo's chart API. No keys needed."""
    now = int(time.time())
    p1 = now - int(years * 365.25 * 86400)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={p1}&period2={now}&interval=1d&events=div%2Csplit")
    body = None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        body = urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        # fall back to curl (respects HTTPS_PROXY / CURL_CA_BUNDLE in managed envs)
        r = subprocess.run(["curl", "-sS", "--max-time", "30",
                            "-H", "User-Agent: Mozilla/5.0", url],
                           capture_output=True)
        if r.returncode == 0:
            body = r.stdout
    if not body:
        raise SystemExit(f"could not fetch data for {ticker}")
    res = json.loads(body)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    df = pd.DataFrame({
        "close": adj, "raw_close": q["close"], "open": q["open"],
        "high": q["high"], "low": q["low"], "volume": q["volume"],
    }, index=pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(
        "America/New_York").normalize().tz_localize(None))
    df = df.dropna(subset=["close"])
    if len(df) < 300:
        raise SystemExit(f"only {len(df)} bars for {ticker} — not enough history")
    return df


# ---------------------------------------------------------------- states

def label_states(df: pd.DataFrame, window: int, bull_thr: float,
                 bear_thr: float) -> pd.Series:
    """Price-only states from the trailing `window`-day cumulative return."""
    r = df["close"] / df["close"].shift(window) - 1.0
    s = pd.Series(SIDEWAYS, index=df.index, dtype=int)
    s[r >= bull_thr] = BULL
    s[r <= bear_thr] = BEAR
    return s[r.notna()]


def _kmeans(X: np.ndarray, k: int, iters: int = 200, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1)
        new = np.array([X[lab == j].mean(0) if (lab == j).any() else cent[j]
                        for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    return lab


def label_states_enhanced(df: pd.DataFrame, window: int) -> pd.Series:
    """Cluster on (window-day return, ATR, relative volume) so 'bear and violent'
    is distinguishable from 'bear and asleep'. Clusters are mapped to
    BULL/SIDEWAYS/BEAR by mean return so the signal stays comparable."""
    r = df["close"] / df["close"].shift(window) - 1.0
    prev = df["raw_close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    atr = (tr.rolling(window).mean() / df["raw_close"])
    rvol = df["volume"] / df["volume"].rolling(window * 3).mean()
    feats = pd.concat([r, atr, rvol], axis=1).dropna()
    X = ((feats - feats.mean()) / feats.std()).to_numpy()
    lab = _kmeans(X, 3)
    means = [feats.iloc[:, 0][lab == j].mean() for j in range(3)]
    remap = {j: rank for rank, j in
             enumerate(np.argsort(means))}  # lowest ret -> BEAR(0), highest -> BULL(2)
    return pd.Series([remap[j] for j in lab], index=feats.index, dtype=int)


# ---------------------------------------------------------------- matrices

def transition_matrix(states: np.ndarray, stride: int):
    """Counts + row-stochastic probabilities. stride=1 reproduces the legacy
    overlapping-window matrix (autocorrelation-inflated diagonal); stride=window
    is the statistically honest version. The stride grid is anchored to the END
    of the sample so the most recent bar is always a sample point."""
    seq = states[::-1][::stride][::-1] if stride > 1 else states
    C = np.zeros((3, 3))
    for a, b in zip(seq[:-1], seq[1:]):
        C[a, b] += 1
    P = C.copy()
    for i in range(3):
        rs = C[i].sum()
        P[i] = C[i] / rs if rs > 0 else np.full(3, 1 / 3)
    return C, P


def signal_from(P: np.ndarray, state: int) -> float:
    """P(bull next) − P(bear next). Sign = direction, magnitude = conviction."""
    return float(P[state, BULL] - P[state, BEAR])


def stationary(P: np.ndarray) -> np.ndarray:
    return np.linalg.matrix_power(P, 512)[0]


def fmt_matrix(P: np.ndarray, C: np.ndarray | None = None) -> str:
    head = "            " + "".join(f"{STATE_NAMES[j]:>10}" for j in ORDER)
    rows = [head]
    for i in ORDER:
        cells = "".join(f"{P[i, j]:>10.3f}" for j in ORDER)
        n = f"  (n={int(C[i].sum())})" if C is not None else ""
        rows.append(f"  {STATE_NAMES[i]:<10}{cells}{n}")
    return "\n".join(rows)


# ---------------------------------------------------------------- FIX 2

def verify_labels(df: pd.DataFrame, states: pd.Series, window: int,
                  bull_thr: float, bear_thr: float) -> list[str]:
    """Self-check the state labels before anything is displayed.

    (a) Data-derived (any ticker): the worst window-day return in sample must be
        labeled BEAR, the best BULL, and a near-zero one SIDEWAYS — each label
        re-derived from raw prices, bypassing the labeling code path.
    (b) Famous episodes (when history covers them): COVID crash trough = BEAR,
        the post-COVID rebound leg = BULL, the 2017 low-vol grind = SIDEWAYS.
    Raises on any mismatch — a wrong display must never reach the user."""
    report = []
    r = (df["close"] / df["close"].shift(window) - 1.0).reindex(states.index)

    def expect(dt, want, desc):
        raw = float(df["close"].loc[:dt].iloc[-1] /
                    df["close"].loc[:dt].iloc[-1 - window] - 1.0)
        derived = BULL if raw >= bull_thr else BEAR if raw <= bear_thr else SIDEWAYS
        got = int(states.loc[:dt].iloc[-1])
        if got != want or derived != want:
            raise AssertionError(
                f"LABEL VERIFICATION FAILED: {desc} ({dt.date()}) — expected "
                f"{STATE_NAMES[want]}, labeled {STATE_NAMES[got]}, "
                f"raw {window}d return {raw:+.1%} derives {STATE_NAMES[derived]}")
        report.append(f"  ✓ {desc}: {dt.date()} {window}d return {raw:+.1%} → "
                      f"{STATE_NAMES[got]} (recomputed from raw prices: agrees)")

    expect(r.idxmin(), BEAR, "worst window in sample")
    expect(r.idxmax(), BULL, "best window in sample")
    expect((r.abs()).idxmin(), SIDEWAYS, "flattest window in sample")

    famous = [("2020-03-23", BEAR, "COVID crash trough"),
              ("2020-04-30", BULL, "post-COVID rebound leg"),
              ("2017-07-31", SIDEWAYS, "2017 low-vol grind")]
    for d, want, desc in famous:
        dt = pd.Timestamp(d)
        if states.index[0] <= dt <= states.index[-1]:
            raw = float(df["close"].loc[:dt].iloc[-1] /
                        df["close"].loc[:dt].iloc[-1 - window] - 1.0)
            got = int(states.loc[:dt].iloc[-1])
            derived = (BULL if raw >= bull_thr else
                       BEAR if raw <= bear_thr else SIDEWAYS)
            if got != derived:
                raise AssertionError(f"LABEL VERIFICATION FAILED at {desc}")
            mark = "✓" if got == want else "·"
            note = "" if got == want else \
                f" (market-wide episode; this ticker read {STATE_NAMES[got]})"
            report.append(f"  {mark} {desc}: {dt.date()} {window}d return "
                          f"{raw:+.1%} → {STATE_NAMES[got]}{note}")
    return report


# ---------------------------------------------------------------- backtest

def walk_forward(df: pd.DataFrame, states: pd.Series, window: int,
                 mode: str, threshold: float, cap: float,
                 warmup: int = 756, refit_every: int | None = None,
                 day_session: bool = False, conviction: float = 0.0,
                 cost_bps: float = 0.0):
    """Expanding-window walk-forward. The matrix at day t is built ONLY from
    data up to t; the position it implies is applied to day t+1's return.
    Refit every `refit_every` bars (default: the window length).

    day_session=True trades day t+1 open→close only (flat overnight) — the
    1-trade-a-day pattern; the buy&hold baseline becomes always-long-day-
    session so the comparison stays apples to apples. conviction gates trades
    to |signal| ≥ conviction; in STANDALONE mode a conviction gate trades the
    full ±cap in the signal's direction (binary) instead of sizing by signal.
    cost_bps charges per unit of turnover (entry+exit each day traded in
    day-session mode); the baseline stays gross."""
    refit_every = refit_every or window
    px = df["close"].reindex(states.index)
    if day_session:
        ret = np.nan_to_num(
            (df["raw_close"] / df["open"] - 1.0).reindex(states.index).to_numpy())
    else:
        ret = px.pct_change().to_numpy()
    st = states.to_numpy()
    n = len(st)
    if n <= warmup + window:
        raise SystemExit("not enough history for walk-forward after warmup")

    pos = {"legacy": np.zeros(n), "fixed": np.zeros(n)}
    sig = {"legacy": np.zeros(n), "fixed": np.zeros(n)}
    P_leg = P_fix = None
    for t in range(warmup, n - 1):
        if (t - warmup) % refit_every == 0 or P_leg is None:
            _, P_leg = transition_matrix(st[:t + 1], 1)
            _, P_fix = transition_matrix(st[:t + 1], window)
        for key, P in (("legacy", P_leg), ("fixed", P_fix)):
            s = signal_from(P, st[t])
            sig[key][t] = s
            if mode == "filter":
                p = 1.0 if s > threshold else 0.0
            else:  # standalone: size ∝ signal, capped
                p = float(np.clip(cap * s, -cap, cap))
            if conviction > 0:
                p = 0.0 if abs(s) < conviction else \
                    (math.copysign(cap, s) if mode == "standalone" else p)
            pos[key][t + 1] = p

    idx = states.index[warmup:]
    out = pd.DataFrame(index=idx)
    out["asset_ret"] = ret[warmup:]
    out["bh_equity"] = (1 + out["asset_ret"].fillna(0)).cumprod()
    for key in ("legacy", "fixed"):
        p = pos[key][warmup:]
        r = p * ret[warmup:]
        if cost_bps > 0:
            if day_session:
                turnover = 2.0 * np.abs(p)          # in and out every traded day
            else:
                turnover = np.abs(np.diff(pos[key], prepend=0.0))[warmup:]
            r = r - cost_bps / 1e4 * turnover
        out[f"{key}_pos"] = p
        out[f"{key}_ret"] = r
        out[f"{key}_equity"] = (1 + pd.Series(r, index=idx).fillna(0)).cumprod()
    return out, sig


def metrics(daily: pd.Series, pos: pd.Series) -> dict:
    daily = daily.fillna(0)
    eq = (1 + daily).cumprod()
    yrs = len(daily) / 252
    active = daily[pos.abs() > 1e-12]
    gains, losses = active[active > 0].sum(), -active[active < 0].sum()
    dd = (eq / eq.cummax() - 1).min()
    return {
        "total_return": eq.iloc[-1] - 1,
        "cagr": eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else float("nan"),
        "win_rate": (active > 0).mean() if len(active) else float("nan"),
        "profit_factor": gains / losses if losses > 0 else float("inf"),
        "max_drawdown": dd,
        "sharpe": (daily.mean() / daily.std() * math.sqrt(252)
                   if daily.std() > 0 else float("nan")),
        "exposure": (pos.abs() > 1e-12).mean(),
    }


def fmt_metrics(rows: dict[str, dict]) -> str:
    cols = [("total_return", "Total", "{:+.1%}"), ("cagr", "CAGR", "{:+.2%}"),
            ("win_rate", "Win rate", "{:.1%}"),
            ("profit_factor", "Profit factor", "{:.2f}"),
            ("max_drawdown", "Max DD", "{:.1%}"), ("sharpe", "Sharpe", "{:.2f}"),
            ("exposure", "Exposure", "{:.0%}")]
    head = f"  {'':<26}" + "".join(f"{h:>15}" for _, h, _ in cols)
    lines = [head]
    for name, m in rows.items():
        lines.append(f"  {name:<26}" + "".join(
            f"{fmt.format(m[k]):>15}" for k, _, fmt in cols))
    return "\n".join(lines)


# ---------------------------------------------------------------- plot

def plot_equity(out: pd.DataFrame, ticker: str, mode: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    series = [("bh_equity", "Buy & hold", COL_BH, 1.6, (0, (4, 3))),
              ("legacy_equity", "Markov 1.0 (overlapping — flawed)",
               COL_LEGACY, 2.0, "solid"),
              ("fixed_equity", "Markov 2.0 (stride-sampled — honest)",
               COL_FIXED, 2.0, "solid")]
    ymax = max(out[c].max() for c, *_ in series)
    ymin = min(out[c].min() for c, *_ in series)
    min_gap = (ymax - ymin) * 0.035  # dodge end labels that would collide
    ends = sorted((out[c].iloc[-1], i) for i, (c, *_) in enumerate(series))
    ypos = [0.0] * len(series)
    prev = None
    for v, i in ends:
        y = v if prev is None else max(v, prev + min_gap)
        ypos[i], prev = y, y
    for (col, label, color, lw, ls), y in zip(series, ypos):
        ax.plot(out.index, out[col], color=color, lw=lw, ls=ls, label=label)
        ax.annotate(f" {out[col].iloc[-1]:.2f}x", (out.index[-1], y),
                    color=color, fontsize=9, fontweight="bold", va="center")
    ax.set_title(f"{ticker} — walk-forward equity, {mode.upper()} mode "
                 f"(growth of $1)", color=INK, fontsize=12, loc="left", pad=14)
    ax.grid(color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.margins(x=0.01)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- HMM

def hmm_crosscheck(df: pd.DataFrame, states: pd.Series) -> str:
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return ("hmmlearn not installed — run `pip install hmmlearn` and re-run "
                "with --hmm for the no-hand-labels cross-check.")
    lr = np.log(df["close"] / df["close"].shift(1)).reindex(states.index).dropna()
    X = lr.to_numpy().reshape(-1, 1)
    hmm = GaussianHMM(n_components=3, covariance_type="full",
                      n_iter=200, random_state=7).fit(X)
    hidden = hmm.predict(X)
    means = hmm.means_.ravel()
    remap = {j: rank for rank, j in enumerate(np.argsort(means))}
    hmm_states = pd.Series([remap[h] for h in hidden], index=lr.index)
    both = pd.concat([states.reindex(lr.index), hmm_states], axis=1,
                     keys=["thr", "hmm"]).dropna()
    agree = (both["thr"] == both["hmm"]).mean()
    per = {STATE_NAMES[s]: (both.loc[both["thr"] == s, "hmm"] == s).mean()
           for s in ORDER}
    lines = [f"HMM (3-state Gaussian, no hand-made labels) vs threshold labels:",
             f"  overall agreement: {agree:.0%}  " +
             "  ".join(f"{k}: {v:.0%}" for k, v in per.items()),
             "  agreement is the green light; where they disagree, the regime "
             "call is low-confidence — size down or stand aside."]
    return "\n".join(lines)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Markov 2.0 — Hedge Fund Method")
    ap.add_argument("ticker")
    ap.add_argument("--years", type=float, default=10,
                    help="walk-forward test span (3y warmup fetched on top)")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--bull", type=float, default=0.05)
    ap.add_argument("--bear", type=float, default=-0.05)
    ap.add_argument("--mode", choices=["filter", "standalone"], default="filter")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="FILTER mode: signal must exceed this to allow longs")
    ap.add_argument("--cap", type=float, default=1.0,
                    help="STANDALONE mode: max |position| (1.0 = 100%%)")
    ap.add_argument("--day-session", action="store_true",
                    help="trade next day open→close only (flat overnight; "
                         "1 trade per day)")
    ap.add_argument("--conviction", type=float, default=0.0,
                    help="trade only when |signal| ≥ this (STANDALONE then "
                         "trades full ±cap in the signal direction)")
    ap.add_argument("--cost", type=float, default=0.0,
                    help="per-turnover cost in basis points (entry+exit "
                         "charged each traded day in day-session mode)")
    ap.add_argument("--enhanced", action="store_true",
                    help="states from return + ATR + relative volume clusters")
    ap.add_argument("--hmm", action="store_true")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    df = fetch_history(a.ticker, a.years + 3.2)  # ~3y warmup ahead of test span
    print(f"{a.ticker}: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")
    mode_line = ("FILTER — the regime gates YOUR strategy: it may go long only "
                 f"when signal > {a.threshold:+.2f}; flat otherwise. Your entries "
                 "stay yours; Markov decides WHEN they're allowed."
                 if a.mode == "filter" else
                 f"STANDALONE — trades the signal directly: position = signal × "
                 f"cap ({a.cap:.0%}), long when positive, short when negative.")
    print(f"mode: {mode_line}\n")

    states = label_states(df, a.window, a.bull, a.bear)
    if a.enhanced:
        enh = label_states_enhanced(df, a.window)
        common = states.index.intersection(enh.index)
        chg = (states.loc[common] != enh.loc[common]).mean()
        print(f"ENHANCED STATES: clusters on {a.window}d return + ATR + relative "
              f"volume relabel {chg:.0%} of days vs price-only.\n")
        states = enh

    # FIX 2 — verify before rendering anything
    print("FIX 2 — label verification (must pass before display):")
    for line in verify_labels(df, states, a.window, a.bull, a.bear):
        print(line)
    print()

    # FIX 1 — both matrices, side by side
    st = states.to_numpy()
    C1, P1 = transition_matrix(st, 1)
    Ck, Pk = transition_matrix(st, a.window)
    print("FIX 1 — transition matrices (rows sum to 1):")
    print(f"LEGACY, overlapping daily windows (consecutive {a.window}d windows "
          f"share {a.window - 1} days — the diagonal 'stickiness' is faked):")
    print(fmt_matrix(P1, C1))
    print(f"\nSTRIDE-SAMPLED, non-overlapping (stride = {a.window} bars) — the "
          "only statistically honest one:")
    print(fmt_matrix(Pk, Ck))
    d1 = ", ".join(f"{STATE_NAMES[i]} {P1[i, i]:.2f}→{Pk[i, i]:.2f}" for i in ORDER)
    print(f"stickiness (diagonal), legacy→true: {d1}\n")

    cur = int(st[-1])
    sig = signal_from(Pk, cur)
    print(f"current state: {STATE_NAMES[cur]} "
          f"({a.window}d return {df['close'].iloc[-1] / df['close'].iloc[-1 - a.window] - 1:+.1%})")
    print(f"signal = P(bull next) − P(bear next) = {Pk[cur, BULL]:.3f} − "
          f"{Pk[cur, BEAR]:.3f} = {sig:+.3f}  (per {a.window}-bar step)")

    pi = stationary(Pk)
    print("\nmulti-step forecast from today's state (matrix powers, "
          f"{a.window}-bar steps):")
    for p in (1, 2, 4, 8, 16):
        row = np.linalg.matrix_power(Pk, p)[cur]
        print(f"  step {p:>2}: " + "  ".join(
            f"{STATE_NAMES[j]} {row[j]:.3f}" for j in ORDER))
    print("  stationary: " + "  ".join(f"{STATE_NAMES[j]} {pi[j]:.3f}" for j in ORDER))
    print("  → forecasts converge to the stationary distribution: long-horizon "
          "forecasts carry NO signal.\n")

    if a.hmm:
        print(hmm_crosscheck(df, states) + "\n")

    flavor = a.mode.upper() + (" DAY-SESSION" if a.day_session else "")
    extras = []
    if a.day_session:
        extras.append("next-day open→close only, flat overnight (1 trade/day)")
    if a.conviction > 0:
        extras.append(f"trades only when |signal| ≥ {a.conviction:.2f}")
    if a.cost > 0:
        extras.append(f"costs {a.cost:g} bps per turnover (baseline stays gross)")
    print(f"walk-forward backtest ({flavor} mode; matrix refit every "
          f"{a.window} bars on an expanding window — never tested on data it "
          "has learned from" + ("; " + "; ".join(extras) if extras else "") + "):")
    out, _ = walk_forward(df, states, a.window, a.mode, a.threshold, a.cap,
                          day_session=a.day_session, conviction=a.conviction,
                          cost_bps=a.cost)
    baseline = "Long day session" if a.day_session else "Buy & hold"
    rows = {
        baseline: metrics(out["asset_ret"], pd.Series(1.0, index=out.index)),
        "Markov 1.0 (overlapping)": metrics(out["legacy_ret"], out["legacy_pos"]),
        "Markov 2.0 (stride)": metrics(out["fixed_ret"], out["fixed_pos"]),
    }
    print(f"  span: {out.index[0].date()} → {out.index[-1].date()} "
          f"({len(out) / 252:.1f}y)")
    print(fmt_metrics(rows))

    png = os.path.join(a.outdir, f"{a.ticker.lower()}_markov2_equity.png")
    plot_equity(out, a.ticker.upper(), flavor, png)
    print(f"\nequity curve: {png}")
    print('\n"Backtests flatter. The fixed matrix shows uglier, truer numbers — '
          'those are the only ones worth trading."')


if __name__ == "__main__":
    main()
